"""HuggingFace dataset loader for world-model training."""

from __future__ import annotations
from typing import List
import numpy as np
import os
import torch
import torchvision.transforms as T
import random

from torch.utils.data import Dataset
from .normalization import NormStats
from .types import SampleBatch
from typing import List
from datasets import load_from_disk, load_dataset, get_dataset_split_names
from datasets import Dataset as HFDataset
class HFWindowDataset(Dataset):
    """Load trajectory windows from a HuggingFace dataset.

    The HF dataset stores **absolute** sensor readings per frame:

    - ``episode_id`` / ``window_id`` (int64)
    - ``frame_idx`` / ``frame_idx_in_window`` (int64)
    - ``image`` (PIL Image, RGB)
    - ``odom_0..12`` — absolute odom: pos(3) + orient(4) + vel(3) + ang_vel(3)
    - ``imu_0..9`` — IMU readings
    - ``action_0..13`` — commanded actions

    **Position deltas**: The first 3 odom dims (pos_x/y/z) are converted to
    per-episode deltas relative to frame 0 at load time. This keeps values
    small and meaningful for dynamics prediction and MPC planning.

    Parameters
    ----------
    dataset : datasets.Dataset | str
        Either a loaded HF ``Dataset`` object, a path to a saved dataset on
        disk, or a HuggingFace Hub dataset ID (e.g. ``"user/dataset-name"``).
    img_size : int
        Images are resized to ``(img_size, img_size)``.
    """

    # Odom-delta column names in the HF dataset
    ODOM_COLS = ["odom_dx", "odom_dy", "odom_dz", "odom_ax", "odom_ay", "odom_az"]

    def __init__(self, dataset, img_size: int = 224,
                 action_stats=None, odom_stats=None, imu_stats=None,
                 collision_stats=None, sample_k: int = None, sample_mode: bool = False,
                 num_hist=None, num_pred=None, frameskip=None):

        """Initialize the dataset."""
        self.sample_k = sample_k
        self.sample_mode = sample_mode
        self.frameskip = frameskip if frameskip is not None else 1
    def __init__(self, dataset, img_size: int = 224, norm_stats: NormStats | None = None):
        """Load a trajectory-window dataset from a local HF directory or Hub ID."""
        from datasets import load_from_disk, load_dataset, get_dataset_split_names
        from datasets import Dataset as HFDataset

        if isinstance(dataset, str):
            dataset = dataset.strip("'\"")
            if os.path.isdir(dataset):
                print(f"[HFWindowDataset] Found local directory: {dataset}. Loading...")
                try:
                    self._ds = load_from_disk(dataset)
                except Exception as e:
                    print(f"[HFWindowDataset] Local load failed: {e}. Falling back to Hub...")
                    print(f"[HFWindowDataset] Attempting to load from Hub: {dataset}")
                    try:
                        splits = get_dataset_split_names(dataset)
                        split = "train" if "train" in splits else splits[0]
                        self._ds = load_dataset(dataset, split=split)
                    except Exception as hf_e:
                        msg = f"Failed to load dataset '{dataset}' from both local disk and Hub.\nLocal error: {e}\nHub error: {hf_e}"
                        raise RuntimeError(msg) from hf_e
            else:
                print(f"[HFWindowDataset] Attempting to load from Hub: {dataset}")
                try:
                    splits = get_dataset_split_names(dataset)
                    split = "train" if "train" in splits else splits[0]
                    self._ds = load_dataset(dataset, split=split)
                except Exception as e:
                    msg = (
                        f"Failed to load dataset '{dataset}' from Hugging Face Hub.\n"
                        f"Hub error: {e}\n"
                    )
                    raise RuntimeError(msg) from e
        elif isinstance(dataset, HFDataset):
            self._ds = dataset
        else:
            raise TypeError(f"Expected str or datasets.Dataset, got {type(dataset)}")

        self.img_size = img_size
        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self._norm_stats = norm_stats

        # Detect Odom, IMU, and Action column names and dimensions
        self.odom_cols = [c for c in self._ds.column_names if c.startswith("odom_")]
        self.imu_cols = [c for c in self._ds.column_names if c.startswith("imu_")]
        self.act_cols = [c for c in self._ds.column_names if c.startswith("action_")]

        # Handle delta-based schema (dx, dy...) vs indexed schema (odom_0, odom_1...)
        if not self.odom_cols:
            if "odom_dx" in self._ds.column_names:
                self.odom_cols = ["odom_dx", "odom_dy", "odom_dz", "odom_ax", "odom_ay", "odom_az"]
            elif "dx" in self._ds.column_names:
                self.odom_cols = ["dx", "dy", "dz", "ax", "ay", "az"]

        # Ensure we sort indexed columns numerically (odom_1, odom_2, odom_10)
        def sort_indexed(cols):
            return sorted(cols, key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else 0)

        if any("_" in c and c.split("_")[-1].isdigit() for c in self.odom_cols):
            self.odom_cols = sort_indexed(self.odom_cols)
        if any("_" in c and c.split("_")[-1].isdigit() for c in self.imu_cols):
            self.imu_cols = sort_indexed(self.imu_cols)
        if any("_" in c and c.split("_")[-1].isdigit() for c in self.act_cols):
            self.act_cols = sort_indexed(self.act_cols)

        # Fallback to defaults if still not found
        if not self.odom_cols:
            self.odom_cols = ["odom_dx", "odom_dy", "odom_dz", "odom_ax", "odom_ay", "odom_az"]

        self._window_rows = {}
        self._build_index()

        # Dimensions exposed for model construction
        self.D_odom = len(self.odom_cols)
        self.D_imu = len(self.imu_cols) or 6 # Default 6 if no IMU
        self.D_act = len(self.act_cols) or 6  # lin_vel(3) + ang_vel(3)
        self.proprio_dim = self.D_imu  # IMU only (6-dim)
        self.action_dim = self.D_act
    def force_velocity_actions(self):
        """Override action columns to use odom velocity (odom[:, 7:13]) instead."""
        self.act_cols = []
        self.D_act = 6

    def _build_index(self) -> None:
        """Group row indices into trajectories (windows/episodes)."""
        # Support both 'window_id' and 'episode_id' for backward compatibility
        if "window_id" in self._ds.column_names:
            win_col = self._ds["window_id"]
            frame_col = self._ds["frame_idx_in_window"]
        elif "episode_id" in self._ds.column_names:
            win_col = self._ds["episode_id"]
            frame_col = self._ds["frame_idx"]
        else:
            raise KeyError("Dataset must contain either 'window_id' or 'episode_id' column.")

        rows_by_win: dict[int, list[tuple[int, int]]] = {}
        for row_idx, (wid, fidx) in enumerate(zip(win_col, frame_col)):
            rows_by_win.setdefault(wid, []).append((fidx, row_idx))

        # Sort each window's rows by frame_idx
        for wid in sorted(rows_by_win):
            rows_by_win[wid].sort(key=lambda x: x[0])
            self._window_rows[wid] = [r for _, r in rows_by_win[wid]]

        self._window_ids = list(self._window_rows.keys())

    def __len__(self) -> int:
        return len(self._window_ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        wid = self._window_ids[idx]
        row_indices = self._window_rows[wid]
        
        # Use simple indexing instead of .select() for speed
        batch_data = self._ds[row_indices]
        n_frames = len(row_indices)

        # rgb
        rgb_list = []
        for img in batch_data["image"]:
            if img is not None:
                rgb_list.append(self._transform(img).numpy())
            else:
                rgb_list.append(np.zeros((3, self.img_size, self.img_size), dtype=np.float32))
        rgb = np.stack(rgb_list)  # (T, 3, H, W)

        # Odom: load raw values, then convert positions to per-episode deltas
        odom = np.zeros((n_frames, self.D_odom), dtype=np.float32)
        for i, col in enumerate(self.odom_cols):
            odom[:, i] = batch_data[col]

        # Convert absolute position (dims 0:3) to deltas relative to frame 0.
        # This keeps position values small (meters of displacement within episode)
        # instead of large absolute world coordinates.
        # Orientation (3:7), velocity (7:10), angular velocity (10:13) stay as-is.
        if self.D_odom >= 3:
            odom[:, 0:3] = odom[:, 0:3] - odom[0, 0:3]

        # actions
        if self.act_cols:
            action = np.zeros((n_frames, self.D_act), dtype=np.float32)
            for i, col in enumerate(self.act_cols):
                action[:, i] = batch_data[col]
        else:
            action = odom[:, 7:13].copy()  # lin_vel(3) + ang_vel(3)

        # imu    
        if self.imu_cols:
            imu = np.zeros((n_frames, self.D_imu), dtype=np.float32)
            for i, col in enumerate(self.imu_cols):
                imu[:, i] = batch_data[col]
        else:
            imu = np.zeros((n_frames, self.D_imu), dtype=np.float32)

        # Apply normalization if stats provided
        if self._norm_stats is not None:
            odom = self._norm_stats.normalize_odom(torch.from_numpy(odom)).numpy()
            if self.imu_cols:
                imu = self._norm_stats.normalize_imu(torch.from_numpy(imu)).numpy()
            action = self._norm_stats.normalize_action(torch.from_numpy(action)).numpy()

        # zeros
        depth = np.zeros((n_frames, 1, self.img_size, self.img_size), dtype=np.float32)
        collision = np.zeros((n_frames, 1), dtype=np.float32)

        return {
            "rgb": rgb,            # (T, 3, H, W)
            "depth": depth,        # (T, 1, H, W)
            "imu": imu,            # (T, D_imu)
            "odom": odom,          # (T, D_odom)
            "action": action,      # (T, D_act)
            "collision": collision, # (T, 1)
        }

    def compute_norm_stats(self, episode_ids: list[int] | None = None) -> NormStats:
        """Compute normalization stats over specified episodes (or all).

        Args:
            episode_ids: If provided, only compute stats from these episode IDs.
                Use this to compute stats from training split only.
        """
        if episode_ids is not None:
            wids = [w for w in self._window_ids if w in set(episode_ids)]
        else:
            wids = self._window_ids

        all_odom, all_imu, all_act = [], [], []
        for wid in wids:
            row_indices = self._window_rows[wid]
            batch_data = self._ds[row_indices]
            n = len(row_indices)

            odom = np.zeros((n, self.D_odom), dtype=np.float32)
            for i, col in enumerate(self.odom_cols):
                odom[:, i] = batch_data[col]
            # Apply same position delta conversion as __getitem__
            if self.D_odom >= 3:
                odom[:, 0:3] = odom[:, 0:3] - odom[0, 0:3]
            all_odom.append(odom)

            if self.act_cols:
                act = np.zeros((n, self.D_act), dtype=np.float32)
                for i, col in enumerate(self.act_cols):
                    act[:, i] = batch_data[col]
            else:
                act = odom[:, 7:13].copy()  # lin_vel(3) + ang_vel(3)
            all_act.append(act)

            if self.imu_cols:
                imu = np.zeros((n, self.D_imu), dtype=np.float32)
                for i, col in enumerate(self.imu_cols):
                    imu[:, i] = batch_data[col]
            else:
                imu = np.zeros((n, self.D_imu), dtype=np.float32)
            all_imu.append(imu)

        return NormStats.from_arrays(
            np.concatenate(all_odom),
            np.concatenate(all_imu),
            np.concatenate(all_act),
        )

def collate_hf(batch: List[dict[str, np.ndarray]]) -> SampleBatch:
    """Collate a list of window dicts into a :class:`SampleBatch`."""
    return SampleBatch(
        rgb=torch.from_numpy(np.stack([b["rgb"] for b in batch])),
        depth=torch.from_numpy(np.stack([b["depth"] for b in batch])),
        imu=torch.from_numpy(np.stack([b["imu"] for b in batch])),
        odom=torch.from_numpy(np.stack([b["odom"] for b in batch])),
        action=torch.from_numpy(np.stack([b["action"] for b in batch])),
        collision=torch.from_numpy(np.stack([b["collision"] for b in batch])),
    )
class EpisodeDataset(torch.utils.data.Dataset):
    """Wraps HFWindowDataset(sample_mode=False) to return full episodes as
    (obs_dict, act, state, {}) 4-tuples for use in openloop_rollout."""

    def __init__(self, hf_window_dataset: HFWindowDataset):
        self._ds = hf_window_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        d = self._ds[idx]  # dict: "rgb", "odom", "imu", "action", "collision", "state"
        odom = d["odom"] if isinstance(d["odom"], torch.Tensor) else torch.from_numpy(d["odom"])
        imu  = d["imu"]  if isinstance(d["imu"],  torch.Tensor) else torch.from_numpy(d["imu"])
        coll = d["collision"] if isinstance(d["collision"], torch.Tensor) else torch.from_numpy(d["collision"])
        obs = {
            "visual": d["rgb"] if isinstance(d["rgb"], torch.Tensor) else torch.from_numpy(d["rgb"]),
            "proprio": imu,  # IMU only (6/10-dim), consistent with collate_dinowm
        }
        act   = d["action"] if isinstance(d["action"], torch.Tensor) else torch.from_numpy(d["action"])
        state = d["state"]  if isinstance(d["state"],  torch.Tensor) else torch.from_numpy(d["state"])
        return obs, act, state, {}


def collate_dinowm(batch: List[dict[str, np.ndarray]]):
    """Collate a list of window dicts into (obs_dict, action, state) for DinoWM training."""
    if "state" not in batch[0]:
        raise KeyError("Expected 'state' key in batch for DinoWM — check hf_datasets.py.")
    rgb = torch.from_numpy(np.stack([b["rgb"] for b in batch]))             # (B, T, 3, H, W)
    proprio = torch.from_numpy(np.stack([ b["imu"] for b in batch]))
    action = torch.from_numpy(np.stack([b["action"] for b in batch]))        # (B, T, D_act)
    state = torch.stack([b["state"] for b in batch])                          # (B, T, D_state)
    obs = {"visual": rgb, "proprio": proprio}
    return obs, action, state


def _is_local_path(s: str) -> bool:
    """Check if the string refers to a local directory or file."""
    # If it exists on disk, it's local.
    if os.path.exists(s):
        return True
    # If it starts with common local path markers, it's local (even if missing/typo)
    if s.startswith("./") or s.startswith("../") or s.startswith("/"):
        return True
    # Otherwise, assume it's a Hugging Face Hub ID (e.g., "user/repo")
    return False
