"""HuggingFace dataset loader for world-model training."""

from __future__ import annotations
import numpy as np
import os
import torch
import torchvision.transforms as T
import random

from torch.utils.data import Dataset
from .types import SampleBatch
from typing import List
from datasets import load_from_disk, load_dataset, get_dataset_split_names
from datasets import Dataset as HFDataset
class HFWindowDataset(Dataset):
    """Load trajectory windows from a HuggingFace dataset.

    The HF dataset is expected to have the schema produced by the data
    curation pipeline (``scripts/world_model/data_pipeline/hf_upload.py``):

    - ``window_id`` (int64)
    - ``frame_idx_in_window`` (int64)
    - ``image`` (PIL Image, RGB)
    - ``odom_dx, odom_dy, odom_dz`` (float64) — body-frame position deltas
    - ``odom_ax, odom_ay, odom_az`` (float64) — body-frame rotation deltas

    Each row is one frame; rows with the same ``window_id`` form a window.
    The 6-DOF odom deltas are used as both the proprioceptive state and the
    action signal.  Fields not present in the HF dataset (depth, IMU,
    collision) are filled with zeros.

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

        if isinstance(dataset, str):
            dataset = dataset.strip("'\"")
            if os.path.isdir(dataset):
                print(f"[HFWindowDataset] Found local directory: {dataset}. Loading...")
                try:
                    self._ds = load_from_disk(dataset)
                except EnvironmentError as e:
                    print(f"[HFWindowDataset] Local load failed: {e}. Falling back to Hub...")
                    print(f"[HFWindowDataset] Attempting to load from Hub: {dataset}")
                    try:
                        splits = get_dataset_split_names(dataset)
                        split = "train" if "train" in splits else splits[0]
                        self._ds = load_dataset(dataset, split=split)
                    except Exception as hf_e:
                        msg = f"Failed to load dataset '{dataset}' \
                            from both local disk and Hub.\nLocal error: {e} \
                            \nHub error: {hf_e}"
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
        # Store normalization/scaling stats; convert OmegaConf ListConfig to numpy
        def _to_numpy_stats(stats):
            if stats is None:
                return None
            return {k: np.array(v, dtype=np.float32) for k, v in stats.items()}

        self.odom_stats = _to_numpy_stats(odom_stats)
        self.imu_stats = _to_numpy_stats(imu_stats)
        self.collision_stats = _to_numpy_stats(collision_stats)
        self.action_stats = _to_numpy_stats(action_stats)

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
            return sorted(cols,
                          key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else 0)

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
        self.D_act = len(self.act_cols) or self.D_odom
        # Derived dims used by the trainer
        self.proprio_dim = self.D_imu  # IMU only (6-dim)
        self.action_dim = self.D_act

    def _build_index(self) -> None:
        """Group row indices into trajectories (windows/episodes)."""
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
        batch_data = self._ds[row_indices]
        n_frames = len(row_indices)

        try:
            if self.sample_mode and self.sample_k is not None and n_frames > 0:
                k = self.sample_k
                fs = self.frameskip
                span = (k - 1) * fs + 1  # raw frames needed to cover k steps at stride fs
                if n_frames >= span:
                    start = np.random.randint(0, n_frames - span + 1)
                    indices = np.arange(start, start + span, fs)
                else:
                    # Not enough frames for full stride; take what's available then pad
                    avail = np.arange(0, n_frames, fs)
                    if len(avail) >= k:
                        indices = avail[:k]
                    else:
                        indices = np.concatenate([avail, np.full(k - len(avail), n_frames - 1)])

                def get_frame(i):
                    img = batch_data["image"][i]
                    if img is not None:
                        rgb = self._transform(img).numpy()
                    else:
                        rgb = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
                    odom = np.array([batch_data[col][i] for col in self.odom_cols], dtype=np.float32)
                    if self.act_cols:
                        action = np.array([batch_data[col][i] for col in self.act_cols], dtype=np.float32)
                    else:
                        action = odom.copy()
                    if self.imu_cols:
                        imu = np.array([batch_data[col][i] for col in self.imu_cols], dtype=np.float32)
                    else:
                        imu = np.zeros((self.D_imu,), dtype=np.float32)
                    collision = np.zeros((1,), dtype=np.float32)
                    depth = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
                    return rgb, odom, imu, action, collision, depth

                rgbs, odoms, imus, actions, collisions, depths = [], [], [], [], [], []
                for i in indices:
                    rgb, odom, imu, action, collision, depth = get_frame(i)
                    rgbs.append(rgb)
                    odoms.append(odom)
                    imus.append(imu)
                    actions.append(action)
                    collisions.append(collision)
                    depths.append(depth)

                rgbs = np.stack(rgbs)
                odoms = np.stack(odoms)
                imus = np.stack(imus)
                actions = np.stack(actions)
                collisions = np.stack(collisions)
                depths = np.stack(depths)

                # Derive action as delta to goal: action[t] = odom[-1] - odom[t]
                if not self.act_cols:
                    actions = odoms[-1:] - odoms  # broadcast subtract last frame

                # Normalize/scale; always keep raw copies for the state vector
                state_odom = odoms.copy()
                state_imu = imus.copy()
                state_collision = collisions.copy()
                if self.odom_stats is not None:
                    if "mean" in self.odom_stats and "std" in self.odom_stats:
                        odoms = (odoms - self.odom_stats["mean"]) / (self.odom_stats["std"] + 1e-8)
                    elif "min" in self.odom_stats and "max" in self.odom_stats:
                        odoms = (odoms - self.odom_stats["min"]) / (self.odom_stats["max"] - self.odom_stats["min"] + 1e-8)
                if self.imu_stats is not None:
                    if "mean" in self.imu_stats and "std" in self.imu_stats:
                        imus = (imus - self.imu_stats["mean"]) / (self.imu_stats["std"] + 1e-8)
                    elif "min" in self.imu_stats and "max" in self.imu_stats:
                        imus = (imus - self.imu_stats["min"]) / (self.imu_stats["max"] - self.imu_stats["min"] + 1e-8)
                if self.collision_stats is not None:
                    if "mean" in self.collision_stats and "std" in self.collision_stats:
                        collisions = (collisions - self.collision_stats["mean"]) / (self.collision_stats["std"] + 1e-8)
                    elif "min" in self.collision_stats and "max" in self.collision_stats:
                        collisions = (collisions - self.collision_stats["min"]) / (self.collision_stats["max"] - self.collision_stats["min"] + 1e-8)
                if self.action_stats is not None:
                    if "mean" in self.action_stats and "std" in self.action_stats:
                        actions = (actions - self.action_stats["mean"]) / (self.action_stats["std"] + 1e-8)
                    elif "min" in self.action_stats and "max" in self.action_stats:
                        actions = (actions - self.action_stats["min"]) / (self.action_stats["max"] - self.action_stats["min"] + 1e-8)

                output = {
                    "rgb": rgbs,            # (k, 3, H, W)
                    "depth": depths,        # (k, 1, H, W)
                    "imu": imus,            # (k, D_imu)
                    "odom": odoms,          # (k, D_odom)
                    "action": actions,      # (k, D_act)
                    "collision": collisions, # (k, 1)
                    "state": torch.from_numpy(np.concatenate([state_odom, state_imu, state_collision], axis=-1)),  # (k, D_odom+D_imu+1)
                }
                return output
            # Default: return all frames in the window (non-sample_mode)
            images = batch_data["image"]
            rgbs = np.stack([self._transform(img).numpy() if img is not None else np.zeros((3, self.img_size, self.img_size), dtype=np.float32) for img in images])
            odoms = np.stack([[batch_data[col][i] for col in self.odom_cols] for i in range(n_frames)]).astype(np.float32)
            if self.act_cols:
                actions = np.stack([[batch_data[col][i] for col in self.act_cols] for i in range(n_frames)]).astype(np.float32)
            else:
                # Derive action as delta to goal: action[t] = odom[-1] - odom[t]
                actions = odoms[-1:] - odoms  # broadcast subtract last frame
            imus = np.stack([[batch_data[col][i] for col in self.imu_cols] for i in range(n_frames)]).astype(np.float32) if self.imu_cols else np.zeros((n_frames, self.D_imu), dtype=np.float32)
            collisions = np.zeros((n_frames, 1), dtype=np.float32)
            depths = np.zeros((n_frames, 1, self.img_size, self.img_size), dtype=np.float32)
            state_odom = odoms.copy()
            state_imu = imus.copy()
            state_collision = collisions.copy()
            if self.odom_stats is not None:
                if "mean" in self.odom_stats and "std" in self.odom_stats:
                    odoms = (odoms - self.odom_stats["mean"]) / (self.odom_stats["std"] + 1e-8)
                elif "min" in self.odom_stats and "max" in self.odom_stats:
                    odoms = (odoms - self.odom_stats["min"]) / (self.odom_stats["max"] - self.odom_stats["min"] + 1e-8)
            if self.imu_stats is not None:
                if "mean" in self.imu_stats and "std" in self.imu_stats:
                    imus = (imus - self.imu_stats["mean"]) / (self.imu_stats["std"] + 1e-8)
                elif "min" in self.imu_stats and "max" in self.imu_stats:
                    imus = (imus - self.imu_stats["min"]) / (self.imu_stats["max"] - self.imu_stats["min"] + 1e-8)
            if self.action_stats is not None:
                if "mean" in self.action_stats and "std" in self.action_stats:
                    actions = (actions - self.action_stats["mean"]) / (self.action_stats["std"] + 1e-8)
                elif "min" in self.action_stats and "max" in self.action_stats:
                    actions = (actions - self.action_stats["min"]) / (self.action_stats["max"] - self.action_stats["min"] + 1e-8)
            output = {
                "rgb": rgbs,
                "depth": depths,
                "imu": imus,
                "odom": odoms,
                "action": actions,
                "collision": collisions,
                "state": torch.from_numpy(np.concatenate([state_odom, state_imu, state_collision], axis=-1)),
            }
            return output
        except Exception as e:
            raise RuntimeError(f"HFWindowDataset.__getitem__({idx}): {e}") from e

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
