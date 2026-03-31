"""HuggingFace dataset loader for world-model training."""

from __future__ import annotations
from typing import List
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .types import SampleBatch

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

    def __init__(self, dataset, img_size: int = 224):
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
                    # 2. Try Hugging Face Hub if local failed
                    print(f"[HFWindowDataset] Attempting to load from Hub: {dataset}")
                    try:
                        splits = get_dataset_split_names(dataset)
                        split = "train" if "train" in splits else splits[0]
                        self._ds = load_dataset(dataset, split=split)
                    except Exception as hf_e:
                        msg = f"Failed to load dataset '{dataset}' from both local disk and Hub.\nLocal error: {e}\nHub error: {hf_e}"
                        raise RuntimeError(msg) from hf_e
            else:
                # 2. Try Hugging Face Hub (not a local dir)
                print(f"[HFWindowDataset] Attempting to load from Hub: {dataset}")
                try:
                    splits = get_dataset_split_names(dataset)
                    split = "train" if "train" in splits else splits[0]
                    self._ds = load_dataset(dataset, split=split)
                except Exception as e:
                    msg = (
                        f"Failed to load dataset '{dataset}' from Hugging Face Hub.\n"
                        f"Hub error: {e}\n\n"
                        "TIP: If this is a private dataset, ensure you have run 'huggingface-cli login'."
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
        self.D_act = len(self.act_cols) or self.D_odom

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

        # --- RGB images ---
        rgb_list = []
        for img in batch_data["image"]:
            if img is not None:
                rgb_list.append(self._transform(img).numpy())
            else:
                rgb_list.append(np.zeros((3, self.img_size, self.img_size), dtype=np.float32))
        rgb = np.stack(rgb_list)  # (T, 3, H, W)

        # --- Odom deltas (serve as odom state and action) ---
        # Construct trajectory arrays from individual columns
        odom = np.zeros((n_frames, self.D_odom), dtype=np.float32)
        for i, col in enumerate(self.odom_cols):
            odom[:, i] = batch_data[col]

        # --- Actions ---
        if self.act_cols:
            action = np.zeros((n_frames, self.D_act), dtype=np.float32)
            for i, col in enumerate(self.act_cols):
                action[:, i] = batch_data[col]
        else:
            action = odom.copy()

        # --- IMU ---
        if self.imu_cols:
            imu = np.zeros((n_frames, self.D_imu), dtype=np.float32)
            for i, col in enumerate(self.imu_cols):
                imu[:, i] = batch_data[col]
        else:
            imu = np.zeros((n_frames, self.D_imu), dtype=np.float32)

        # --- Fields not in HF dataset: filled with zeros ---
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
