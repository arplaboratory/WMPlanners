"""HuggingFace dataset loader for world-model training."""

from __future__ import annotations
from typing import List
import numpy as np
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
        from datasets import load_from_disk, load_dataset
        from datasets import Dataset as HFDataset

        if isinstance(dataset, str):
            if _is_local_path(dataset):
                self._ds = load_from_disk(dataset)
            else:
                self._ds = load_dataset(dataset, split="train")
        elif isinstance(dataset, HFDataset):
            self._ds = dataset
        else:
            raise TypeError(f"Expected str or datasets.Dataset, got {type(dataset)}")

        self.img_size = img_size
        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # -> (3, H, W) float in [0, 1]
        ])

        # Build index: window_id -> list of row indices, sorted by frame_idx
        self._window_ids: list[int] = []
        self._window_rows: dict[int, list[int]] = {}
        self._build_index()

        # Dimensions exposed for model construction
        self.D_odom = 6   # dx, dy, dz, ax, ay, az
        self.D_imu = 6    # zeros (not available); match odom dim for symmetry
        self.D_act = 6    # odom deltas used as action

    def _build_index(self) -> None:
        """Group row indices by window_id."""
        win_col = self._ds["window_id"]
        frame_col = self._ds["frame_idx_in_window"]

        rows_by_win: dict[int, list[tuple[int, int]]] = {}
        for row_idx, (wid, fidx) in enumerate(zip(win_col, frame_col)):
            rows_by_win.setdefault(wid, []).append((fidx, row_idx))

        # Sort each window's rows by frame_idx_in_window
        for wid in sorted(rows_by_win):
            rows_by_win[wid].sort(key=lambda x: x[0])
            self._window_rows[wid] = [r for _, r in rows_by_win[wid]]

        self._window_ids = list(self._window_rows.keys())

    def __len__(self) -> int:
        return len(self._window_ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        wid = self._window_ids[idx]
        row_indices = self._window_rows[wid]
        rows = self._ds.select(row_indices)

        n_frames = len(row_indices)

        # --- RGB images ---
        rgb_list = []
        for img in rows["image"]:
            if img is not None:
                tensor = self._transform(img)  # (3, H, W)
            else:
                tensor = torch.zeros(3, self.img_size, self.img_size)
            rgb_list.append(tensor.numpy())
        rgb = np.stack(rgb_list)  # (T, 3, H, W)

        # --- Odom deltas (serve as odom state and action) ---
        odom = np.stack(
            [np.array([row[c] for c in self.ODOM_COLS], dtype=np.float32) for row in rows],
            axis=0,
        )  # (T, 6)

        # --- Fields not in HF dataset: filled with zeros ---
        depth = np.zeros((n_frames, 1, self.img_size, self.img_size), dtype=np.float32)
        imu = np.zeros((n_frames, self.D_imu), dtype=np.float32)
        collision = np.zeros((n_frames, 1), dtype=np.float32)

        return {
            "rgb": rgb,            # (T, 3, H, W)
            "depth": depth,        # (T, 1, H, W)
            "imu": imu,            # (T, D_imu)
            "odom": odom,          # (T, D_odom)
            "action": odom.copy(), # (T, D_act) — odom deltas as actions
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
    """Heuristic: if it looks like a filesystem path, treat it as local."""
    import os
    return os.path.sep in s or s.startswith(".") or os.path.exists(s)
