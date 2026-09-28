"""Optional in-memory cache for the shared Parquet window reader."""

from __future__ import annotations

from torch.utils.data import Dataset

from .parquet_dataset import ParquetWindowDataset


class CachedWindowDataset(Dataset):
    """Materialize Parquet windows once; retain the canonical sample format."""

    def __init__(self, base: ParquetWindowDataset, max_windows: int | None = None):
        if max_windows is not None and max_windows < 1:
            raise ValueError("max_windows must be positive")
        self.base = base
        self.manifest = base.manifest
        self.D_odom = base.D_odom
        self.D_imu = base.D_imu
        self.D_act = base.D_act
        self.proprio_dim = base.proprio_dim
        self.action_dim = base.action_dim
        limit = len(base) if max_windows is None else min(len(base), max_windows)
        self.windows = [base[index] for index in range(limit)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        return self.windows[index]

