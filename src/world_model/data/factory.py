"""Schema-aware entry point for WM trajectory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def is_parquet_dataset(path: str) -> bool:
    """A local frame/window Parquet directory, with or without a manifest."""
    root = Path(str(path).strip("'\"")).expanduser()
    if root.is_file():
        root = root.parent
    return (root / "dataset_manifest.json").is_file() or (
        (root / "frames.parquet").is_file() and (root / "windows.parquet").is_file()
    )


def open_window_dataset(
    *, dataset_type: str, path: str, split: str = "train",
    image_size: int | None = None, window_size: int = 17,
    schema_config: str | None = None,
) -> Any:
    """Open Hub rows, local Parquet tables, or NPZ episodes."""

    kind = dataset_type.casefold()
    if kind == "npz":
        if schema_config is not None:
            raise ValueError("--schema_config applies only to local Parquet tables")
        from .dataloaders import NPZEpisodeDataset
        return NPZEpisodeDataset(
            path, split=split, window=window_size, img_size=image_size
        )
    if kind == "hf":
        if schema_config is not None or is_parquet_dataset(path):
            from .parquet_dataset import ParquetWindowDataset
            return ParquetWindowDataset(
                path, split=split, img_size=image_size, schema_config=schema_config,
            )
        from .hf_datasets import HFWindowDataset
        return HFWindowDataset(path, split=split, img_size=image_size or 224)
    raise ValueError(f"unsupported dataset_type {dataset_type!r}")


def collate_for_dataset(dataset: Any) -> Callable:
    from .dataloaders import NPZEpisodeDataset, collate_npz
    from .parquet_dataset import ParquetWindowDataset
    from .hf_datasets import HFWindowDataset, collate_hf
    from .cached_dataset import CachedWindowDataset

    if isinstance(dataset, HFWindowDataset):
        return collate_hf
    if isinstance(dataset, (ParquetWindowDataset, NPZEpisodeDataset, CachedWindowDataset)):
        return collate_npz
    raise TypeError(f"no WM collator registered for {type(dataset).__name__}")
