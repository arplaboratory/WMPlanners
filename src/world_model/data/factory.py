"""Schema-aware entry point for WM trajectory datasets."""

from __future__ import annotations

from typing import Any, Callable


def open_window_dataset(
    *, dataset_type: str, path: str, split: str = "train",
    image_size: int | None = None, window_size: int = 17,
) -> Any:
    """Open a canonical Gold/export dataset without deriving a new split."""

    kind = dataset_type.casefold()
    if kind == "gold":
        from .gold_dataset import GoldWindowDataset
        return GoldWindowDataset(path, split=split, img_size=image_size)
    if kind == "npz":
        from .dataloaders import NPZEpisodeDataset
        return NPZEpisodeDataset(
            path, split=split, window=window_size, img_size=image_size
        )
    if kind == "hf":
        from .hf_datasets import HFWindowDataset
        return HFWindowDataset(path, split=split, img_size=image_size or 224)
    raise ValueError(f"unsupported dataset_type {dataset_type!r}")


def collate_for_dataset(dataset: Any) -> Callable:
    from .dataloaders import NPZEpisodeDataset, collate_npz
    from .gold_dataset import GoldWindowDataset
    from .hf_datasets import HFWindowDataset, collate_hf

    if isinstance(dataset, HFWindowDataset):
        return collate_hf
    if isinstance(dataset, (GoldWindowDataset, NPZEpisodeDataset)):
        return collate_npz
    raise TypeError(f"no WM collator registered for {type(dataset).__name__}")
