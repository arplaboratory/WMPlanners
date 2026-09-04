"""Data-loading public API with lazy imports for the external Gold adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "SampleBatch": ("world_model.data.types", "SampleBatch"),
    "NPZEpisodeDataset": ("world_model.data.dataloaders", "NPZEpisodeDataset"),
    "collate_npz": ("world_model.data.dataloaders", "collate_npz"),
    "GoldWindowDataset": ("world_model.data.gold_dataset", "GoldWindowDataset"),
    "collate_gold": ("world_model.data.gold_dataset", "collate_gold"),
    "HFWindowDataset": ("world_model.data.hf_datasets", "HFWindowDataset"),
    "collate_hf": ("world_model.data.hf_datasets", "collate_hf"),
    "collate_dinowm": ("world_model.data.hf_datasets", "collate_dinowm"),
    "EpisodeDataset": ("world_model.data.hf_datasets", "EpisodeDataset"),
    "open_window_dataset": ("world_model.data.factory", "open_window_dataset"),
    "collate_for_dataset": ("world_model.data.factory", "collate_for_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
