"""World-model public API with lazy imports for lightweight data tooling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "WorldModel": ("world_model.architecture", "WorldModel"),
    "WMOutput": ("world_model.architecture", "WMOutput"),
    "HFWindowDataset": ("world_model.data", "HFWindowDataset"),
    "SampleBatch": ("world_model.data", "SampleBatch"),
    "collate_hf": ("world_model.data", "collate_hf"),
    "TrainConfig": ("world_model.engine", "TrainConfig"),
    "rollout_loss": ("world_model.engine", "rollout_loss"),
    "CEMConfig": ("world_model.mpc", "CEMConfig"),
    "cem_plan": ("world_model.mpc", "cem_plan"),
    "seed_everything": ("world_model.utils", "seed_everything"),
    "RunningMeanStd": ("world_model.utils", "RunningMeanStd"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
