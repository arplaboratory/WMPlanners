"""World model: observation encoding, latent dynamics, and MPC planning.

Subpackages
-----------
- :mod:`world_model.backbones` – interchangeable vision encoders
- :mod:`world_model.components` – dynamics, fusion, and prediction heads
- :mod:`world_model.data` – dataset loading utilities
- :mod:`world_model.engine` – training loop and configuration
"""

from world_model.architecture import WorldModel, WMOutput
from world_model.data import NPZEpisodeDataset, SampleBatch, collate_npz
from world_model.engine import TrainConfig, rollout_loss
from world_model.mpc import CEMConfig, cem_plan
from world_model.utils import seed_everything, RunningMeanStd

__all__ = [
    "WorldModel",
    "WMOutput",
    "NPZEpisodeDataset",
    "SampleBatch",
    "collate_npz",
    "TrainConfig",
    "rollout_loss",
    "CEMConfig",
    "cem_plan",
    "seed_everything",
    "RunningMeanStd",
]
