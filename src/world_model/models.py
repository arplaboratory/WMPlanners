"""Backward-compatibility shim.

The world-model classes have been moved to :mod:`world_model.architecture`
and :mod:`world_model.components`.  This module re-exports the key
symbols so that existing ``from world_model.models import WorldModel``
imports continue to work.
"""

from world_model.architecture import WorldModel, WMOutput
from world_model.components import (
    Fusion,
    LatentDynamics,
    ProprioEncoder,
    StateHead,
    RiskHead,
    mlp,
)
from world_model.backbones.cnn import VisionEncoder

__all__ = [
    "WorldModel",
    "WMOutput",
    "Fusion",
    "LatentDynamics",
    "ProprioEncoder",
    "StateHead",
    "RiskHead",
    "VisionEncoder",
    "mlp",
]
