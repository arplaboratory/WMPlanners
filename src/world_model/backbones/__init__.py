"""Interchangeable vision encoders for the world model.

Available backends are auto-registered on import:

- ``cnn``     lightweight CNN baseline  
- ``dinov2``  DINOv2 via ``external/dino``  
- ``vjepa2``  V-JEPA2 via ``external/vjepa2``
"""

from .base import BaseBackbone, ModelBundle
from .registry import create_model, register

# Auto-register all bundled backends
from . import cnn as _cnn  # noqa: F401
from . import dino as _dino  # noqa: F401
from . import jepa as _jepa  # noqa: F401

__all__ = [
    "BaseBackbone",
    "ModelBundle",
    "create_model",
    "register",
]
