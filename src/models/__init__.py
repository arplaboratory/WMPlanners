"""Backward-compatibility shim for ``src.models``.

All backbone functionality has moved to :mod:`world_model.backbones`.
This module re-exports the key symbols so that old imports such as
``from models import create_model`` continue to work.
"""

from world_model.backbones import ModelBundle, create_model, register
from world_model.backbones import _dino as dino_backend  # noqa: F401
from world_model.backbones import _jepa as jepa_backend  # noqa: F401

__all__ = ["ModelBundle", "create_model", "register"]
