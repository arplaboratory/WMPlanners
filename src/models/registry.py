"""Deprecated – moved to :mod:`world_model.backbones.registry`."""

from world_model.backbones.base import ModelBundle
from world_model.backbones.registry import register, create_model, _BUILDERS

__all__ = ["ModelBundle", "register", "create_model", "_BUILDERS"]
