"""Backward-compatibility shim – trainer has moved to ``world_model.engine``."""

from world_model.engine.trainer import TrainConfig, rollout_loss

__all__ = ["TrainConfig", "rollout_loss"]
