"""Prediction heads that decode from the latent state."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import mlp


class StateHead(nn.Module):
    """Decode a compact 'planning state' (e.g. position/velocity/goal) from latent *z*."""

    def __init__(self, z_dim: int, state_dim: int):
        super().__init__()
        self.net = mlp(z_dim, 256, state_dim, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiskHead(nn.Module):
    """Predict collision risk probability from latent *z*."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.net = mlp(z_dim, 256, 1, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))
