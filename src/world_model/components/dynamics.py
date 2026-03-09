"""Latent dynamics module: action-conditioned state transitions."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import mlp


class LatentDynamics(nn.Module):
    """Action-conditioned latent transition: ``z_{t+1} = z_t + f(z_t, a_t)``."""

    def __init__(self, z_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(z_dim + act_dim, hidden, z_dim, depth=3)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dz = self.net(torch.cat([z, a], dim=-1))
        return z + dz
