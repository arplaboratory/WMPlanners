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


class RSSMDynamics(nn.Module):
    """Recurrent State-Space Model dynamics using a GRU.
    Computes h_t = GRU(concat(z_t, a_t), h_{t-1}) and z_{t+1} = prior(h_t).
    """

    def __init__(self, z_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.gru = nn.GRUCell(z_dim + act_dim, hidden)
        self.prior = mlp(hidden, hidden, z_dim, depth=2)
        self.hidden_dim = hidden

    def get_initial_state(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)
        h_next = self.gru(x, h)
        z_next = z + self.prior(h_next)  # Residual connection
        return z_next, h_next
