"""Fusion and proprioception encoding modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import mlp


class ProprioEncoder(nn.Module):
    """Encode odometry + IMU into a proprioceptive latent vector."""

    def __init__(self, odom_dim: int, imu_dim: int, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = mlp(odom_dim + imu_dim, hidden, z_dim, depth=3)

    def forward(self, odom: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        x = torch.cat([odom, imu], dim=-1)
        z = self.net(x)
        z = F.layer_norm(z, z.shape[-1:])
        return z


class Fusion(nn.Module):
    """Fuse visual and proprioceptive latent vectors."""

    def __init__(self, z_vis: int, z_prop: int, z_out: int):
        super().__init__()
        self.net = mlp(z_vis + z_prop, 256, z_out, depth=3)

    def forward(self, z_vis: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        # z_vis may be (B, N, D) for patch-based backbones (e.g. DINOv2); pool to (B, D)
        if z_vis.ndim == 3:
            z_vis = z_vis.mean(dim=1)
        z = self.net(torch.cat([z_vis, z_prop], dim=-1))
        z = F.layer_norm(z, z.shape[-1:])
        return z
