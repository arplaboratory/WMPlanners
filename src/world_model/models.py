from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0) -> nn.Module:
    layers = []
    d = in_dim
    for i in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class VisionEncoder(nn.Module):
    """Simple CNN encoder for RGB(+depth). Replace with DINO/MAE/JEPA encoder as needed."""
    def __init__(self, in_ch: int = 4, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(128, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.net(x).flatten(1)
        z = self.proj(h)
        z = F.layer_norm(z, z.shape[-1:])
        return z

class ProprioEncoder(nn.Module):
    def __init__(self, odom_dim: int, imu_dim: int, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = mlp(odom_dim + imu_dim, hidden, z_dim, depth=3)

    def forward(self, odom: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        x = torch.cat([odom, imu], dim=-1)
        z = self.net(x)
        z = F.layer_norm(z, z.shape[-1:])
        return z

class Fusion(nn.Module):
    def __init__(self, z_vis: int, z_prop: int, z_out: int):
        super().__init__()
        self.net = mlp(z_vis + z_prop, 256, z_out, depth=3)

    def forward(self, z_vis: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        z = self.net(torch.cat([z_vis, z_prop], dim=-1))
        z = F.layer_norm(z, z.shape[-1:])
        return z

class LatentDynamics(nn.Module):
    """Action-conditioned latent transition: z_{t+1} = z_t + f(z_t, a_t)."""
    def __init__(self, z_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(z_dim + act_dim, hidden, z_dim, depth=3)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dz = self.net(torch.cat([z, a], dim=-1))
        return z + dz

class StateHead(nn.Module):
    """Decode a compact 'planning state' (e.g. position/velocity/goal) from latent z."""
    def __init__(self, z_dim: int, state_dim: int):
        super().__init__()
        self.net = mlp(z_dim, 256, state_dim, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class RiskHead(nn.Module):
    """Predict collision risk probability from latent z."""
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = mlp(z_dim, 256, 1, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))

@dataclass
class WMOutput:
    z: torch.Tensor
    state: torch.Tensor
    risk: torch.Tensor

class WorldModel(nn.Module):
    def __init__(
        self,
        odom_dim: int,
        imu_dim: int,
        act_dim: int,
        z_vis_dim: int = 128,
        z_prop_dim: int = 128,
        z_dim: int = 256,
        state_dim: int = 6,
    ):
        super().__init__()
        self.vision = VisionEncoder(in_ch=4, z_dim=z_vis_dim)
        self.proprio = ProprioEncoder(odom_dim=odom_dim, imu_dim=imu_dim, z_dim=z_prop_dim)
        self.fusion = Fusion(z_vis=z_vis_dim, z_prop=z_prop_dim, z_out=z_dim)
        self.dynamics = LatentDynamics(z_dim=z_dim, act_dim=act_dim)
        self.state_head = StateHead(z_dim=z_dim, state_dim=state_dim)
        self.risk_head = RiskHead(z_dim=z_dim)

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor, odom: torch.Tensor, imu: torch.Tensor) -> WMOutput:
        # rgb: (B,3,H,W) in [0,1], depth: (B,1,H,W)
        x = torch.cat([rgb, depth], dim=1)
        z_vis = self.vision(x)
        z_prop = self.proprio(odom, imu)
        z = self.fusion(z_vis, z_prop)
        state = self.state_head(z)
        risk = self.risk_head(z)
        return WMOutput(z=z, state=state, risk=risk)

    def rollout(self, z0: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """Roll out latent dynamics for a sequence of actions.
        z0: (B,z)
        a_seq: (B,T,act)
        returns z_seq: (B,T+1,z) including z0
        """
        B, T, _ = a_seq.shape
        z = z0
        zs = [z0]
        for t in range(T):
            z = self.dynamics(z, a_seq[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)  # (B,T+1,z)
