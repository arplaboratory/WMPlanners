"""Pose decoder trained on a frozen world model's latents.

Objectives without a state loss (jepa-lite) leave ``WorldModel.state_head``
untrained, so their latents cannot be read as poses. This head is fit after
the fact on the frozen model and maps a latent to the window-relative planar
pose ``[x, y, sin(yaw), cos(yaw)]``, which lets the planner optimize toward a
goal pose instead of a goal latent. It is trained on both encoded latents and
latents rolled out with recorded actions, since planning decodes rollouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

POSE_DIM = 4  # x, y, sin(yaw), cos(yaw)


def pose_targets(odom: np.ndarray | torch.Tensor) -> torch.Tensor:
    """``[..., x, y, sin(yaw), cos(yaw)]`` from window-relative odom ``[..., x, y, z, qx, qy, qz, qw, ...]``."""
    odom = torch.as_tensor(odom, dtype=torch.float32)
    qx, qy, qz, qw = odom[..., 3], odom[..., 4], odom[..., 5], odom[..., 6]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack([odom[..., 0], odom[..., 1], torch.sin(yaw), torch.cos(yaw)], dim=-1)


class PoseDecoder(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, POSE_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(..., z_dim)`` → ``(..., 4)`` pose ``[x, y, sin(yaw), cos(yaw)]``."""
        return self.net(z)

    def save(self, path: str | Path, **metadata) -> None:
        torch.save({"state_dict": self.state_dict(), "z_dim": self.z_dim, "hidden": self.hidden, **metadata}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PoseDecoder":
        payload = torch.load(path, map_location=device, weights_only=False)
        decoder = cls(payload["z_dim"], payload["hidden"]).to(device)
        decoder.load_state_dict(payload["state_dict"])
        decoder.metadata = {k: v for k, v in payload.items() if k not in {"state_dict", "z_dim", "hidden"}}
        return decoder.eval()


def integrate_body_velocity(actions: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Kinematic end pose ``(..., 3)`` = ``[x, y, yaw]`` from the start frame.

    ``actions``: ``(..., T, 3)`` = ``[v_forward, v_lateral, yaw_rate]`` per step,
    ``dt``: ``(T,)`` seconds. This inverts how the Parquet loader's
    ``body_velocity`` actions are computed (displacement in frame k's body
    frame over dt), so recorded actions reproduce the recorded pose; for a
    planned sequence it gives a model-free check of where the commands lead.
    """
    x = torch.zeros(actions.shape[:-2], device=actions.device)
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    for k in range(actions.shape[-2]):
        vx, vy, wz = actions[..., k, 0], actions[..., k, 1], actions[..., k, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        x = x + (vx * c - vy * s) * dt[k]
        y = y + (vx * s + vy * c) * dt[k]
        yaw = yaw + wz * dt[k]
    return torch.stack([x, y, yaw], dim=-1)
