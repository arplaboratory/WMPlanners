"""The main WorldModel class: backbone + components stitched together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .backbones import ModelBundle, create_model as create_backbone
from .backbones.cnn import VisionEncoder
from .components import Fusion, LatentDynamics, ProprioEncoder, StateHead, RiskHead


@dataclass
class WMOutput:
    """Container for a single world-model encoding step."""

    z: torch.Tensor
    state: torch.Tensor
    risk: torch.Tensor


class WorldModel(nn.Module):
    """Modular world model with a swappable vision backbone.

    Parameters:
        backbone: Name of a registered backbone (e.g. ``"cnn"``, ``"dinov2"``,
            ``"vjepa2"``).  When ``None`` or ``"cnn"``, the built-in
            :class:`~world_model.backbones.cnn.VisionEncoder` is used.
        backbone_kwargs: Extra keyword arguments forwarded to the backbone
            builder (e.g. ``model_name``, ``device``).
        freeze_backbone: If ``True`` the backbone parameters are frozen so
            only the downstream components are trained.
    """

    def __init__(
        self,
        odom_dim: int,
        imu_dim: int,
        act_dim: int,
        z_vis_dim: int = 128,
        z_prop_dim: int = 128,
        z_dim: int = 256,
        state_dim: int = 6,
        backbone: Optional[str] = None,
        backbone_kwargs: Optional[dict] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ----- vision backbone -----
        bkw = backbone_kwargs or {}
        if backbone is None or backbone == "cnn":
            self.vision: nn.Module = VisionEncoder(in_ch=bkw.get("in_ch", 4), z_dim=z_vis_dim)
            self._vis_dim = z_vis_dim
        else:
            bundle: ModelBundle = create_backbone(backbone, **bkw)
            self.vision = bundle.model
            self._vis_dim = bundle.feature_dim or z_vis_dim
            # Adapt feature dim to z_vis_dim if they differ
            if self._vis_dim != z_vis_dim:
                self.vision_proj = nn.Linear(self._vis_dim, z_vis_dim)
            else:
                self.vision_proj = nn.Identity()

        if freeze_backbone:
            for p in self.vision.parameters():
                p.requires_grad = False

        # ----- proprioception -----
        self.proprio = ProprioEncoder(odom_dim=odom_dim, imu_dim=imu_dim, z_dim=z_prop_dim)

        # ----- fusion & dynamics -----
        self.fusion = Fusion(z_vis=z_vis_dim, z_prop=z_prop_dim, z_out=z_dim)
        self.dynamics = LatentDynamics(z_dim=z_dim, act_dim=act_dim)

        # ----- heads -----
        self.state_head = StateHead(z_dim=z_dim, state_dim=state_dim)
        self.risk_head = RiskHead(z_dim=z_dim)

    def encode(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        odom: torch.Tensor,
        imu: torch.Tensor,
    ) -> WMOutput:
        """Encode a single time-step observation into a latent state.

        Args:
            rgb: ``(B, 3, H, W)`` in ``[0, 1]``.
            depth: ``(B, 1, H, W)``.
            odom: ``(B, D_odom)``.
            imu: ``(B, D_imu)``.
        """
        x = torch.cat([rgb, depth], dim=1)
        z_vis = self.vision(x)
        if hasattr(self, "vision_proj"):
            z_vis = self.vision_proj(z_vis)
        z_prop = self.proprio(odom, imu)
        z = self.fusion(z_vis, z_prop)
        state = self.state_head(z)
        risk = self.risk_head(z)
        return WMOutput(z=z, state=state, risk=risk)

    def rollout(self, z0: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """Roll out latent dynamics for a sequence of actions.

        Args:
            z0: ``(B, z_dim)``.
            a_seq: ``(B, T, act_dim)``.

        Returns:
            ``(B, T+1, z_dim)`` including *z0*.
        """
        B, T, _ = a_seq.shape
        z = z0
        zs = [z0]
        for t in range(T):
            z = self.dynamics(z, a_seq[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)
