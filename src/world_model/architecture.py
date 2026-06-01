"""The main WorldModel class: backbone + components stitched together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .backbones import ModelBundle, create_model as create_backbone
from .backbones.cnn import VisionEncoder
from .components import Fusion, LatentDynamics, RSSMDynamics, ProprioEncoder, StateHead, RiskHead


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
        dynamics_type: str = "mlp",
    ):
        super().__init__()

        # vision backbone
        bkw = backbone_kwargs or {}
        if backbone is None or backbone == "cnn":
            in_ch = bkw.get("in_ch", 4)
            self.vision: nn.Module = VisionEncoder(in_ch=in_ch, z_dim=z_vis_dim)
            self._vis_dim = z_vis_dim
            self._vis_in_ch = in_ch
        else:
            bundle: ModelBundle = create_backbone(backbone, **bkw)
            self.vision = bundle.model
            self._vis_dim = bundle.feature_dim or z_vis_dim
            self._vis_in_ch = bkw.get("in_ch", 3)  # Pretrained backbones default to 3
            # Adapt feature dim to z_vis_dim if they differ
            if self._vis_dim != z_vis_dim:
                self.vision_proj = nn.Linear(self._vis_dim, z_vis_dim)
            else:
                self.vision_proj = nn.Identity()

        if freeze_backbone:
            for p in self.vision.parameters():
                p.requires_grad = False

        # proprioception
        self.proprio = ProprioEncoder(odom_dim=odom_dim, imu_dim=imu_dim, z_dim=z_prop_dim)

        # fusion & dynamics
        self.fusion = Fusion(z_vis=z_vis_dim, z_prop=z_prop_dim, z_out=z_dim)
        
        if dynamics_type.lower() == "rssm":
            self.dynamics = RSSMDynamics(z_dim=z_dim, act_dim=act_dim)
        else:
            self.dynamics = LatentDynamics(z_dim=z_dim, act_dim=act_dim)

        # heads
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
        if self._vis_in_ch == 4:
            x = torch.cat([rgb, depth], dim=1)
        else:
            x = rgb
        z_vis = self.vision(x)
        if hasattr(self, "vision_proj"):
            z_vis = self.vision_proj(z_vis)
        # Pool patch-level features (ViT/DINOv2) → single vector per image
        if z_vis.ndim == 3:
            z_vis = z_vis.mean(dim=1)
        z_prop = self.proprio(odom, imu)
        z = self.fusion(z_vis, z_prop)
        state = self.state_head(z)
        risk = self.risk_head(z)
        return WMOutput(z=z, state=state, risk=risk)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        odom: torch.Tensor,
        imu: torch.Tensor,
        action: torch.Tensor,
        horizon: int,
        target_model: "WorldModel | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode targets, rollout dynamics, decode heads.

        This method exists so that :class:`DDP` can wrap the complete
        computation graph and register backward hooks for gradient sync.

        When *target_model* is ``None`` (MLP / RSSM mode), step-0 targets
        retain gradients through the backbone while steps 1..H are detached.

        When *target_model* is provided (TD-JEPA mode), **all** targets come
        from the frozen EMA model under ``no_grad``; the online backbone is
        unused and only the dynamics + heads receive gradients.

        Args:
            rgb: ``(B, T, 3, H, W)``.
            depth: ``(B, T, 1, H, W)``.
            odom: ``(B, T, D_odom)``.
            imu: ``(B, T, D_imu)``.
            action: ``(B, T, D_act)``.
            horizon: number of rollout steps *H*.  ``T >= H + 1``.
            target_model: optional frozen EMA model for TD-JEPA targets.

        Returns:
            dict with ``z_targets``, ``z_pred``, ``state_pred``, ``risk_pred``.
        """
        t0 = 0
        H = horizon

        # encode targets
        if target_model is not None:
            # TD-JEPA: all targets from frozen EMA model
            z_targets = []
            with torch.no_grad():
                for k in range(H + 1):
                    out_k = target_model.encode(
                        rgb[:, t0 + k], depth[:, t0 + k],
                        odom[:, t0 + k], imu[:, t0 + k],
                    )
                    z_targets.append(out_k.z)
        else:
            # MLP / RSSM: retains gradients, rest detached
            out0 = self.encode(rgb[:, t0], depth[:, t0], odom[:, t0], imu[:, t0])
            z_targets = [out0.z]
            with torch.no_grad():
                for k in range(1, H + 1):
                    out_k = self.encode(
                        rgb[:, t0 + k], depth[:, t0 + k],
                        odom[:, t0 + k], imu[:, t0 + k],
                    )
                    z_targets.append(out_k.z)

        z_targets = torch.stack(z_targets, dim=1)          # (B, H+1, z)

        # rollout dynamics
        z0 = z_targets[:, 0]
        a_seq = action[:, t0 : t0 + H]                     # (B, H, act)
        z_pred = self.rollout(z0, a_seq)                    # (B, H+1, z)

        # decode
        state_pred = self.state_head(z_pred)                # (B, H+1, state)
        risk_pred = self.risk_head(z_pred)                  # (B, H+1, 1)

        return {
            "z_targets": z_targets,
            "z_pred": z_pred,
            "state_pred": state_pred,
            "risk_pred": risk_pred,
        }

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
        
        # Check if the dynamics model requires a hidden state (e.g., RSSM)
        if hasattr(self.dynamics, "get_initial_state"):
            h = self.dynamics.get_initial_state(B, z0.device)
            for t in range(T):
                z, h = self.dynamics(z, a_seq[:, t], h)
                zs.append(z)
        else:
            for t in range(T):
                z = self.dynamics(z, a_seq[:, t])
                zs.append(z)
                
        return torch.stack(zs, dim=1)
