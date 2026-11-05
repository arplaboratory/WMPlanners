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
        image_size: int = 224,
    ):
        super().__init__()

        # vision backbone
        bkw = backbone_kwargs or {}
        if backbone is None or backbone == "cnn":
            in_ch = bkw.get("in_ch", 4)
            self.vision: nn.Module = VisionEncoder(
                in_ch=in_ch, z_dim=z_vis_dim, image_size=image_size,
                pool_features=bkw.get("pool_features", False),
            )
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
        num_hist: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode targets, rollout dynamics, decode heads.

        Supports multi-frame history (``num_hist``). When ``num_hist > 1``,
        the first ``num_hist`` frames are encoded with gradients and used to
        warm up the dynamics model (building RSSM hidden state). Prediction
        targets start from the last history frame onward.

        Args:
            rgb: ``(B, T, 3, H, W)``.
            depth: ``(B, T, 1, H, W)``.
            odom: ``(B, T, D_odom)``.
            imu: ``(B, T, D_imu)``.
            action: ``(B, T, D_act)``.
            horizon: number of rollout steps *H*.  ``T >= num_hist + H``.
            target_model: optional frozen EMA model for TD-JEPA targets.
            num_hist: number of history frames to encode (default 1).

        Returns:
            dict with ``z_targets``, ``z_pred``, ``state_pred``, ``risk_pred``.
            All prediction tensors have shape ``(B, H+1, ...)``.
        """
        H = horizon
        nh = num_hist
        # Prediction starts at frame (nh-1) and goes H steps forward
        pred_start = nh - 1

        # ── Encode history frames (with gradients) ──
        z_hist = []
        for k in range(nh):
            out_k = self.encode(rgb[:, k], depth[:, k], odom[:, k], imu[:, k])
            z_hist.append(out_k.z)

        # ── Encode target frames (no gradients) ──
        # Targets: frames pred_start .. pred_start+H (H+1 frames total)
        if target_model is not None:
            z_targets = []
            with torch.no_grad():
                for k in range(pred_start, pred_start + H + 1):
                    out_k = target_model.encode(
                        rgb[:, k], depth[:, k], odom[:, k], imu[:, k],
                    )
                    z_targets.append(out_k.z)
        else:
            # First target is z_hist[-1] (last history frame, has gradients)
            z_targets = [z_hist[-1]]
            with torch.no_grad():
                for k in range(pred_start + 1, pred_start + H + 1):
                    out_k = self.encode(
                        rgb[:, k], depth[:, k], odom[:, k], imu[:, k],
                    )
                    z_targets.append(out_k.z)

        z_targets = torch.stack(z_targets, dim=1)  # (B, H+1, z)

        # ── Warm up dynamics with history ──
        B = rgb.shape[0]
        h0 = None  # RSSM hidden state

        if nh > 1 and hasattr(self.dynamics, "get_initial_state"):
            # RSSM: step through history frames to build hidden state
            h = self.dynamics.get_initial_state(B, rgb.device)
            z = z_hist[0]
            for k in range(nh - 1):
                z, h = self.dynamics(z, action[:, k], h)
            h0 = h  # warm hidden state for rollout
        elif nh > 1:
            # MLP: step through history (no hidden state, but refines z)
            z = z_hist[0]
            for k in range(nh - 1):
                z = self.dynamics(z, action[:, k])

        # ── Rollout predictions from last history frame ──
        z_start = z_hist[-1]
        a_seq = action[:, pred_start : pred_start + H]  # (B, H, act)
        z_pred = self.rollout(z_start, a_seq, h0=h0)    # (B, H+1, z)

        # ── Decode ──
        state_pred = self.state_head(z_pred)   # (B, H+1, state)
        risk_pred = self.risk_head(z_pred)      # (B, H+1, 1)

        return {
            "z_targets": z_targets,
            "z_pred": z_pred,
            "state_pred": state_pred,
            "risk_pred": risk_pred,
        }

    def rollout(self, z0: torch.Tensor, a_seq: torch.Tensor, h0=None) -> torch.Tensor:
        """Roll out latent dynamics for a sequence of actions.

        Args:
            z0: ``(B, z_dim)``.
            a_seq: ``(B, T, act_dim)``.
            h0: optional pre-warmed RSSM hidden state ``(B, hidden_dim)``.
                If ``None`` and dynamics has hidden state, starts from zeros.

        Returns:
            ``(B, T+1, z_dim)`` including *z0*.
        """
        B, T, _ = a_seq.shape
        z = z0
        zs = [z0]

        if hasattr(self.dynamics, "get_initial_state"):
            h = h0 if h0 is not None else self.dynamics.get_initial_state(B, z0.device)
            for t in range(T):
                z, h = self.dynamics(z, a_seq[:, t], h)
                zs.append(z)
        else:
            for t in range(T):
                z = self.dynamics(z, a_seq[:, t])
                zs.append(z)

        return torch.stack(zs, dim=1)
