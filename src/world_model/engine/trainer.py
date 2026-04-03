from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    horizon: int = 8   # multi-step rollout horizon during training
    lambda_z: float = 1.0
    lambda_state: float = 1.0
    lambda_risk: float = 0.2

def rollout_loss(model, batch, cfg: TrainConfig, target_model=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-step latent dynamics loss.

    batch contains sequences length T. We take a random start and roll out H steps.
    Loss terms:
      - latent prediction: z_pred vs z_target (stop-grad)
      - state prediction: decode(z_pred) vs state_target
      - risk prediction:  risk(z_pred) vs collision_target

    IMPORTANT: this function calls *model(...)* (which may be a DDP wrapper)
    rather than *model.module.encode(...)* so that DDP's forward hooks fire
    and gradient allreduce works correctly.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    device = torch.device(cfg.device)
    rgb = batch.rgb.to(device)       # (B,T,3,H,W)
    depth = batch.depth.to(device)   # (B,T,1,H,W)
    imu = batch.imu.to(device)
    odom = batch.odom.to(device)
    action = batch.action.to(device)
    collision = batch.collision.to(device)  # (B,T,1)

    B, T = rgb.shape[0], rgb.shape[1]
    H = cfg.horizon
    assert T >= H + 1, f"Sequence length T={T} must be >= horizon+1={H+1}"

    t0 = 0

    # ---------- forward pass through the DDP wrapper ----------
    target_raw = target_model.module if target_model is not None and hasattr(target_model, "module") else target_model

    # Call model(...) — which is DDP.forward() when wrapped — so that DDP's
    # backward hooks are registered and gradient allreduce works correctly.
    outputs = model(
        rgb, depth, odom, imu, action,
        horizon=H, target_model=target_raw,
    )

    z_targets = outputs["z_targets"]
    z_pred = outputs["z_pred"]
    state_pred = outputs["state_pred"]
    risk_pred = outputs["risk_pred"]

    # ---------- build ground-truth targets ----------
    state_targets = torch.stack([odom[:, t0 + k] for k in range(H + 1)], dim=1)
    risk_targets = torch.stack([collision[:, t0 + k] for k in range(H + 1)], dim=1)

    # Targets: stop-grad on latent target to avoid collapse (JEPA-ish)
    z_tgt = z_targets.detach()

    loss_z = F.mse_loss(z_pred, z_tgt)

    # Map odom targets to state_dim if necessary (here we assume state_head predicts first state_dim of odom)
    state_dim = state_pred.shape[-1]
    loss_state = F.mse_loss(state_pred, state_targets[..., :state_dim])

    loss_risk = F.binary_cross_entropy(risk_pred, risk_targets)

    loss = cfg.lambda_z * loss_z + cfg.lambda_state * loss_state + cfg.lambda_risk * loss_risk

    logs = {
        "loss": float(loss.item()),
        "loss_z": float(loss_z.item()),
        "loss_state": float(loss_state.item()),
        "loss_risk": float(loss_risk.item()),
    }
    return loss, logs
