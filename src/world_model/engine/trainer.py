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

def rollout_loss(model, batch, cfg: TrainConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-step latent dynamics loss.

    batch contains sequences length T. We take a random start and roll out H steps.
    Loss terms:
      - latent prediction: z_pred vs z_target (stop-grad)
      - state prediction: decode(z_pred) vs state_target
      - risk prediction:  risk(z_pred) vs collision_target
    """
    # Unwrap DDP if needed to access sub-modules directly
    from torch.nn.parallel import DistributedDataParallel as DDP
    raw = model.module if isinstance(model, DDP) else model

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

    # Use the *first* part of sequence as start for simplicity
    t0 = 0

    # Encode all steps we need (t0..t0+H)
    z_targets = []
    state_targets = []
    risk_targets = []
    for k in range(H + 1):
        out = raw.encode(rgb[:, t0+k], depth[:, t0+k], odom[:, t0+k], imu[:, t0+k])
        z_targets.append(out.z)
        state_targets.append(odom[:, t0+k])  # Here we decode odom-like state; adapt if needed
        risk_targets.append(collision[:, t0+k])

    z_targets = torch.stack(z_targets, dim=1)  # (B,H+1,z)
    state_targets = torch.stack(state_targets, dim=1)  # (B,H+1,D_odom)
    risk_targets = torch.stack(risk_targets, dim=1)    # (B,H+1,1)

    # Rollout dynamics from z0 using actions (t0..t0+H-1)
    z0 = z_targets[:, 0]
    a_seq = action[:, t0:t0+H]  # (B,H,act)
    z_pred = raw.rollout(z0, a_seq)  # (B,H+1,z)

    # Decode state/risk
    state_pred = raw.state_head(z_pred)  # (B,H+1,state_dim)
    risk_pred = raw.risk_head(z_pred)    # (B,H+1,1)

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
