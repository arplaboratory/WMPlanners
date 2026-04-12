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
    num_hist: int = 1  # number of history frames (1 = original, 3 = multi-frame)
    lambda_z: float = 1.0
    lambda_state: float = 1.0
    lambda_risk: float = 0.2

def rollout_loss(
    model,
    batch,
    cfg: TrainConfig,
    target_model=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-step latent dynamics loss.

    Rolls out ``cfg.horizon`` steps from the start of each sequence window.
    Loss terms:
      - latent prediction: z_pred vs z_target (stop-grad)
      - state prediction: decode(z_pred) vs state_target
      - risk prediction:  risk(z_pred) vs collision_target
    """
    device = torch.device(cfg.device)
    rgb = batch.rgb.to(device)       # (B,T,3,H,W)
    depth = batch.depth.to(device)   # (B,T,1,H,W)
    imu = batch.imu.to(device)
    odom = batch.odom.to(device)
    action = batch.action.to(device)
    collision = batch.collision.to(device)  # (B,T,1)

    _, T = rgb.shape[0], rgb.shape[1]
    H = cfg.horizon
    nh = cfg.num_hist
    pred_start = nh - 1
    assert T >= nh + H, f"Sequence length T={T} must be >= num_hist+horizon={nh+H}"

    # forward pass through model
    target_raw = target_model.module if target_model is not None and hasattr(target_model, "module") else target_model

    outputs = model(
        rgb, depth, odom, imu, action,
        horizon=H, target_model=target_raw, num_hist=nh,
    )

    z_targets = outputs["z_targets"]
    z_pred = outputs["z_pred"]
    state_pred = outputs["state_pred"]
    risk_pred = outputs["risk_pred"]

    # build gt — targets start from pred_start (last history frame)
    state_targets = torch.stack([odom[:, pred_start + k] for k in range(H + 1)], dim=1)
    risk_targets = torch.stack([collision[:, pred_start + k] for k in range(H + 1)], dim=1)

    # compute losses
    z_tgt = z_targets.detach()

    loss_z = F.mse_loss(z_pred, z_tgt)

    # state loss only on state dims
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
