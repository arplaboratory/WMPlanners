from __future__ import annotations

import random
from dataclasses import dataclass

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
    objective: str = "standard"  # "standard" or "jepa-lite"
    jepa_mask_ratio: float = 0.0
    lambda_var: float = 0.0  # jepa-lite variance-preservation weight


def rollout_loss(
    model,
    batch,
    cfg: TrainConfig,
    target_model=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-step latent dynamics loss.

    Rolls out ``cfg.horizon`` steps from the start of each sequence window.
    Loss terms:
      - latent prediction: z_pred vs z_target (stop-grad)
      - state prediction: decode(z_pred) vs state_target
      - risk prediction:  risk(z_pred) vs collision_target
    """
    device = torch.device(cfg.device)
    rgb = batch.rgb.to(device)       # (B,T,3,H,W)
    depth = torch.nan_to_num(
        batch.depth.to(device), nan=0.0, posinf=0.0, neginf=0.0
    )  # (B,T,1,H,W)
    imu = batch.imu.to(device)
    odom = batch.odom.to(device)
    action = batch.action.to(device)
    collision = batch.collision.to(device)  # (B,T,1)
    for name, value in (("rgb", rgb), ("imu", imu), ("odom", odom), ("action", action)):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Training batch contains nonfinite {name} values")
    collision_valid = (
        batch.collision_valid.to(device)
        if batch.collision_valid is not None
        else torch.ones_like(collision, dtype=torch.bool)
    )

    # T = rgb.shape[1]
    B, T = rgb.shape[0], rgb.shape[1]
    H = cfg.horizon
    nh = cfg.num_hist
    window = nh + H  # total frames needed: history + horizon
    assert T >= window, f"Sequence length T={T} must be >= num_hist+horizon={window}"

    # Random start offset: pick a random window within the episode.
    # This gives ~(T - window) extra training signal per episode.
    # At T=50, window=17: 34 possible starting points instead of always 0.
    if model.training:
        t0 = random.randint(0, T - window)
    else:
        t0 = 0  # deterministic for validation

    # Slice all tensors to the random window
    rgb = rgb[:, t0 : t0 + window]
    depth = depth[:, t0 : t0 + window]
    imu = imu[:, t0 : t0 + window]
    odom = odom[:, t0 : t0 + window]
    action = action[:, t0 : t0 + window]
    collision = collision[:, t0 : t0 + window]

    collision_valid = collision_valid[:, t0 : t0 + window]

    # forward pass through model (now T == window, starts from 0)
    target_raw = (
        target_model.module
        if target_model is not None and hasattr(target_model, "module")
        else target_model
    )
    pred_start = nh - 1

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
    risk_targets = torch.stack(
        [collision[:, pred_start + k] for k in range(H + 1)], dim=1
    )
    risk_valid = torch.stack([
        collision_valid[:, pred_start + k] for k in range(H + 1)
    ], dim=1)

    # compute losses
    z_tgt = z_targets.detach()

    if cfg.objective == "jepa-lite" and cfg.jepa_mask_ratio > 0.0:
        # Mask rollout steps JEPA-style rather than supervising every frame.
        step_mask = (torch.rand(B, H + 1, device=device) < cfg.jepa_mask_ratio).float()
        step_mask[:, 0] = 1.0  # always include anchor step
        per_step_latent_mse = F.mse_loss(z_pred, z_tgt, reduction="none").mean(dim=-1)
        loss_z = (
            (per_step_latent_mse * step_mask).sum()
            / step_mask.sum().clamp_min(1.0)
        )
    else:
        loss_z = F.mse_loss(z_pred, z_tgt)

    if cfg.objective == "jepa-lite":
        loss_state = torch.zeros((), device=device)
        loss_risk = torch.zeros((), device=device)
    else:
        # state loss only on state dims
        state_dim = state_pred.shape[-1]
        loss_state = F.mse_loss(state_pred, state_targets[..., :state_dim])
        per_target_risk = F.binary_cross_entropy(
            risk_pred, risk_targets, reduction="none"
        )
        loss_risk = (
            (per_target_risk * risk_valid).sum()
            / risk_valid.sum().clamp_min(1)
        )

    # Variance-preservation term (VICReg-style hinge): without state/risk
    # supervision, jepa-lite's pure latent-matching objective is free to
    # collapse to a constant z (online + EMA target converge together).
    # Penalize per-dimension std of the online rollout falling below 1.
    if cfg.lambda_var > 0.0:
        z_flat = z_pred.reshape(-1, z_pred.shape[-1])
        std_z = torch.sqrt(z_flat.var(dim=0, correction=0) + 1e-4)
        loss_var = F.relu(1.0 - std_z).mean()
    else:
        loss_var = torch.zeros((), device=device)

    loss = (
        cfg.lambda_z * loss_z
        + cfg.lambda_state * loss_state
        + cfg.lambda_risk * loss_risk
        + cfg.lambda_var * loss_var
    )

    logs = {
        "loss": float(loss.item()),
        "loss_z": float(loss_z.item()),
        "loss_state": float(loss_state.item()),
        "loss_risk": float(loss_risk.item()),
        "loss_var": float(loss_var.item()),
    }
    return loss, logs
