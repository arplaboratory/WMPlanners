from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class CEMConfig:
    horizon: int = 15
    iters: int = 5
    population: int = 256
    elites: int = 32
    action_dim: int = 6  # lin_vel(3) + ang_vel(3)
    action_min: float = -4.0
    action_max: float = 4.0
    init_std: float = 2.0
    alpha: float = 0.25  # smoothing for mean/std updates

    # Optional per-dimension overrides, shape (action_dim,). When set, these
    # take priority over the scalar action_min/action_max/init_std above.
    # Needed whenever the action vector mixes dimensions on very different
    # natural scales (e.g. raw absolute position alongside quaternion and
    # velocity components) -- a single scalar bound is meaningless there.
    action_low: torch.Tensor | None = None
    action_high: torch.Tensor | None = None
    action_init_mean: torch.Tensor | None = None
    action_init_std: torch.Tensor | None = None


def resolve_action_bounds(cfg: CEMConfig, device: torch.device):
    """Return (low, high, init_mean, init_std) tensors of shape (action_dim,).

    Falls back to broadcasting the scalar action_min/action_max/init_std
    fields when the per-dimension overrides aren't set.
    """
    low = (
        cfg.action_low.to(device)
        if cfg.action_low is not None
        else torch.full((cfg.action_dim,), cfg.action_min, device=device)
    )
    high = (
        cfg.action_high.to(device)
        if cfg.action_high is not None
        else torch.full((cfg.action_dim,), cfg.action_max, device=device)
    )
    init_mean = (
        cfg.action_init_mean.to(device)
        if cfg.action_init_mean is not None
        else torch.zeros(cfg.action_dim, device=device)
    )
    init_std = (
        cfg.action_init_std.to(device)
        if cfg.action_init_std is not None
        else torch.full((cfg.action_dim,), cfg.init_std, device=device)
    )
    return low, high, init_mean, init_std


def compute_action_bounds(dataset, device: torch.device):
    """Per-dimension action mean/std/min/max computed from a dataset.

    The action vector can mix wildly different scales (e.g. raw absolute
    position in the hundreds alongside quaternion components in [-1, 1] and
    velocities in the tens), so a single scalar CEM bound is meaningless.
    Bounds are the observed per-dim min/max; init mean/std are the observed
    per-dim mean/std, so CEM starts its search centered on the real data.

    Bounds must come from ``dataset[i]`` (the ``__getitem__`` path), not the
    raw underlying columns (``dataset._ds[...]``): position dims are
    converted to per-episode deltas relative to each window's frame 0 inside
    ``__getitem__``, so reading raw columns would give CEM a search space in
    absolute world coordinates while the eval target it's scored against is
    on the small delta scale -- silently making the plan and the target
    incomparable.
    """
    all_actions = [dataset[i]["action"] for i in range(len(dataset))]
    arr = np.concatenate(all_actions, axis=0)  # (total_frames, act_dim)
    mean = torch.from_numpy(arr.mean(axis=0)).to(device)
    std = torch.from_numpy(arr.std(axis=0)).clamp_min(1e-3).to(device)
    low = torch.from_numpy(arr.min(axis=0)).to(device)
    high = torch.from_numpy(arr.max(axis=0)).to(device)
    return mean, std, low, high


@dataclass
class LatentCEMCostConfig:
    """Cost shaping for latent-target CEM planning."""

    terminal_weight: float = 1.0
    path_weight: float = 0.1


@torch.no_grad()
def cem_plan(
    model,
    z0: torch.Tensor,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: CEMConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan actions with CEM, returning the first action and full sequence."""
    first_action, sequence, _ = cem_plan_with_cost(
        model, z0, goal, cost_fn, cfg, device
    )
    return first_action, sequence


@torch.no_grad()
def cem_plan_with_cost(
    model,
    z0: torch.Tensor,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: CEMConfig,
    device: torch.device,
    *,
    use_risk_head: bool = True,
    state_decoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """CEM planner with optional risk decoding and best-sequence cost.

    With ``use_risk_head=False``, the cost callback receives ``None`` for its
    risk argument. This supports models with an untrained risk head.
    ``state_decoder`` replaces ``model.state_head`` for decoding rollouts, e.g.
    a :class:`~world_model.state_decoder.PoseDecoder` fit to a model trained
    without a state loss.
    """
    model = model.module if isinstance(model, DDP) else model

    assert cfg.iters > 0, "CEMConfig.iters must be > 0"
    assert z0.shape[0] == 1, (
        "cem_plan_with_cost currently expects batch=1 for simplicity."
    )

    low, high, init_mean, init_std = resolve_action_bounds(cfg, device)
    mean = init_mean.unsqueeze(0).repeat(cfg.horizon, 1).clone()
    std = init_std.unsqueeze(0).repeat(cfg.horizon, 1).clone()

    best_seq = None
    best_cost = None

    for _ in range(cfg.iters):
        eps = torch.randn(cfg.population, cfg.horizon, cfg.action_dim, device=device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = torch.max(torch.min(a, high), low)

        z0_rep = z0.repeat(cfg.population, 1)
        z_seq = model.rollout(z0_rep, a)
        state_seq = (state_decoder or model.state_head)(z_seq)
        risk_seq = model.risk_head(z_seq).squeeze(-1) if use_risk_head else None
        costs = cost_fn(state_seq, risk_seq, goal)

        elite_idx = torch.topk(costs, k=cfg.elites, largest=False).indices
        elite_a = a[elite_idx]

        elite_mean = elite_a.mean(dim=0)
        elite_std = elite_a.std(dim=0).clamp_min(1e-3)

        mean = (1 - cfg.alpha) * mean + cfg.alpha * elite_mean
        std = (1 - cfg.alpha) * std + cfg.alpha * elite_std

        min_cost, min_idx = costs.min(dim=0)
        if best_cost is None or min_cost.item() < best_cost:
            best_cost = float(min_cost.item())
            best_seq = a[min_idx].detach()

    return best_seq[0], best_seq, best_cost


@torch.no_grad()
def cem_plan_latent_target(
    model,
    z0: torch.Tensor,
    target_z: torch.Tensor,
    cfg: CEMConfig,
    device: torch.device,
    cost_cfg: LatentCEMCostConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """CEM planner that directly optimizes latent rollouts toward a target latent.

    Inputs:
      model: WorldModel with .rollout()
      z0: (1, z_dim)
      target_z: (1, z_dim) or (z_dim,)
      cfg: CEMConfig
      cost_cfg: latent cost shaping config

    Returns:
      best_action: (action_dim,) first action of best sequence
      best_seq:    (horizon, action_dim)
      best_cost:   float
    """
    model = model.module if isinstance(model, DDP) else model

    assert cfg.iters > 0, "CEMConfig.iters must be > 0"
    assert z0.shape[0] == 1, "cem_plan_latent_target currently expects batch=1."

    if target_z.dim() == 1:
        target_z = target_z.unsqueeze(0)

    target_z = target_z.to(device)

    if cost_cfg is None:
        cost_cfg = LatentCEMCostConfig()

    low, high, init_mean, init_std = resolve_action_bounds(cfg, device)
    mean = init_mean.unsqueeze(0).repeat(cfg.horizon, 1).clone()
    std = init_std.unsqueeze(0).repeat(cfg.horizon, 1).clone()

    best_seq = None
    best_cost = None

    for _ in range(cfg.iters):
        eps = torch.randn(cfg.population, cfg.horizon, cfg.action_dim, device=device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = torch.max(torch.min(a, high), low)

        z0_rep = z0.repeat(cfg.population, 1)
        z_seq = model.rollout(z0_rep, a)  # (N, H+1, z)

        target = target_z.repeat(cfg.population, 1)
        terminal_dist = torch.norm(z_seq[:, -1, :] - target, dim=-1)

        if cost_cfg.path_weight > 0:
            path_dist = torch.norm(z_seq[:, 1:, :] - target.unsqueeze(1), dim=-1).mean(
                dim=-1
            )
        else:
            path_dist = torch.zeros_like(terminal_dist)

        costs = (
            cost_cfg.terminal_weight * terminal_dist + cost_cfg.path_weight * path_dist
        )

        elite_idx = torch.topk(costs, k=cfg.elites, largest=False).indices
        elite_a = a[elite_idx]

        elite_mean = elite_a.mean(dim=0)
        elite_std = elite_a.std(dim=0).clamp_min(1e-3)

        mean = (1 - cfg.alpha) * mean + cfg.alpha * elite_mean
        std = (1 - cfg.alpha) * std + cfg.alpha * elite_std

        min_cost, min_idx = costs.min(dim=0)
        if best_cost is None or min_cost.item() < best_cost:
            best_cost = float(min_cost.item())
            best_seq = a[min_idx].detach()

    return best_seq[0], best_seq, best_cost
