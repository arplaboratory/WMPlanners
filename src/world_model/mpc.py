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
    action_unit_norm_dims: tuple[int, ...] | None = None
    temporal_noise: float = 0.0
    action_smoothness_weight: float = 0.0
    action_position_dims: tuple[int, ...] | None = None
    action_quaternion_dims: tuple[int, ...] | None = None


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


def validate_cem_config(cfg: CEMConfig) -> None:
    if cfg.horizon < 1 or cfg.iters < 1 or cfg.population < 1:
        raise ValueError("CEM horizon, iterations, and population must be positive")
    if not 1 <= cfg.elites <= cfg.population:
        raise ValueError("CEM elites must be between 1 and population")
    if not 0.0 <= cfg.alpha <= 1.0:
        raise ValueError("CEM alpha must be in [0, 1]")
    if not 0.0 <= cfg.temporal_noise < 1.0:
        raise ValueError("CEM temporal_noise must be in [0, 1)")
    if cfg.action_smoothness_weight < 0.0:
        raise ValueError("CEM action_smoothness_weight must be nonnegative")


def constrain_actions(actions: torch.Tensor, cfg: CEMConfig,
                      low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    actions = torch.max(torch.min(actions, high), low)
    if cfg.action_unit_norm_dims:
        dims = list(cfg.action_unit_norm_dims)
        values = actions[..., dims]
        actions[..., dims] = values / torch.linalg.vector_norm(
            values, dim=-1, keepdim=True).clamp_min(1e-6)
    return actions


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


@dataclass
class StateCEMCostConfig:
    """Cost shaping for decoded state trajectories."""

    terminal_weight: float = 1.0
    path_weight: float = 0.0
    reverse_progress_weight: float = 0.0
    trajectory_smoothness_weight: float = 0.0
    risk_weight: float = 0.0


def state_trajectory_cost(
    state_seq: torch.Tensor,
    risk_seq: torch.Tensor | None,
    goal: torch.Tensor,
    cfg: StateCEMCostConfig,
) -> torch.Tensor:
    """Score relative decoded trajectories while canceling decoder offset bias."""
    positions = state_seq[..., :3] - state_seq[:, :1, :3]
    target = goal[:, None, :3]
    terminal = torch.linalg.vector_norm(positions[:, -1] - target[:, 0], dim=-1)
    path = torch.linalg.vector_norm(positions[:, 1:] - target, dim=-1).mean(dim=-1)

    goal_direction = goal[:, :3] / torch.linalg.vector_norm(
        goal[:, :3], dim=-1, keepdim=True
    ).clamp_min(1e-6)
    increments = positions[:, 1:] - positions[:, :-1]
    signed_progress = (increments * goal_direction[:, None]).sum(dim=-1)
    reverse_progress = torch.relu(-signed_progress).mean(dim=-1)

    if positions.shape[1] >= 3:
        acceleration = positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]
        trajectory_smoothness = torch.linalg.vector_norm(
            acceleration, dim=-1
        ).mean(dim=-1)
    else:
        trajectory_smoothness = torch.zeros_like(terminal)
    risk = risk_seq.mean(dim=1) if risk_seq is not None else torch.zeros_like(terminal)
    return (
        cfg.terminal_weight * terminal
        + cfg.path_weight * path
        + cfg.reverse_progress_weight * reverse_progress
        + cfg.trajectory_smoothness_weight * trajectory_smoothness
        + cfg.risk_weight * risk
    )


def action_smoothness_cost(
    actions: torch.Tensor,
    cfg: CEMConfig,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Return pose-aware action acceleration cost for each candidate sequence."""
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    costs = []
    if cfg.action_position_dims:
        dims = list(cfg.action_position_dims)
        scale = (high[dims] - low[dims]).clamp_min(1e-3)
        positions = actions[..., dims] / scale
        acceleration = positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]
        costs.append(torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1))

    if cfg.action_quaternion_dims:
        dims = list(cfg.action_quaternion_dims)
        quaternions = actions[..., dims]
        quaternions = quaternions / torch.linalg.vector_norm(
            quaternions, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        dots = (
            (quaternions[:, 1:] * quaternions[:, :-1])
            .sum(dim=-1)
            .abs()
            .clamp(max=1.0)
        )
        angular_steps = 2.0 * torch.acos(dots)
        angular_acceleration = angular_steps[:, 1:] - angular_steps[:, :-1]
        costs.append(angular_acceleration.abs().mean(dim=-1))

    if not costs:
        scale = (high - low).clamp_min(1e-3)
        normalized = actions / scale
        acceleration = (
            normalized[:, 2:] - 2.0 * normalized[:, 1:-1] + normalized[:, :-2]
        )
        costs.append(torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1))
    return torch.stack(costs).sum(dim=0)


def sample_cem_noise(cfg: CEMConfig, device: torch.device) -> torch.Tensor:
    """Sample optionally time-correlated unit Gaussian CEM noise."""
    validate_cem_config(cfg)
    noise = torch.randn(cfg.population, cfg.horizon, cfg.action_dim, device=device)
    if cfg.temporal_noise == 0.0:
        return noise
    innovation_scale = (1.0 - cfg.temporal_noise ** 2) ** 0.5
    for step in range(1, cfg.horizon):
        noise[:, step] = (
            cfg.temporal_noise * noise[:, step - 1]
            + innovation_scale * noise[:, step]
        )
    return noise


@torch.no_grad()
def cem_plan(
    model,
    z0: torch.Tensor,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: CEMConfig,
    device: torch.device,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan actions with CEM, returning the first action and full sequence."""
    first_action, sequence, _ = cem_plan_with_cost(
        model, z0, goal, cost_fn, cfg, device, should_cancel=should_cancel
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
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """CEM planner with optional risk decoding and best-sequence cost.

    With ``use_risk_head=False``, the cost callback receives ``None`` for its
    risk argument. This supports models with an untrained risk head.
    """
    model = model.module if isinstance(model, DDP) else model

    validate_cem_config(cfg)
    assert z0.shape[0] == 1, (
        "cem_plan_with_cost currently expects batch=1 for simplicity."
    )

    low, high, init_mean, init_std = resolve_action_bounds(cfg, device)
    mean = init_mean.unsqueeze(0).repeat(cfg.horizon, 1).clone()
    std = init_std.unsqueeze(0).repeat(cfg.horizon, 1).clone()

    best_seq = None
    best_cost = None

    for _ in range(cfg.iters):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("World-model planning canceled or timed out")
        eps = sample_cem_noise(cfg, device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = constrain_actions(a, cfg, low, high)

        z0_rep = z0.repeat(cfg.population, 1)
        z_seq = model.rollout(z0_rep, a)
        state_seq = model.state_head(z_seq)
        risk_seq = model.risk_head(z_seq).squeeze(-1) if use_risk_head else None
        costs = cost_fn(state_seq, risk_seq, goal)
        if cfg.action_smoothness_weight > 0.0:
            costs = costs + cfg.action_smoothness_weight * action_smoothness_cost(
                a, cfg, low, high
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

    if should_cancel is not None and should_cancel():
        raise InterruptedError("World-model planning canceled or timed out")
    return best_seq[0], best_seq, best_cost


@torch.no_grad()
def cem_plan_latent_target(
    model,
    z0: torch.Tensor,
    target_z: torch.Tensor,
    cfg: CEMConfig,
    device: torch.device,
    cost_cfg: LatentCEMCostConfig | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
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

    validate_cem_config(cfg)
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
        if should_cancel is not None and should_cancel():
            raise InterruptedError("World-model planning canceled or timed out")
        eps = sample_cem_noise(cfg, device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = constrain_actions(a, cfg, low, high)

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

    if should_cancel is not None and should_cancel():
        raise InterruptedError("World-model planning canceled or timed out")
    return best_seq[0], best_seq, best_cost
