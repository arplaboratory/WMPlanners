from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .mpc import CEMConfig, LatentCEMCostConfig, cem_plan_latent_target, resolve_action_bounds


# ------------------------------------------------------------
# RGER memory structures
# ------------------------------------------------------------


@dataclass
class EliteState:
    """Single elite memory entry m_i = (z_i, p_i, q_i)."""

    z: torch.Tensor
    q: float
    timestep: int
    p: Optional[torch.Tensor] = None


@dataclass
class EliteMemoryConfig:
    capacity: int = 100
    top_quality_fraction: float = 0.4


class EliteMemory:
    """Top-quality + diverse latent memory for recovery waypoints."""

    def __init__(self, cfg: EliteMemoryConfig | None = None):
        self.cfg = cfg or EliteMemoryConfig()
        self.states: List[EliteState] = []

    @staticmethod
    def _canonical_latent(z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z.detach().cpu()

    @staticmethod
    def _pairwise_min_distance(candidate: torch.Tensor, selected: List[EliteState]) -> float:
        if not selected:
            return float("inf")
        d = [torch.norm(candidate - s.z, p=2).item() for s in selected]
        return float(min(d))

    def add(
        self,
        z: torch.Tensor,
        q: float,
        timestep: int,
        p: Optional[torch.Tensor] = None,
    ) -> None:
        entry = EliteState(
            z=self._canonical_latent(z),
            q=float(q),
            timestep=timestep,
            p=None if p is None else p.detach().cpu(),
        )
        self.states.append(entry)
        self._prune_to_capacity()

    def _prune_to_capacity(self) -> None:
        if len(self.states) <= self.cfg.capacity:
            self.states.sort(key=lambda x: x.q, reverse=True)
            return

        states_sorted = sorted(self.states, key=lambda x: x.q, reverse=True)

        top_n = max(1, int(self.cfg.capacity * self.cfg.top_quality_fraction))
        selected = states_sorted[:top_n]
        remaining = states_sorted[top_n:]

        # Farthest-point style fill for latent diversity.
        while len(selected) < self.cfg.capacity and remaining:
            best_idx = 0
            best_dist = -1.0
            for i, candidate in enumerate(remaining):
                min_dist = self._pairwise_min_distance(candidate.z, selected)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        selected.sort(key=lambda x: x.q, reverse=True)
        self.states = selected

    def __len__(self) -> int:
        return len(self.states)


# ------------------------------------------------------------
# RGER configs
# ------------------------------------------------------------


@dataclass
class RecoveryConfig:
    reachability_threshold: float = 0.35  # tau_r
    failure_threshold: float = 0.05  # tau_f
    progress_window: int = 20

    min_memory_size: int = 10
    top_k_memory: int = 50

    recoverability_samples: int = 64  # K
    recoverability_horizon: int = 10  # H
    success_cost_threshold: float = 1.0

    reachability_probe_population: int = 64
    reachability_probe_horizon: int = 5
    latent_distance_scale: float = 2.0

    revalidation_threshold: Optional[float] = None

    def effective_revalidation_threshold(self) -> float:
        return (
            self.revalidation_threshold
            if self.revalidation_threshold is not None
            else self.reachability_threshold
        )


@dataclass
class RecoveryStats:
    memory_size: int = 0
    reachable_count: int = 0
    evaluated_count: int = 0
    selected_recoverability: float = 0.0


@dataclass
class RecoverySelection:
    state: EliteState
    recoverability: float
    reachability: float


# ------------------------------------------------------------
# Failure detection
# ------------------------------------------------------------


def should_trigger_recovery(
    goal_distance_history: List[float],
    planner_objective_max: Optional[float],
    cfg: RecoveryConfig,
    constraint_violated: bool = False,
    planner_best_cost: Optional[float] = None,
    planner_cost_threshold: Optional[float] = None,
) -> bool:
    """Implements RGER failure triggers.

    Trigger if any of the following hold:
      1) progress stalls: the reduction in goal distance over the window,
         d(g,z_{t-k}) - d(g,z_t), is below tau_f
      2) planner cannot find good plans: max_a J(a) < tau_f
      3) explicit constraint violation
    """

    if constraint_violated:
        return True

    if planner_objective_max is not None and planner_objective_max < cfg.failure_threshold:
        return True

    if (
        planner_best_cost is not None
        and planner_cost_threshold is not None
        and planner_best_cost > planner_cost_threshold
    ):
        return True

    if len(goal_distance_history) < cfg.progress_window + 1:
        return False

    d_t = goal_distance_history[-1]
    d_prev = goal_distance_history[-(cfg.progress_window + 1)]

    # d_prev - d_t is the distance closed over the window: large and positive
    # when making good progress, small/negative when stalled or regressing.
    # (d_t - d_prev, the previous formula, is negative under normal forward
    # progress and therefore almost always satisfies "< tau_f" regardless of
    # how well the agent is actually doing.)
    return (d_prev - d_t) < cfg.failure_threshold


# ------------------------------------------------------------
# Reachability and recoverability estimation
# ------------------------------------------------------------


@torch.no_grad()
def _latent_distance_reachability(
    current_z: torch.Tensor,
    target_z: torch.Tensor,
    scale: float,
) -> float:
    if current_z.dim() == 1:
        current_z = current_z.unsqueeze(0)
    if target_z.dim() == 1:
        target_z = target_z.unsqueeze(0)

    target_z = target_z.to(current_z.device)

    d = torch.norm(current_z - target_z, p=2, dim=-1).mean()
    return float(torch.exp(-d / max(scale, 1e-6)).item())


@torch.no_grad()
def _rollout_probe_reachability(
    model,
    current_z: torch.Tensor,
    target_z: torch.Tensor,
    cem_cfg: CEMConfig,
    cfg: RecoveryConfig,
    device: torch.device,
) -> float:
    model = model.module if isinstance(model, DDP) else model

    if current_z.dim() == 1:
        current_z = current_z.unsqueeze(0)
    if target_z.dim() == 1:
        target_z = target_z.unsqueeze(0)

    pop = cfg.reachability_probe_population
    horizon = max(1, min(cfg.reachability_probe_horizon, cem_cfg.horizon))

    low, high, _, _ = resolve_action_bounds(cem_cfg, device)
    actions = low + (high - low) * torch.rand(pop, horizon, cem_cfg.action_dim, device=device)

    z_rep = current_z.to(device).repeat(pop, 1)
    z_seq = model.rollout(z_rep, actions)  # (N, H+1, z)

    target = target_z.to(device).repeat(pop, 1)
    dists = torch.norm(z_seq - target.unsqueeze(1), dim=-1)
    best_path_dist = dists.min(dim=1).values
    min_dist = best_path_dist.min()

    return float(torch.exp(-min_dist / max(cfg.latent_distance_scale, 1e-6)).item())


@torch.no_grad()
def estimate_reachability(
    model,
    current_z: torch.Tensor,
    target_z: torch.Tensor,
    cem_cfg: CEMConfig,
    cfg: RecoveryConfig,
    device: torch.device,
) -> float:
    """Estimate Reach(z_t, z_i) using latent distance + short rollout probes."""

    distance_score = _latent_distance_reachability(
        current_z=current_z,
        target_z=target_z,
        scale=cfg.latent_distance_scale,
    )

    rollout_score = _rollout_probe_reachability(
        model=model,
        current_z=current_z,
        target_z=target_z,
        cem_cfg=cem_cfg,
        cfg=cfg,
        device=device,
    )

    return 0.5 * distance_score + 0.5 * rollout_score


@torch.no_grad()
def recoverability_score(
    model,
    z: torch.Tensor,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cem_cfg: CEMConfig,
    cfg: RecoveryConfig,
    device: torch.device,
    use_optionality_metric: bool = False,
) -> float:
    """Estimate R(z_i) with JEPA imagined futures.

    Default metric:
      R(z_i) = (# successful futures) / K

    Optional metric:
      log det Cov(Z_i^H)
    """

    model = model.module if isinstance(model, DDP) else model

    if z.dim() == 1:
        z = z.unsqueeze(0)

    k = cfg.recoverability_samples
    horizon = max(1, min(cfg.recoverability_horizon, cem_cfg.horizon))

    low, high, _, _ = resolve_action_bounds(cem_cfg, device)
    actions = low + (high - low) * torch.rand(k, horizon, cem_cfg.action_dim, device=device)

    z_rep = z.to(device).repeat(k, 1)
    z_seq = model.rollout(z_rep, actions)

    if use_optionality_metric:
        endpoints = z_seq[:, -1, :]
        centered = endpoints - endpoints.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(1, endpoints.shape[0] - 1)
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6
        sign, logdet = torch.linalg.slogdet(cov)
        if sign <= 0:
            return float("-inf")
        return float(logdet.item())

    state_seq = model.state_head(z_seq)
    risk_seq = model.risk_head(z_seq).squeeze(-1)

    costs = cost_fn(state_seq, risk_seq, goal)
    success = costs < cfg.success_cost_threshold

    return float(success.float().mean().item())


# ------------------------------------------------------------
# RGER selection and revalidation
# ------------------------------------------------------------


@torch.no_grad()
def select_recovery_state(
    model,
    current_z: torch.Tensor,
    goal: torch.Tensor,
    memory: EliteMemory,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cem_cfg: CEMConfig,
    cfg: RecoveryConfig,
    device: torch.device,
    use_optionality_metric: bool = False,
) -> tuple[Optional[RecoverySelection], RecoveryStats]:
    """RGER selection pipeline.

    1) Reachability filtering over elite memory
    2) Recoverability scoring on reachable candidates
    3) Argmax recoverability state selection
    """

    stats = RecoveryStats(memory_size=len(memory))

    if len(memory) < cfg.min_memory_size:
        return None, stats

    candidates = memory.states[: cfg.top_k_memory]

    reachable: List[tuple[EliteState, float]] = []
    for candidate in candidates:
        reach = estimate_reachability(
            model=model,
            current_z=current_z,
            target_z=candidate.z,
            cem_cfg=cem_cfg,
            cfg=cfg,
            device=device,
        )
        if reach > cfg.reachability_threshold:
            reachable.append((candidate, reach))

    stats.reachable_count = len(reachable)

    if not reachable:
        return None, stats

    best: Optional[RecoverySelection] = None

    for candidate, reach in reachable:
        score = recoverability_score(
            model=model,
            z=candidate.z,
            goal=goal,
            cost_fn=cost_fn,
            cem_cfg=cem_cfg,
            cfg=cfg,
            device=device,
            use_optionality_metric=use_optionality_metric,
        )
        stats.evaluated_count += 1

        if best is None or score > best.recoverability:
            best = RecoverySelection(
                state=candidate,
                recoverability=score,
                reachability=reach,
            )

    if best is not None:
        stats.selected_recoverability = best.recoverability

    return best, stats


@torch.no_grad()
def revalidate_recovery_state(
    model,
    selected: RecoverySelection,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cem_cfg: CEMConfig,
    cfg: RecoveryConfig,
    device: torch.device,
    use_optionality_metric: bool = False,
) -> bool:
    """Step 5 in RGER: recompute R(z*) after recovery navigation."""

    score = recoverability_score(
        model=model,
        z=selected.state.z,
        goal=goal,
        cost_fn=cost_fn,
        cem_cfg=cem_cfg,
        cfg=cfg,
        device=device,
        use_optionality_metric=use_optionality_metric,
    )
    return score > cfg.effective_revalidation_threshold()


@torch.no_grad()
def plan_recovery_navigation(
    model,
    current_z: torch.Tensor,
    selected: RecoverySelection,
    cem_cfg: CEMConfig,
    device: torch.device,
    cost_cfg: LatentCEMCostConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Step 4 in RGER: use MPC to navigate from z_t to z*."""

    return cem_plan_latent_target(
        model=model,
        z0=current_z,
        target_z=selected.state.z,
        cfg=cem_cfg,
        device=device,
        cost_cfg=cost_cfg,
    )
