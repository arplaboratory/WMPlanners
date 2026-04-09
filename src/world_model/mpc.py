from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class CEMConfig:
    horizon: int = 15
    iters: int = 5
    population: int = 256
    elites: int = 32
    action_dim: int = 2
    action_min: float = -4.0
    action_max: float = 4.0
    init_std: float = 2.0
    alpha: float = 0.25  # smoothing for mean/std updates

@torch.no_grad()
def cem_plan(
    model,
    z0: torch.Tensor,
    goal: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: CEMConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-Entropy Method (CEM) planning in action space.

    Inputs:
      model: WorldModel with .rollout() and state/risk heads
      z0:   (1, z_dim)
      goal: (1, goal_dim) (passed to cost_fn)
      cost_fn: takes (state_seq, risk_seq, goal) -> cost per sample (N,)
      cfg:  CEMConfig

    Returns:
      best_action: (action_dim,) first action of best sequence
      best_seq:    (horizon, action_dim)
    """
    model = model.module if isinstance(model, DDP) else model

    assert cfg.iters > 0, "CEMConfig.iters must be > 0"
    assert z0.shape[0] == 1, "cem_plan currently expects batch=1 for simplicity."

    mean = torch.zeros(cfg.horizon, cfg.action_dim, device=device)
    std = torch.ones(cfg.horizon, cfg.action_dim, device=device) * cfg.init_std

    best_seq = None
    best_cost = None

    for _ in range(cfg.iters):
        # Sample action sequences
        eps = torch.randn(cfg.population, cfg.horizon, cfg.action_dim, device=device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = a.clamp(cfg.action_min, cfg.action_max)

        # Rollout in latent space
        z0_rep = z0.repeat(cfg.population, 1)
        z_seq = model.rollout(z0_rep, a)  # (N, H+1, z)
        # Decode planning state and risk for each step
        state_seq = model.state_head(z_seq)          # (N, H+1, state_dim)
        risk_seq = model.risk_head(z_seq).squeeze(-1)  # (N, H+1)

        # Cost per sequence
        costs = cost_fn(state_seq, risk_seq, goal)   # (N,)
        # Select elites
        elite_idx = torch.topk(costs, k=cfg.elites, largest=False).indices
        elite_a = a[elite_idx]  # (E,H,A)

        elite_mean = elite_a.mean(dim=0)
        elite_std = elite_a.std(dim=0).clamp_min(1e-3)

        # Smooth updates
        mean = (1 - cfg.alpha) * mean + cfg.alpha * elite_mean
        std = (1 - cfg.alpha) * std + cfg.alpha * elite_std

        # Track best
        min_cost, min_idx = costs.min(dim=0)
        if best_cost is None or min_cost.item() < best_cost:
            best_cost = float(min_cost.item())
            best_seq = a[min_idx].detach()

    return best_seq[0], best_seq
