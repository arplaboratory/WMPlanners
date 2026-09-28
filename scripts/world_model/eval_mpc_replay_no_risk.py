"""Evaluate replay MPC without calling an untrained risk head.

This is useful for JEPA-lite checkpoints. The state-tracking score is the same
as the standard replay evaluator with its default risk weight of zero.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from world_model.mpc import CEMConfig, cem_plan_with_cost
from world_model.replay import (
    evaluate_replay,
    run_replay_cli,
    trajectory_tracking_cost,
)


def make_state_only_cost(w_goal: float = 1.0, w_smooth: float = 0.1):
    """Compatibility callback that scores decoded state trajectories only."""

    def cost_fn(state_seq: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return trajectory_tracking_cost(
            state_seq, goal, w_goal=w_goal, w_smooth=w_smooth
        )

    return cost_fn


@torch.no_grad()
def cem_plan_state_only(
    model,
    z0: torch.Tensor,
    goal: torch.Tensor,
    cfg: CEMConfig,
    device: torch.device,
    cost_fn,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
    action_init_mean: torch.Tensor | None = None,
    action_init_std: torch.Tensor | None = None,
):
    """Keep the original state-only planner API on the shared CEM loop."""
    cfg = replace(
        cfg,
        action_low=action_low if action_low is not None else cfg.action_low,
        action_high=action_high if action_high is not None else cfg.action_high,
        action_init_mean=(
            action_init_mean if action_init_mean is not None else cfg.action_init_mean
        ),
        action_init_std=(
            action_init_std if action_init_std is not None else cfg.action_init_std
        ),
    )
    return cem_plan_with_cost(
        model,
        z0,
        goal,
        lambda states, _risk, target: cost_fn(states, target),
        cfg,
        device,
        use_risk_head=False,
    )


def eval_mpc_replay_no_risk(
    model,
    dataset,
    norm_stats,
    device,
    horizon=16,
    num_episodes=50,
    cem_population=128,
    cem_iters=5,
    cem_elites=16,
    goal_success_threshold=0.1,
):
    """Run the shared replay evaluator without risk-head decoding."""
    return evaluate_replay(
        model,
        dataset,
        norm_stats,
        device,
        use_risk_head=False,
        horizon=horizon,
        num_episodes=num_episodes,
        cem_population=cem_population,
        cem_iters=cem_iters,
        cem_elites=cem_elites,
        goal_success_threshold=goal_success_threshold,
    )


def main():
    run_replay_cli(
        use_risk_head=False,
        output_name="mpc_eval_no_risk.json",
        title="MPC Replay (No Risk Head) Results",
    )


if __name__ == "__main__":
    main()
