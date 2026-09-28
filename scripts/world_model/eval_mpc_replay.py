"""Evaluate replay MPC with a standard WorldModel checkpoint.

By default the historical score tracks recorded positions and smoothness. Use
``--risk_weight`` to add collision risk to the planning score.
"""

from __future__ import annotations

from world_model.replay import (
    evaluate_replay,
    run_replay_cli,
)


def eval_mpc_replay(
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
    risk_weight=0.0,
):
    """Run the shared replay evaluator with risk-head decoding enabled."""
    return evaluate_replay(
        model,
        dataset,
        norm_stats,
        device,
        use_risk_head=True,
        horizon=horizon,
        num_episodes=num_episodes,
        cem_population=cem_population,
        cem_iters=cem_iters,
        cem_elites=cem_elites,
        goal_success_threshold=goal_success_threshold,
        risk_weight=risk_weight,
    )


def main():
    run_replay_cli(
        use_risk_head=True,
        output_name="mpc_eval.json",
        title="MPC Replay Evaluation Results",
    )


if __name__ == "__main__":
    main()
