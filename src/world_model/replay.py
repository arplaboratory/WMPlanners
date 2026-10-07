"""Shared checkpoint, dataset, planning, and scoring path for replay evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .architecture import WorldModel, checkpoint_backbone_kwargs
from .data import HFWindowDataset
from .data.normalization import NormStats
from .mpc import CEMConfig, cem_plan_with_cost, compute_action_bounds


def load_model_and_stats(checkpoint_path: str, device: str):
    """Restore the standard WorldModel and optional saved normalization."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    model_config = checkpoint["model_config"]
    dynamics_type = train_args.get("dynamics_type", "mlp")
    if dynamics_type == "td-jepa":
        dynamics_type = "mlp"

    model = WorldModel(
        odom_dim=model_config["odom_dim"],
        imu_dim=model_config["imu_dim"],
        act_dim=model_config["act_dim"],
        state_dim=model_config["state_dim"],
        backbone=train_args.get("backbone", "cnn"),
        backbone_kwargs=checkpoint_backbone_kwargs(checkpoint, device),
        dynamics_type=dynamics_type,
        image_size=train_args.get("img_size", 224),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    stats_path = Path(checkpoint_path).parent / "norm_stats.pt"
    norm_stats = NormStats.load(stats_path) if stats_path.exists() else None
    return model, norm_stats, train_args, model_config


def trajectory_tracking_cost(
    state_seq: torch.Tensor,
    goal: torch.Tensor,
    *,
    w_goal: float = 1.0,
    w_smooth: float = 0.1,
) -> torch.Tensor:
    """Score decoded paths against the recorded future positions."""
    population = state_seq.shape[0]
    horizon = goal.shape[1]
    predicted = state_seq[:, 1 : horizon + 1, :3]
    target = goal.expand(population, -1, -1)
    position_cost = ((predicted - target) ** 2).sum(dim=-1).mean(dim=1)

    # Two future transitions are needed to compute an acceleration term.
    if horizon >= 3:
        velocity = state_seq[:, 2 : horizon + 1, :3] - state_seq[:, 1:horizon, :3]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        smooth_cost = acceleration.square().sum(dim=-1).mean(dim=1)
    else:
        smooth_cost = torch.zeros_like(position_cost)
    return w_goal * position_cost + w_smooth * smooth_cost


def make_goal_tracking_cost(
    w_goal: float = 1.0,
    w_smooth: float = 0.1,
    risk_weight: float = 0.0,
):
    """Build the CEM callback; risk weight defaults to the historical score."""

    def cost_fn(state_seq, risk_seq, goal):
        cost = trajectory_tracking_cost(
            state_seq, goal, w_goal=w_goal, w_smooth=w_smooth
        )
        if risk_weight:
            if risk_seq is None:
                raise ValueError("risk_weight requires an enabled risk head")
            cost = cost + risk_weight * risk_seq[:, 1:].mean(dim=1)
        return cost

    return cost_fn


@torch.no_grad()
def evaluate_replay(
    model: WorldModel,
    dataset: HFWindowDataset,
    norm_stats: NormStats | None,
    device: str,
    *,
    use_risk_head: bool,
    horizon: int = 16,
    num_episodes: int = 50,
    cem_population: int = 128,
    cem_iters: int = 5,
    cem_elites: int = 16,
    goal_success_threshold: float = 0.1,
    risk_weight: float = 0.0,
):
    """Run one replay evaluation with or without risk-head decoding."""
    if risk_weight and not use_risk_head:
        raise ValueError("risk_weight requires an enabled risk head")
    device_obj = torch.device(device)
    action_mean, action_std, action_low, action_high = compute_action_bounds(
        dataset, device_obj
    )
    print(
        f"[CEM] per-dim action bounds -- min={action_low.tolist()}\n"
        f"                              max={action_high.tolist()}"
    )
    cfg = CEMConfig(
        horizon=horizon,
        iters=cem_iters,
        population=cem_population,
        elites=cem_elites,
        action_dim=dataset.D_act,
        alpha=0.25,
        action_low=action_low,
        action_high=action_high,
        action_init_mean=action_mean,
        action_init_std=action_std,
    )
    cost_fn = make_goal_tracking_cost(risk_weight=risk_weight)
    metrics = (
        "action_mse",
        "action_cosine_sim",
        "trajectory_ade",
        "planning_time_ms",
        "plan_cost",
        "goal_min_dist",
        "goal_success",
    )
    results = {name: [] for name in metrics}
    n_eval = min(num_episodes, len(dataset))
    mode = "MPC" if use_risk_head else "state-only MPC"
    print(
        f"Running {mode} replay on {n_eval} episodes "
        f"(horizon={horizon}, pop={cem_population}, iters={cem_iters})..."
    )

    for index in range(n_eval):
        sample = dataset[index]
        if sample["odom"].shape[0] < horizon + 1:
            continue
        rgb = torch.as_tensor(sample["rgb"], device=device_obj).unsqueeze(0)
        depth = torch.as_tensor(sample["depth"], device=device_obj).unsqueeze(0)
        odom = torch.as_tensor(sample["odom"], device=device_obj).unsqueeze(0)
        imu = torch.as_tensor(sample["imu"], device=device_obj).unsqueeze(0)
        action_gt = torch.as_tensor(sample["action"], device=device_obj).unsqueeze(0)

        latent = model.encode(rgb[:, 0], depth[:, 0], odom[:, 0], imu[:, 0]).z
        future_position = odom[:, 1 : horizon + 1, :3]
        start = time.perf_counter()
        _, best_sequence, best_cost = cem_plan_with_cost(
            model,
            latent,
            future_position,
            cost_fn,
            cfg,
            device_obj,
            use_risk_head=use_risk_head,
        )
        planning_ms = (time.perf_counter() - start) * 1000

        actual_first = action_gt[0, 0]
        if actual_first.numel() != dataset.D_act:
            actual_first = odom[0, 0, 7 : 7 + dataset.D_act]
        planned_first = best_sequence[0]
        action_mse = F.mse_loss(planned_first, actual_first).item()
        cosine = F.cosine_similarity(
            planned_first.unsqueeze(0), actual_first.unsqueeze(0)
        ).item()

        latent_path = model.rollout(latent, best_sequence.unsqueeze(0))
        planned_position = model.state_head(latent_path)[:, 1 : horizon + 1, :3]
        displacement = torch.linalg.vector_norm(
            planned_position - future_position, dim=-1
        )
        episode_goal = odom[:, -1, :3]
        goal_displacement = torch.linalg.vector_norm(
            planned_position - episode_goal.unsqueeze(1), dim=-1
        )
        if norm_stats is not None:
            # Preserve the historical mean-std approximation of metres.
            scale = norm_stats.odom_std[:3].to(device_obj).mean()
            displacement = displacement * scale
            goal_displacement = goal_displacement * scale
        ade = displacement.mean().item()
        goal_min_dist = goal_displacement.min().item()

        results["action_mse"].append(action_mse)
        results["action_cosine_sim"].append(cosine)
        results["trajectory_ade"].append(ade)
        results["planning_time_ms"].append(planning_ms)
        if not use_risk_head:
            results["plan_cost"].append(best_cost)
        results["goal_min_dist"].append(goal_min_dist)
        results["goal_success"].append(float(goal_min_dist <= goal_success_threshold))
        if use_risk_head and (index + 1) % 10 == 0:
            print(
                f"  [{index + 1}/{n_eval}] act_mse={action_mse:.4f} "
                f"cos_sim={cosine:.3f} traj_ade={ade:.3f}m "
                f"plan={planning_ms:.0f}ms"
            )

    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key, values in results.items()
        if values
    }


def load_replay_dataset(path, image_size, norm_stats, expected_action_dim):
    """Open and align the validation windows used by both replay CLIs."""
    dataset = HFWindowDataset(path, img_size=image_size, norm_stats=norm_stats)
    if dataset.D_act != expected_action_dim:
        if expected_action_dim == 6:
            dataset.force_velocity_actions()
        if dataset.D_act != expected_action_dim:
            raise ValueError(
                "Action dim mismatch: checkpoint expects "
                f"act_dim={expected_action_dim}, "
                f"dataset provides D_act={dataset.D_act}."
            )
    print(
        f"Using action dim: {dataset.D_act} "
        f"(checkpoint expects {expected_action_dim})"
    )
    validation_ids = {episode for episode in dataset._window_ids if episode % 5 == 0}
    dataset._window_ids = [
        episode for episode in dataset._window_ids if episode in validation_ids
    ]
    dataset._window_rows = {
        key: value
        for key, value in dataset._window_rows.items()
        if key in validation_ids
    }
    print(f"Val episodes: {len(dataset)}")
    return dataset


def run_replay_cli(*, use_risk_head: bool, output_name: str, title: str) -> None:
    """Parse common replay flags, evaluate, print, and save the result."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--cem_population", type=int, default=128)
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--cem_elites", type=int, default=16)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--goal_success_threshold", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    if use_risk_head:
        parser.add_argument(
            "--risk_weight",
            type=float,
            default=0.0,
            help="Optional collision-risk cost; default preserves past scores",
        )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, train_args, model_config = load_model_and_stats(
        args.checkpoint, args.device
    )
    image_size = args.img_size or train_args.get("img_size", 224)
    print(f"Loading dataset: {args.dataset}")
    dataset = load_replay_dataset(
        args.dataset, image_size, norm_stats, int(model_config["act_dim"])
    )
    summary = evaluate_replay(
        model,
        dataset,
        norm_stats,
        args.device,
        use_risk_head=use_risk_head,
        horizon=args.horizon,
        num_episodes=args.num_episodes,
        cem_population=args.cem_population,
        cem_iters=args.cem_iters,
        cem_elites=args.cem_elites,
        goal_success_threshold=args.goal_success_threshold,
        risk_weight=args.risk_weight if use_risk_head else 0.0,
    )

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for metric, stats in summary.items():
        print(f"  {metric}:")
        print(
            f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
            f"min={stats['min']:.4f}  max={stats['max']:.4f}"
        )
    print("=" * 60)
    if use_risk_head:
        _print_takeaways(summary)

    output = Path(args.checkpoint).parent / output_name
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}")


def _print_takeaways(summary):
    if "trajectory_ade" in summary:
        ade = summary["trajectory_ade"]["mean"]
        print(f"\n  Planned trajectory ADE: {ade:.3f}m")
        if ade < 0.5:
            print("  → Excellent — planner closely tracks expert trajectory")
        elif ade < 1.0:
            print("  → Good — planner follows general trajectory shape")
        elif ade < 2.0:
            print("  → Marginal — planner gets rough direction right")
        else:
            print("  → Poor — planner deviates significantly")
    if "planning_time_ms" in summary:
        planning_ms = summary["planning_time_ms"]["mean"]
        print(f"  Planning time: {planning_ms:.0f}ms per step")
        if planning_ms < 200:
            print("  → Real-time capable (5 Hz control loop)")
        else:
            print(f"  → Too slow for real-time (need <200ms, got {planning_ms:.0f}ms)")
    if "action_cosine_sim" in summary:
        cosine = summary["action_cosine_sim"]["mean"]
        print(f"  Action cosine similarity: {cosine:.3f}")
        if cosine > 0.8:
            print("  → Planned actions closely match expert")
        elif cosine > 0.5:
            print("  → Planned actions roughly aligned with expert")
        else:
            print("  → Planned actions diverge from expert")
