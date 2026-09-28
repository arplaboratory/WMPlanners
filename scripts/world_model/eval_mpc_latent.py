"""Latent-only MPC replay evaluation for wm_sim.

This evaluator avoids using the model's state decoder entirely. It plans in
latent space by encoding a future held-out observation as the goal latent and
optimizing action sequences with CEM to minimize terminal/path latent distance.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from world_model.architecture import WorldModel
from world_model.data import ParquetWindowDataset, open_window_dataset
from world_model.mpc import (
    CEMConfig,
    LatentCEMCostConfig,
    cem_plan_latent_target,
    compute_action_bounds,
)


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    model_cfg = ckpt["model_config"]

    dynamics_type = args_dict.get("dynamics_type", "mlp")
    if dynamics_type == "td-jepa":
        dynamics_type = "mlp"

    model = WorldModel(
        odom_dim=model_cfg["odom_dim"],
        imu_dim=model_cfg["imu_dim"],
        act_dim=model_cfg["act_dim"],
        state_dim=model_cfg["state_dim"],
        backbone=args_dict.get("backbone", "cnn"),
        backbone_kwargs={
            "model_name": args_dict.get("backbone_name", "dinov2_vitb14"),
            "device": device,
        },
        dynamics_type=dynamics_type,
        image_size=args_dict.get("img_size", 224),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, args_dict, model_cfg


def encode_step(model: WorldModel, sample: dict[str, np.ndarray], t: int, device: str):
    rgb = torch.from_numpy(sample["rgb"][t : t + 1]).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    depth = torch.from_numpy(sample["depth"][t : t + 1]).unsqueeze(1).to(device)
    odom = torch.from_numpy(sample["odom"][t : t + 1]).to(device)
    imu = torch.from_numpy(sample["imu"][t : t + 1]).to(device)
    return model.encode(rgb, depth, odom, imu).z


@torch.no_grad()
def eval_mpc_wmsim_latent(
    model: WorldModel,
    dataset: ParquetWindowDataset,
    device: str,
    horizon: int = 16,
    num_episodes: int = 50,
    goal_offset: int | None = None,
    cem_population: int = 128,
    cem_iters: int = 5,
    cem_elites: int = 16,
    terminal_weight: float = 1.0,
    path_weight: float = 0.1,
    latent_success_threshold: float = 5.0,
):
    act_dim = dataset.D_act
    goal_offset = horizon if goal_offset is None else goal_offset

    action_mean, action_std, action_low, action_high = compute_action_bounds(dataset, torch.device(device))
    cfg = CEMConfig(
        horizon=horizon,
        iters=cem_iters,
        population=cem_population,
        elites=cem_elites,
        action_dim=act_dim,
        alpha=0.25,
        action_low=action_low,
        action_high=action_high,
        action_init_mean=action_mean,
        action_init_std=action_std,
    )
    cost_cfg = LatentCEMCostConfig(
        terminal_weight=terminal_weight,
        path_weight=path_weight,
    )

    results = {
        "latent_terminal_dist": [],
        "latent_path_dist": [],
        "planning_time_ms": [],
        "action_mse": [],
        "action_cosine_sim": [],
        "goal_success": [],
        "best_cost": [],
    }

    n_eval = min(num_episodes, len(dataset))
    print(
        f"Running latent wm_sim MPC on {n_eval} held-out windows "
        f"(horizon={horizon}, goal_offset={goal_offset}, pop={cem_population}, iters={cem_iters})..."
    )

    for idx in range(n_eval):
        sample = dataset[idx]
        T = sample["rgb"].shape[0]
        if T < max(horizon + 1, goal_offset + 1):
            continue

        z0 = encode_step(model, sample, 0, device)
        target_z = encode_step(model, sample, goal_offset, device)

        t_start = time.time()
        best_action, best_seq, best_cost = cem_plan_latent_target(
            model,
            z0,
            target_z,
            cfg,
            device=torch.device(device),
            cost_cfg=cost_cfg,
        )
        plan_time_ms = (time.time() - t_start) * 1000.0

        z_roll = model.rollout(z0, best_seq.unsqueeze(0))
        target_rep = target_z.unsqueeze(1)
        terminal_dist = torch.norm(z_roll[:, -1, :] - target_z, dim=-1).item()
        path_dist = torch.norm(z_roll[:, 1:, :] - target_rep, dim=-1).mean().item()

        planned_first = best_action
        actual_first = torch.from_numpy(sample["action"][0]).to(device)
        action_mse = F.mse_loss(planned_first, actual_first).item()
        cosine_sim = F.cosine_similarity(
            planned_first.unsqueeze(0), actual_first.unsqueeze(0)
        ).item()

        results["latent_terminal_dist"].append(terminal_dist)
        results["latent_path_dist"].append(path_dist)
        results["planning_time_ms"].append(plan_time_ms)
        results["action_mse"].append(action_mse)
        results["action_cosine_sim"].append(cosine_sim)
        results["goal_success"].append(float(terminal_dist <= latent_success_threshold))
        results["best_cost"].append(float(best_cost))

        print(
            f"  [{idx + 1}/{n_eval}] latent_terminal={terminal_dist:.4f} "
            f"latent_path={path_dist:.4f} action_mse={action_mse:.4f} "
            f"cos={cosine_sim:.3f} plan={plan_time_ms:.0f}ms"
        )

    summary = {}
    for metric, values in results.items():
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Latent-only wm_sim MPC replay evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--schema_config", default=None, help="Parquet layout YAML")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--goal_offset", type=int, default=None)
    ap.add_argument("--num_episodes", type=int, default=50)
    ap.add_argument("--cem_population", type=int, default=128)
    ap.add_argument("--cem_iters", type=int, default=5)
    ap.add_argument("--cem_elites", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--terminal_weight", type=float, default=1.0)
    ap.add_argument("--path_weight", type=float, default=0.1)
    ap.add_argument("--latent_success_threshold", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, train_args, model_cfg = load_model(args.checkpoint, args.device)
    img_size = args.img_size or train_args.get("img_size", 224)

    print(f"Loading held-out wm_sim dataset: {args.dataset}")
    ds = open_window_dataset(
        dataset_type="hf", path=args.dataset, split="val", image_size=img_size,
        schema_config=args.schema_config,
    )
    expected_act_dim = int(model_cfg["act_dim"])
    if ds.D_act != expected_act_dim:
        raise ValueError(
            f"Action dim mismatch: checkpoint expects act_dim={expected_act_dim}, "
            f"dataset provides D_act={ds.D_act}."
        )
    print(f"Held-out windows: {len(ds)}")

    summary = eval_mpc_wmsim_latent(
        model,
        ds,
        args.device,
        horizon=args.horizon,
        goal_offset=args.goal_offset,
        num_episodes=args.num_episodes,
        cem_population=args.cem_population,
        cem_iters=args.cem_iters,
        cem_elites=args.cem_elites,
        terminal_weight=args.terminal_weight,
        path_weight=args.path_weight,
        latent_success_threshold=args.latent_success_threshold,
    )

    print("\n" + "=" * 60)
    print("Latent wm_sim MPC Evaluation Results")
    print("=" * 60)
    for metric, stats in summary.items():
        print(
            f"  {metric}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )

    out_path = os.path.join(str(Path(args.checkpoint).parent), "mpc_eval_wmsim_latent.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()