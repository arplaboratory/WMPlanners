"""Multi-horizon trajectory evaluation with ground/flight mode breakdown.

Loads a trained world model checkpoint, runs action-conditioned rollouts on
validation episodes, and reports ADE/FDE metrics at multiple horizons.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from world_model.architecture import WorldModel
from world_model.data import HFWindowDataset
from world_model.data.normalization import NormStats


def load_model_and_stats(checkpoint_path: str, device: str):
    """Load a trained WorldModel from a checkpoint file.

    Returns:
        model, norm_stats, args_dict, mcfg
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    mcfg = ckpt["model_config"]

    backbone = args_dict.get("backbone", "cnn")
    dynamics_type = args_dict.get("dynamics_type", "mlp")
    # td-jepa uses mlp dynamics internally
    if dynamics_type == "td-jepa":
        dynamics_type = "mlp"

    model = WorldModel(
        odom_dim=mcfg["odom_dim"],
        imu_dim=mcfg["imu_dim"],
        act_dim=mcfg["act_dim"],
        state_dim=mcfg["state_dim"],
        backbone=backbone,
        backbone_kwargs={
            "model_name": args_dict.get("backbone_name", "dinov2_vitb14"),
            "device": device,
        },
        dynamics_type=dynamics_type,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load norm_stats from same directory if it exists
    norm_stats = None
    ckpt_dir = str(Path(checkpoint_path).parent)
    norm_stats_path = os.path.join(ckpt_dir, "norm_stats.pt")
    if os.path.exists(norm_stats_path):
        norm_stats = NormStats.load(norm_stats_path)
        print(f"[INFO] Loaded norm_stats from {norm_stats_path}")
    elif ckpt.get("has_norm_stats", False):
        print("[WARN] Checkpoint expects norm_stats but norm_stats.pt not found "
              f"in {ckpt_dir}")

    return model, norm_stats, args_dict, mcfg


def classify_mode(odom: np.ndarray, speed_threshold: float = 2.0) -> str:
    """Classify an episode as 'ground' or 'flight' based on mean speed.

    Args:
        odom: (T, D_odom) numpy array. If D >= 10, columns 7/8/9 are
            linear_vel_x/y/z; otherwise columns 0:3 are used.
        speed_threshold: mean speed threshold in m/s.

    Returns:
        'flight' if mean speed >= threshold, else 'ground'.
    """
    D = odom.shape[1]
    if D >= 10:
        vel = odom[:, 7:10]
    else:
        vel = odom[:, 0:3]
    speeds = np.linalg.norm(vel, axis=1)
    mean_speed = float(np.mean(speeds))
    return "flight" if mean_speed >= speed_threshold else "ground"


@torch.no_grad()
def evaluate_trajectories(
    model: WorldModel,
    dataset: HFWindowDataset,
    norm_stats: NormStats | None,
    horizons: list[int],
    device: str,
    speed_threshold: float = 2.0,
) -> dict:
    """Run action-conditioned rollouts and compute ADE/FDE at each horizon.

    Returns:
        dict keyed by horizon string (e.g. '0.8s') with mean ade, fde,
        ground_ade, flight_ade, and episode counts.
    """
    max_h = max(horizons)
    state_dim = model.state_head.net[-1].out_features

    # Accumulators per horizon
    results = {h: {"ade": [], "fde": [], "ground_ade": [], "flight_ade": []}
               for h in horizons}

    n_skipped = 0
    n_evaluated = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        T = sample["odom"].shape[0]

        if T < max_h + 1:
            n_skipped += 1
            continue

        # Classify mode using raw (denormalized) odom
        odom_tensor = torch.from_numpy(sample["odom"]).float()
        if norm_stats is not None:
            raw_odom = norm_stats.denormalize_odom(odom_tensor).numpy()
        else:
            raw_odom = sample["odom"].copy()
        mode = classify_mode(raw_odom, speed_threshold)

        # Prepare inputs — add batch dim
        rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(device)      # (1,T,3,H,W)
        depth = torch.from_numpy(sample["depth"]).unsqueeze(0).to(device)  # (1,T,1,H,W)
        odom_in = torch.from_numpy(sample["odom"]).unsqueeze(0).to(device) # (1,T,D)
        imu = torch.from_numpy(sample["imu"]).unsqueeze(0).to(device)     # (1,T,D)
        action = torch.from_numpy(sample["action"]).unsqueeze(0).to(device) # (1,T,D)

        # Encode frame 0 -> z0
        out0 = model.encode(rgb[:, 0], depth[:, 0], odom_in[:, 0], imu[:, 0])
        z0 = out0.z  # (1, z_dim)

        # Rollout dynamics with real actions for max_h steps
        a_seq = action[:, 0:max_h]  # (1, max_h, act_dim)
        z_rollout = model.rollout(z0, a_seq)  # (1, max_h+1, z_dim)

        # Decode state predictions
        state_pred = model.state_head(z_rollout)  # (1, max_h+1, state_dim)

        # Denormalize predictions if norm_stats exists
        # state_head predicts the first state_dim dims of odom
        if norm_stats is not None:
            # Build partial mean/std for state_dim dims
            odom_mean = norm_stats.odom_mean[:state_dim].to(device)
            odom_std = norm_stats.odom_std[:state_dim].to(device)
            state_pred = state_pred * odom_std + odom_mean

        # GT positions (first 3 dims of raw odom)
        gt_pos = torch.from_numpy(raw_odom[:max_h + 1, :3]).unsqueeze(0).to(device)  # (1, max_h+1, 3)

        # Predicted positions (first 3 dims of state prediction)
        pred_pos = state_pred[:, :, :3]  # (1, max_h+1, 3)

        # Compute per-step displacement errors
        displacements = torch.norm(pred_pos - gt_pos, dim=-1)  # (1, max_h+1)
        displacements = displacements.squeeze(0)  # (max_h+1,)

        for h in horizons:
            fde = displacements[h].item()
            ade = displacements[:h + 1].mean().item()

            results[h]["ade"].append(ade)
            results[h]["fde"].append(fde)
            if mode == "ground":
                results[h]["ground_ade"].append(ade)
            else:
                results[h]["flight_ade"].append(ade)

        n_evaluated += 1

    print(f"[INFO] Evaluated {n_evaluated} episodes, skipped {n_skipped} "
          f"(T < {max_h + 1})")

    # Aggregate into summary dict keyed by horizon string
    summary = {}
    for h in horizons:
        key = f"{h * 0.1:.1f}s"
        r = results[h]
        summary[key] = {
            "horizon_steps": h,
            "ade": float(np.mean(r["ade"])) if r["ade"] else 0.0,
            "fde": float(np.mean(r["fde"])) if r["fde"] else 0.0,
            "ground_ade": float(np.mean(r["ground_ade"])) if r["ground_ade"] else 0.0,
            "flight_ade": float(np.mean(r["flight_ade"])) if r["flight_ade"] else 0.0,
            "count": len(r["ade"]),
            "ground_count": len(r["ground_ade"]),
            "flight_count": len(r["flight_ade"]),
        }

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Multi-horizon trajectory evaluation for trained world models."
    )
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Path to HuggingFace dataset directory or Hub ID")
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 25],
                    help="Evaluation horizons in timesteps (default: 8 16 25)")
    ap.add_argument("--img_size", type=int, default=224,
                    help="Image resize dimension (default: 224)")
    ap.add_argument("--speed_threshold", type=float, default=2.0,
                    help="Speed threshold for ground/flight classification (default: 2.0)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load model and stats
    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, args_dict, mcfg = load_model_and_stats(
        args.checkpoint, args.device
    )
    print(f"  backbone={args_dict.get('backbone', 'cnn')}, "
          f"dynamics={args_dict.get('dynamics_type', 'mlp')}, "
          f"state_dim={mcfg['state_dim']}")

    # Load dataset and filter to val episodes (eid % 5 == 0)
    print(f"Loading dataset: {args.dataset}")
    ds = HFWindowDataset(args.dataset, img_size=args.img_size, norm_stats=norm_stats)
    all_ids = ds._window_ids
    val_ids = set(eid for eid in all_ids if eid % 5 == 0)
    ds._window_ids = [eid for eid in ds._window_ids if eid in val_ids]
    ds._window_rows = {eid: ds._window_rows[eid] for eid in ds._window_ids}
    print(f"  Val episodes (eid%%5==0): {len(ds)}")

    # Evaluate
    print(f"\nEvaluating at horizons: {args.horizons}")
    summary = evaluate_trajectories(
        model, ds, norm_stats, args.horizons, args.device, args.speed_threshold
    )

    # Print formatted table
    print("\n" + "=" * 78)
    print(f"{'Horizon':>10s} | {'ADE':>8s} | {'FDE':>8s} | "
          f"{'Gnd ADE':>8s} | {'Flt ADE':>8s} | {'#Eps':>5s} | "
          f"{'#Gnd':>5s} | {'#Flt':>5s}")
    print("-" * 78)
    for key in sorted(summary.keys(), key=lambda k: summary[k]["horizon_steps"]):
        s = summary[key]
        print(f"{key:>10s} | {s['ade']:8.4f} | {s['fde']:8.4f} | "
              f"{s['ground_ade']:8.4f} | {s['flight_ade']:8.4f} | "
              f"{s['count']:5d} | {s['ground_count']:5d} | {s['flight_count']:5d}")
    print("=" * 78)

    # Save results to JSON in checkpoint directory
    ckpt_dir = str(Path(args.checkpoint).parent)
    out_path = os.path.join(ckpt_dir, "trajectory_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
