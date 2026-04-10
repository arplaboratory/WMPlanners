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


def classify_mode(
    raw_abs_z: float,
    odom_denorm: np.ndarray,
    z_altitude_threshold: float = 1.5,
    z_std_threshold: float = 0.5,
    vz_threshold: float = 0.3,
) -> str:
    """Classify an episode as 'ground' or 'flight'.

    Uses three signals (any one triggers 'flight'):
      1. Absolute Z altitude at episode start — if the robot starts above
         z_altitude_threshold, it's airborne.
      2. Z-position variation within episode — if altitude changes
         significantly, vertical maneuvering occurred.
      3. Mean |vz| — sustained vertical velocity means flying.

    Args:
        raw_abs_z: Absolute Z-position at frame 0 (meters, from raw HF data
            before delta conversion). This is the primary signal.
        odom_denorm: (T, D_odom) denormalized odom with positions as deltas.
            Used for within-episode Z variation and velocity.
        z_altitude_threshold: Absolute Z height to classify as flight (meters).
        z_std_threshold: Std of Z-delta within episode (meters).
        vz_threshold: Mean |vz| threshold (m/s).

    Returns:
        'flight' if any airborne signal detected, else 'ground'.
    """
    # Signal 1: absolute altitude at start of episode
    if abs(raw_abs_z) >= z_altitude_threshold:
        return "flight"

    D = odom_denorm.shape[1]

    # Signal 2: altitude variation within episode (Z is delta from frame 0)
    if D >= 3:
        z_delta = odom_denorm[:, 2]
        if float(np.std(z_delta)) >= z_std_threshold:
            return "flight"

    # Signal 3: vertical velocity
    if D >= 10:
        vz = odom_denorm[:, 9]
        if float(np.mean(np.abs(vz))) >= vz_threshold:
            return "flight"

    return "ground"


@torch.no_grad()
def evaluate_trajectories(
    model: WorldModel,
    dataset: HFWindowDataset,
    norm_stats: NormStats | None,
    horizons: list[int],
    device: str,
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

    # Pre-fetch raw absolute Z at frame 0 for each val episode (before delta conversion).
    # Access the underlying HF dataset directly to get the original odom_2 values.
    raw_z0_by_wid = {}
    if hasattr(dataset, '_ds') and 'odom_2' in dataset._ds.column_names:
        for wid in dataset._window_ids:
            first_row = dataset._window_rows[wid][0]
            raw_z0_by_wid[wid] = float(dataset._ds[first_row]['odom_2'])

    for idx in range(len(dataset)):
        sample = dataset[idx]
        T = sample["odom"].shape[0]

        if T < max_h + 1:
            n_skipped += 1
            continue

        # The dataset returns odom with positions as deltas from frame 0.
        odom_np = sample["odom"]  # (T, D) — may be normalized, positions are deltas
        if norm_stats is not None:
            odom_denorm = norm_stats.denormalize_odom(
                torch.from_numpy(odom_np).float()
            ).numpy()
        else:
            odom_denorm = odom_np.copy()

        # Get raw absolute Z at frame 0 for this episode
        wid = dataset._window_ids[idx]
        raw_abs_z = raw_z0_by_wid.get(wid, 0.0)

        # Classify mode using absolute altitude + within-episode vertical motion
        mode = classify_mode(raw_abs_z, odom_denorm)

        # Prepare inputs — add batch dim
        rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(device)
        depth = torch.from_numpy(sample["depth"]).unsqueeze(0).to(device)
        odom_in = torch.from_numpy(sample["odom"]).unsqueeze(0).to(device)
        imu = torch.from_numpy(sample["imu"]).unsqueeze(0).to(device)
        action = torch.from_numpy(sample["action"]).unsqueeze(0).to(device)

        # Encode frame 0 -> z0
        out0 = model.encode(rgb[:, 0], depth[:, 0], odom_in[:, 0], imu[:, 0])
        z0 = out0.z

        # Rollout dynamics with real actions for max_h steps
        a_seq = action[:, 0:max_h]
        z_rollout = model.rollout(z0, a_seq)  # (1, max_h+1, z_dim)

        # Decode state predictions (in normalized delta space)
        state_pred = model.state_head(z_rollout)  # (1, max_h+1, state_dim)

        # Denormalize predictions back to real delta-position units (meters)
        if norm_stats is not None:
            odom_mean = norm_stats.odom_mean[:state_dim].to(device)
            odom_std = norm_stats.odom_std[:state_dim].to(device)
            state_pred = state_pred * odom_std + odom_mean

        # GT position deltas (first 3 dims of denormalized odom — already in delta form)
        gt_pos = torch.from_numpy(
            odom_denorm[:max_h + 1, :3]
        ).unsqueeze(0).to(device)  # (1, max_h+1, 3)

        # Predicted position deltas (first 3 dims of decoded state)
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
        key = f"{h / 5.0:.1f}s"  # 5 Hz frame rate
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
    ap.add_argument("--z_altitude", type=float, default=1.5,
                    help="Absolute Z height to classify as flight (default: 1.5 m)")
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
        model, ds, norm_stats, args.horizons, args.device,
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
