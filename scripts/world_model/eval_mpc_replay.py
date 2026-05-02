"""MPC evaluation by replaying real episodes.

For each val episode:
  1. Start from the real initial observation
  2. At each step, run CEM planning to find the best action sequence
  3. Compare the MPC-chosen action vs the actual action the robot took
  4. Measure: action error, predicted trajectory quality, planning speed

This tests: "if the robot used this world model for planning, how close
would its chosen actions be to what a human/expert actually did?"

Usage:
    python scripts/world_model/eval_mpc_replay.py \
        --checkpoint /scratch/$USER/wm_experiments_v2/dino_rssm_c1u/latest.pt \
        --dataset lquan9/collect1_unfiltered
"""

from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from world_model.architecture import WorldModel
from world_model.data import HFWindowDataset
from world_model.data.normalization import NormStats
from world_model.mpc import cem_plan, CEMConfig


def load_model_and_stats(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt['args']
    mcfg = ckpt['model_config']

    dynamics_type = args_dict.get('dynamics_type', 'mlp')
    if dynamics_type == 'td-jepa':
        dynamics_type = 'mlp'

    model = WorldModel(
        odom_dim=mcfg['odom_dim'],
        imu_dim=mcfg['imu_dim'],
        act_dim=mcfg['act_dim'],
        state_dim=mcfg['state_dim'],
        backbone=args_dict.get('backbone', 'cnn'),
        backbone_kwargs={
            'model_name': args_dict.get('backbone_name', 'dinov2_vitb14'),
            'device': device,
        },
        dynamics_type=dynamics_type,
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    norm_dir = str(Path(checkpoint_path).parent)
    norm_path = os.path.join(norm_dir, 'norm_stats.pt')
    norm_stats = NormStats.load(norm_path) if os.path.exists(norm_path) else None

    return model, norm_stats, args_dict, mcfg


def make_goal_tracking_cost(w_goal=1.0, w_smooth=0.1):
    """Cost function: track the ground-truth future trajectory.

    At eval time we know where the robot actually went. The cost
    measures how close the planned trajectory would get to the
    robot's actual future positions. This tests whether CEM + the
    world model can recover the expert's trajectory.
    """
    def cost_fn(state_seq, risk_seq, goal):
        # state_seq: (N, H+1, state_dim) — predicted poses
        # goal: (1, H, 3) — ground truth future positions (first 3 dims of odom)
        N = state_seq.shape[0]
        H = goal.shape[1]

        pred_pos = state_seq[:, 1:H+1, :3]  # (N, H, 3) skip z_0
        gt_pos = goal.expand(N, -1, -1)     # (N, H, 3)

        # Position tracking cost
        pos_cost = ((pred_pos - gt_pos) ** 2).sum(dim=-1).mean(dim=1)  # (N,)

        # Action smoothness (penalize jerky plans) — implicit via state smoothness
        if state_seq.shape[1] > 2:
            vel = state_seq[:, 2:, :3] - state_seq[:, 1:-1, :3]
            accel = vel[:, 1:] - vel[:, :-1]
            smooth_cost = (accel ** 2).sum(dim=-1).mean(dim=1)
        else:
            smooth_cost = torch.zeros(N, device=state_seq.device)

        return w_goal * pos_cost + w_smooth * smooth_cost

    return cost_fn


@torch.no_grad()
def eval_mpc_replay(
    model: WorldModel,
    dataset: HFWindowDataset,
    norm_stats: NormStats | None,
    device: str,
    horizon: int = 16,
    num_episodes: int = 50,
    cem_population: int = 128,
    cem_iters: int = 5,
    cem_elites: int = 16,
):
    """Run MPC replay evaluation.

    For each val episode:
      1. Encode frame 0 → z₀
      2. Run CEM with goal = actual future positions
      3. Execute first planned action
      4. Measure: action similarity, trajectory tracking, planning time
    """
    state_dim = model.state_head.net[-1].out_features
    act_dim = dataset.D_act

    # Determine action range from dataset (for CEM bounds)
    # Use normalized range since model operates in normalized space
    action_min = -3.0  # ~3 std in normalized space
    action_max = 3.0

    cfg = CEMConfig(
        horizon=horizon,
        iters=cem_iters,
        population=cem_population,
        elites=cem_elites,
        action_dim=act_dim,
        action_min=action_min,
        action_max=action_max,
        init_std=1.0,
        alpha=0.25,
    )

    cost_fn = make_goal_tracking_cost()

    results = {
        'action_mse': [],
        'action_cosine_sim': [],
        'trajectory_ade': [],
        'planning_time_ms': [],
        'plan_cost': [],
    }

    n_eval = min(num_episodes, len(dataset))
    print(f"Running MPC replay on {n_eval} episodes (horizon={horizon}, "
          f"pop={cem_population}, iters={cem_iters})...")

    for idx in range(n_eval):
        sample = dataset[idx]
        T = sample['odom'].shape[0]
        if T < horizon + 1:
            continue

        rgb = torch.from_numpy(sample['rgb']).unsqueeze(0).to(device)
        depth = torch.from_numpy(sample['depth']).unsqueeze(0).to(device)
        odom = torch.from_numpy(sample['odom']).unsqueeze(0).to(device)
        imu = torch.from_numpy(sample['imu']).unsqueeze(0).to(device)
        action_gt = torch.from_numpy(sample['action']).unsqueeze(0).to(device)

        # Encode current observation
        out = model.encode(rgb[:, 0], depth[:, 0], odom[:, 0], imu[:, 0])
        z0 = out.z  # (1, z_dim)

        # Ground truth future positions as goal (for cost function)
        # These are normalized odom positions (deltas from frame 0)
        gt_future_pos = odom[:, 1:horizon+1, :3]  # (1, H, 3)

        # Run CEM planning
        t_start = time.time()
        best_action, best_seq = cem_plan(
            model, z0, gt_future_pos, cost_fn, cfg, device=torch.device(device)
        )
        plan_time = (time.time() - t_start) * 1000  # ms

        # Compare planned action vs actual action
        planned_first = best_seq[0]   # (act_dim,)
        actual_first = action_gt[0, 0]  # (act_dim,)

        action_mse = F.mse_loss(planned_first, actual_first).item()
        cosine_sim = F.cosine_similarity(
            planned_first.unsqueeze(0), actual_first.unsqueeze(0)
        ).item()

        # Rollout planned sequence and measure trajectory quality
        z_planned = model.rollout(z0, best_seq.unsqueeze(0))  # (1, H+1, z_dim)
        state_planned = model.state_head(z_planned)[:, 1:horizon+1, :3]  # (1, H, 3)

        # ADE between planned trajectory and actual
        disp = torch.norm(state_planned - gt_future_pos, dim=-1)  # (1, H)
        traj_ade = disp.mean().item()

        # Denormalize ADE to meters
        if norm_stats is not None:
            pos_std = norm_stats.odom_std[:3].to(device)
            traj_ade_m = (disp * pos_std.mean()).mean().item()
        else:
            traj_ade_m = traj_ade

        results['action_mse'].append(action_mse)
        results['action_cosine_sim'].append(cosine_sim)
        results['trajectory_ade'].append(traj_ade_m)
        results['planning_time_ms'].append(plan_time)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{n_eval}] act_mse={action_mse:.4f} "
                  f"cos_sim={cosine_sim:.3f} traj_ade={traj_ade_m:.3f}m "
                  f"plan={plan_time:.0f}ms")

    # Aggregate
    summary = {}
    for k, v in results.items():
        if v:
            summary[k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'min': float(np.min(v)),
                'max': float(np.max(v)),
            }

    return summary


def main():
    ap = argparse.ArgumentParser(description="MPC replay evaluation")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--horizon', type=int, default=16)
    ap.add_argument('--num_episodes', type=int, default=50)
    ap.add_argument('--cem_population', type=int, default=128)
    ap.add_argument('--cem_iters', type=int, default=5)
    ap.add_argument('--img_size', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, train_args, mcfg = load_model_and_stats(args.checkpoint, args.device)
    img_size = args.img_size or train_args.get('img_size', 224)

    print(f"Loading dataset: {args.dataset}")
    ds = HFWindowDataset(args.dataset, img_size=img_size, norm_stats=norm_stats)
    ds.force_velocity_actions()  # actions = odom[7:13] (6-dim velocity)

    # Filter to val episodes
    val_ids = set(eid for eid in ds._window_ids if eid % 5 == 0)
    ds._window_ids = [eid for eid in ds._window_ids if eid in val_ids]
    ds._window_rows = {k: v for k, v in ds._window_rows.items() if k in val_ids}
    print(f"Val episodes: {len(ds)}")

    summary = eval_mpc_replay(
        model, ds, norm_stats, args.device,
        horizon=args.horizon,
        num_episodes=args.num_episodes,
        cem_population=args.cem_population,
        cem_iters=args.cem_iters,
    )

    # Print results
    print("\n" + "=" * 60)
    print("MPC Replay Evaluation Results")
    print("=" * 60)
    for metric, stats in summary.items():
        print(f"  {metric}:")
        print(f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}")
    print("=" * 60)

    # Key takeaways
    if 'trajectory_ade' in summary:
        ade = summary['trajectory_ade']['mean']
        print(f"\n  Planned trajectory ADE: {ade:.3f}m")
        if ade < 0.5:
            print("  \u2192 Excellent — planner closely tracks expert trajectory")
        elif ade < 1.0:
            print("  \u2192 Good — planner follows general trajectory shape")
        elif ade < 2.0:
            print("  \u2192 Marginal — planner gets rough direction right")
        else:
            print("  \u2192 Poor — planner deviates significantly")

    if 'planning_time_ms' in summary:
        t = summary['planning_time_ms']['mean']
        print(f"  Planning time: {t:.0f}ms per step")
        if t < 200:
            print("  \u2192 Real-time capable (5 Hz control loop)")
        else:
            print(f"  \u2192 Too slow for real-time (need <200ms, got {t:.0f}ms)")

    if 'action_cosine_sim' in summary:
        cs = summary['action_cosine_sim']['mean']
        print(f"  Action cosine similarity: {cs:.3f}")
        if cs > 0.8:
            print("  \u2192 Planned actions closely match expert")
        elif cs > 0.5:
            print("  \u2192 Planned actions roughly aligned with expert")
        else:
            print("  \u2192 Planned actions diverge from expert")

    # Save
    ckpt_dir = str(Path(args.checkpoint).parent)
    out_path = os.path.join(ckpt_dir, 'mpc_eval.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
