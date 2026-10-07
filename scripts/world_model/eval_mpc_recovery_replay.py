from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from world_model.architecture import WorldModel, checkpoint_backbone_kwargs
from world_model.data import HFWindowDataset
from world_model.data.normalization import NormStats
from world_model.mpc import CEMConfig, cem_plan_with_cost, compute_action_bounds
from world_model.recovery import (
    EliteMemory,
    EliteMemoryConfig,
    RecoveryConfig,
    plan_recovery_navigation,
    recoverability_score,
    revalidate_recovery_state,
    select_recovery_state,
    should_trigger_recovery,
)


def load_model_and_stats(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    mcfg = ckpt["model_config"]

    dynamics_type = args_dict.get("dynamics_type", "mlp")
    if dynamics_type == "td-jepa":
        dynamics_type = "mlp"

    model = WorldModel(
        odom_dim=mcfg["odom_dim"],
        imu_dim=mcfg["imu_dim"],
        act_dim=mcfg["act_dim"],
        state_dim=mcfg["state_dim"],
        backbone=args_dict.get("backbone", "cnn"),
        backbone_kwargs=checkpoint_backbone_kwargs(ckpt, device),
        dynamics_type=dynamics_type,
        image_size=args_dict.get("img_size", 224),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    norm_dir = str(Path(checkpoint_path).parent)
    norm_path = os.path.join(norm_dir, "norm_stats.pt")
    norm_stats = NormStats.load(norm_path) if os.path.exists(norm_path) else None

    return model, norm_stats, args_dict, mcfg


def make_goal_tracking_cost(w_goal=1.0, w_smooth=0.1):
    def cost_fn(state_seq, risk_seq, goal):
        pred_pos = state_seq[:, 1 : goal.shape[1] + 1, :3]
        gt_pos = goal.expand(pred_pos.shape[0], -1, -1)

        pos_cost = ((pred_pos - gt_pos) ** 2).sum(dim=-1).mean(dim=1)

        if state_seq.shape[1] > 2:
            vel = state_seq[:, 2:, :3] - state_seq[:, 1:-1, :3]
            accel = vel[:, 1:] - vel[:, :-1]
            smooth_cost = (accel ** 2).sum(dim=-1).mean(dim=1)
        else:
            smooth_cost = torch.zeros(pred_pos.shape[0], device=state_seq.device)

        return w_goal * pos_cost + w_smooth * smooth_cost

    return cost_fn


def fit_linear_probe_from_rows(rows: list[dict], bin_threshold: float) -> dict:
    X = np.stack([np.asarray(r["z"], dtype=np.float32) for r in rows], axis=0)
    y = np.asarray([float(r["recoverability"]) for r in rows], dtype=np.float32)
    eid = np.asarray([int(r["episode_id"]) for r in rows], dtype=np.int64)

    train_mask = (eid % 5) != 0
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return {
            "skipped": True,
            "reason": "Train/test split is empty under episode_id%5 rule",
            "n_total": int(len(rows)),
        }

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    reg = make_pipeline(StandardScaler(), LinearRegression())
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    out = {
        "n_total": int(len(rows)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "recoverability_mean_train": float(y_train.mean()),
        "recoverability_mean_test": float(y_test.mean()),
        "linear_regression": {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "corr": float(np.corrcoef(y_test, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0,
        },
    }

    yb_train = (y_train >= bin_threshold).astype(np.int64)
    yb_test = (y_test >= bin_threshold).astype(np.int64)
    if np.unique(yb_train).size > 1 and np.unique(yb_test).size > 1:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_train, yb_train)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        out["linear_classification"] = {
            "threshold": float(bin_threshold),
            "accuracy": float(accuracy_score(yb_test, pred)),
            "auroc": float(roc_auc_score(yb_test, prob)),
            "positive_rate_test": float(yb_test.mean()),
        }
    else:
        out["linear_classification"] = {
            "threshold": float(bin_threshold),
            "skipped": True,
            "reason": "Binary labels collapsed in train or test split",
            "positive_rate_test": float(yb_test.mean()),
        }

    return out


@torch.no_grad()
def eval_mpc_recovery_replay(
    model,
    dataset: HFWindowDataset,
    norm_stats,
    device: str,
    horizon: int,
    num_episodes: int,
    cem_population: int,
    cem_iters: int,
    cem_elites: int,
    memory_capacity: int,
    reachability_threshold: float,
    progress_window: int,
    progress_threshold: float,
    plan_cost_threshold: float,
    trigger_cost_threshold: float,
    recoverability_samples: int,
    recoverability_horizon: int,
    top_k_memory: int,
    goal_success_threshold: float,
    collect_probe_rows: bool,
    collect_probe_max_states: int,
    collect_probe_bin_threshold: float,
    collect_probe_optionality: bool,
    use_optionality_metric: bool,
):
    act_dim = dataset.D_act

    # Per-dimension action bounds from the actual data -- a single scalar
    # range is meaningless when the action vector mixes wildly different
    # scales (e.g. raw absolute position vs. quaternion vs. velocity).
    action_mean, action_std, action_low, action_high = compute_action_bounds(dataset, device)
    print(
        f"[CEM] per-dim action bounds -- min={action_low.tolist()}\n"
        f"                              max={action_high.tolist()}"
    )

    cem_cfg = CEMConfig(
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
    recovery_cfg = RecoveryConfig(
        reachability_threshold=reachability_threshold,
        failure_threshold=progress_threshold,
        progress_window=progress_window,
        min_memory_size=min(10, memory_capacity),
        top_k_memory=top_k_memory,
        recoverability_samples=recoverability_samples,
        recoverability_horizon=recoverability_horizon,
        success_cost_threshold=plan_cost_threshold,
        revalidation_threshold=reachability_threshold,
    )
    memory_cfg = EliteMemoryConfig(capacity=memory_capacity)
    cost_fn = make_goal_tracking_cost()

    results = {
        "episodes_evaluated": 0,
        "steps_evaluated": 0,
        "recovery_triggers": 0,
        "recovery_selections": 0,
        "revalidation_successes": 0,
        "base_action_mse": [],
        "recovery_action_mse": [],
        "base_trajectory_ade": [],
        "recovery_trajectory_ade": [],
        "base_goal_min_dist": [],
        "recovery_goal_min_dist": [],
        "base_goal_success": [],
        "recovery_goal_success": [],
        "base_plan_cost": [],
        "recovery_nav_cost": [],
        "selected_recoverability": [],
        "selected_reachability": [],
        "reachable_count": [],
        "memory_size": [],
        "planning_time_ms": [],
        "recovery_total_time_ms": [],
        "recovery_overhead_time_ms": [],
    }
    probe_rows: list[dict] = []

    n_eval = min(num_episodes, len(dataset))
    print(
        f"Running MPC+RGER replay on {n_eval} episodes "
        f"(horizon={horizon}, pop={cem_population}, iters={cem_iters})..."
    )

    for idx in range(n_eval):
        sample = dataset[idx]
        total_frames = sample["odom"].shape[0]
        if total_frames < horizon + 2:
            continue

        rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(device)
        depth = torch.from_numpy(sample["depth"]).unsqueeze(0).to(device)
        odom = torch.from_numpy(sample["odom"]).unsqueeze(0).to(device)
        imu = torch.from_numpy(sample["imu"]).unsqueeze(0).to(device)
        action_gt = torch.from_numpy(sample["action"]).unsqueeze(0).to(device)

        final_goal = odom[:, -1, :3]
        goal_distance_history: list[float] = []
        elite_memory = EliteMemory(memory_cfg)
        episode_base_success = False
        episode_recovery_success = False
        results["episodes_evaluated"] += 1

        max_start = total_frames - horizon - 1
        for t in range(max_start):
            out = model.encode(rgb[:, t], depth[:, t], odom[:, t], imu[:, t])
            z_t = out.z
            current_pos = odom[:, t, :3]
            current_goal_distance = torch.norm(current_pos - final_goal, dim=-1).mean().item()
            goal_distance_history.append(current_goal_distance)

            quality = -current_goal_distance
            elite_memory.add(
                z=z_t.squeeze(0),
                q=quality,
                timestep=t,
                p=current_pos.squeeze(0),
            )

            gt_future_pos = odom[:, t + 1 : t + horizon + 1, :3]
            recovery_goal = gt_future_pos[:, : recovery_cfg.recoverability_horizon]

            t_start = time.time()
            best_action, best_seq, best_cost = cem_plan_with_cost(
                model=model,
                z0=z_t,
                goal=gt_future_pos,
                cost_fn=cost_fn,
                cfg=cem_cfg,
                device=torch.device(device),
            )
            plan_time_ms = (time.time() - t_start) * 1000.0

            if action_gt.shape[-1] == act_dim:
                actual_first = action_gt[0, t]
            else:
                actual_first = odom[0, t, 7 : 7 + act_dim]
            base_action_mse = F.mse_loss(best_action, actual_first).item()

            z_planned = model.rollout(z_t, best_seq.unsqueeze(0))
            state_planned = model.state_head(z_planned)[:, 1 : horizon + 1, :3]
            base_disp = torch.norm(state_planned - gt_future_pos, dim=-1)
            if norm_stats is not None:
                pos_std = norm_stats.odom_std[:3].to(device)
                base_traj_ade = (base_disp * pos_std.mean()).mean().item()
            else:
                base_traj_ade = base_disp.mean().item()

            base_goal_disp = torch.norm(state_planned - final_goal.unsqueeze(1), dim=-1)
            if norm_stats is not None:
                pos_std = norm_stats.odom_std[:3].to(device)
                base_goal_min_dist = (base_goal_disp * pos_std.mean()).min().item()
            else:
                base_goal_min_dist = base_goal_disp.min().item()
            base_goal_success = float(base_goal_min_dist <= goal_success_threshold)
            episode_base_success = episode_base_success or bool(base_goal_success)

            results["base_action_mse"].append(base_action_mse)
            results["base_trajectory_ade"].append(base_traj_ade)
            results["base_goal_min_dist"].append(base_goal_min_dist)
            results["base_goal_success"].append(base_goal_success)
            results["base_plan_cost"].append(best_cost)
            results["planning_time_ms"].append(plan_time_ms)
            results["steps_evaluated"] += 1

            trigger_recovery = should_trigger_recovery(
                goal_distance_history=goal_distance_history,
                planner_objective_max=None,
                cfg=recovery_cfg,
                planner_best_cost=best_cost,
                planner_cost_threshold=trigger_cost_threshold,
            )

            if collect_probe_rows and len(probe_rows) < collect_probe_max_states:
                current_r = recoverability_score(
                    model=model,
                    z=z_t.squeeze(0),
                    goal=recovery_goal,
                    cost_fn=cost_fn,
                    cem_cfg=cem_cfg,
                    cfg=recovery_cfg,
                    device=torch.device(device),
                    use_optionality_metric=collect_probe_optionality,
                )
                probe_rows.append(
                    {
                        "episode_id": int(dataset._window_ids[idx]),
                        "t": int(t),
                        "z": z_t.squeeze(0).detach().cpu().numpy().tolist(),
                        "recoverability": float(current_r),
                        "triggered": float(trigger_recovery),
                    }
                )

            if not trigger_recovery:
                continue

            results["recovery_triggers"] += 1
            recovery_t_start = time.time()
            selection, stats = select_recovery_state(
                model=model,
                current_z=z_t,
                goal=recovery_goal,
                memory=elite_memory,
                cost_fn=cost_fn,
                cem_cfg=cem_cfg,
                cfg=recovery_cfg,
                device=torch.device(device),
                use_optionality_metric=use_optionality_metric,
            )
            results["reachable_count"].append(stats.reachable_count)
            results["memory_size"].append(stats.memory_size)

            if selection is None:
                continue

            results["recovery_selections"] += 1
            results["selected_recoverability"].append(selection.recoverability)
            results["selected_reachability"].append(selection.reachability)

            recovery_action, recovery_seq, recovery_nav_cost = plan_recovery_navigation(
                model=model,
                current_z=z_t,
                selected=selection,
                cem_cfg=cem_cfg,
                device=torch.device(device),
            )
            recovery_action_mse = F.mse_loss(recovery_action, actual_first).item()

            z_recovery = model.rollout(z_t, recovery_seq.unsqueeze(0))
            state_recovery = model.state_head(z_recovery)[:, 1 : horizon + 1, :3]
            recovery_disp = torch.norm(state_recovery - gt_future_pos, dim=-1)
            if norm_stats is not None:
                pos_std = norm_stats.odom_std[:3].to(device)
                recovery_traj_ade = (recovery_disp * pos_std.mean()).mean().item()
            else:
                recovery_traj_ade = recovery_disp.mean().item()

            recovery_goal_disp = torch.norm(state_recovery - final_goal.unsqueeze(1), dim=-1)
            if norm_stats is not None:
                pos_std = norm_stats.odom_std[:3].to(device)
                recovery_goal_min_dist = (recovery_goal_disp * pos_std.mean()).min().item()
            else:
                recovery_goal_min_dist = recovery_goal_disp.min().item()
            recovery_goal_success = float(recovery_goal_min_dist <= goal_success_threshold)
            episode_recovery_success = episode_recovery_success or bool(recovery_goal_success)

            results["recovery_action_mse"].append(recovery_action_mse)
            results["recovery_trajectory_ade"].append(recovery_traj_ade)
            results["recovery_goal_min_dist"].append(recovery_goal_min_dist)
            results["recovery_goal_success"].append(recovery_goal_success)
            results["recovery_nav_cost"].append(recovery_nav_cost)

            if revalidate_recovery_state(
                model=model,
                selected=selection,
                goal=recovery_goal,
                cost_fn=cost_fn,
                cem_cfg=cem_cfg,
                cfg=recovery_cfg,
                device=torch.device(device),
                use_optionality_metric=use_optionality_metric,
            ):
                results["revalidation_successes"] += 1

            recovery_total_time_ms = (time.time() - recovery_t_start) * 1000.0
            results["recovery_total_time_ms"].append(recovery_total_time_ms)
            results["recovery_overhead_time_ms"].append(recovery_total_time_ms + plan_time_ms)

        if (idx + 1) % 5 == 0:
            print(
                f"  [{idx + 1}/{n_eval}] triggers={results['recovery_triggers']} "
                f"selections={len(results['selected_recoverability'])}"
            )

        results.setdefault("base_episode_success", []).append(float(episode_base_success))
        results.setdefault("recovery_episode_success", []).append(float(episode_recovery_success))

    summary: dict[str, object] = {
        "episodes_evaluated": results["episodes_evaluated"],
        "steps_evaluated": results["steps_evaluated"],
        "recovery_triggers": results["recovery_triggers"],
        "recovery_selections": len(results["selected_recoverability"]),
        "revalidation_successes": results["revalidation_successes"],
    }

    scalar_lists = [
        "base_action_mse",
        "recovery_action_mse",
        "base_trajectory_ade",
        "recovery_trajectory_ade",
        "base_goal_min_dist",
        "recovery_goal_min_dist",
        "base_goal_success",
        "recovery_goal_success",
        "base_plan_cost",
        "recovery_nav_cost",
        "selected_recoverability",
        "selected_reachability",
        "reachable_count",
        "memory_size",
        "planning_time_ms",
        "recovery_total_time_ms",
        "recovery_overhead_time_ms",
        "base_episode_success",
        "recovery_episode_success",
    ]
    for key in scalar_lists:
        values = results[key]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
            }

    if results["steps_evaluated"] > 0:
        summary["trigger_rate"] = results["recovery_triggers"] / results["steps_evaluated"]
    if results["recovery_triggers"] > 0:
        summary["selection_rate_given_trigger"] = (
            len(results["selected_recoverability"]) / results["recovery_triggers"]
        )
    if len(results["selected_recoverability"]) > 0:
        summary["revalidation_rate_given_selection"] = (
            results["revalidation_successes"] / len(results["selected_recoverability"])
        )

    if collect_probe_rows and probe_rows:
        summary["collected_probe"] = fit_linear_probe_from_rows(
            probe_rows,
            bin_threshold=collect_probe_bin_threshold,
        )
        summary["collected_probe"]["target"] = (
            "optionality" if collect_probe_optionality else "success_fraction"
        )
        summary["collected_probe"]["n_rows"] = int(len(probe_rows))
        summary["collected_probe_rows"] = probe_rows

    summary["recoverability_metric"] = (
        "optionality" if use_optionality_metric else "success_fraction"
    )

    return summary


def main():
    ap = argparse.ArgumentParser(description="Replay MPC + recoverability evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--cem_population", type=int, default=128)
    ap.add_argument("--cem_iters", type=int, default=5)
    ap.add_argument("--cem_elites", type=int, default=16)
    ap.add_argument("--memory_capacity", type=int, default=64)
    ap.add_argument("--reachability_threshold", type=float, default=0.35)
    ap.add_argument("--progress_window", type=int, default=5)
    ap.add_argument("--progress_threshold", type=float, default=0.02)
    ap.add_argument("--plan_cost_threshold", type=float, default=0.2,
                    help="Recoverability 'success' bar (recoverability_score's success_cost_threshold)")
    ap.add_argument("--trigger_cost_threshold", type=float, default=None,
                    help="Base-planner-is-failing bar for should_trigger_recovery. Distinct from "
                         "--plan_cost_threshold: that one asks 'is this candidate state good enough "
                         "to recover to', this one asks 'is the live planner doing badly enough right "
                         "now to invoke recovery'. Defaults to --plan_cost_threshold if unset.")
    ap.add_argument("--recoverability_samples", type=int, default=64)
    ap.add_argument("--recoverability_horizon", type=int, default=8)
    ap.add_argument("--top_k_memory", type=int, default=32)
    ap.add_argument("--goal_success_threshold", type=float, default=0.1)
    ap.add_argument("--collect_probe_rows", action="store_true")
    ap.add_argument("--collect_probe_max_states", type=int, default=1000)
    ap.add_argument("--collect_probe_bin_threshold", type=float, default=0.5)
    ap.add_argument("--collect_probe_optionality", action="store_true")
    ap.add_argument("--use_optionality_metric", action="store_true")
    ap.add_argument("--no_val_filter", action="store_true")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, train_args, mcfg = load_model_and_stats(args.checkpoint, args.device)
    img_size = args.img_size or train_args.get("img_size", 224)
    expected_act_dim = int(mcfg["act_dim"])

    print(f"Loading dataset: {args.dataset}")
    ds = HFWindowDataset(args.dataset, img_size=img_size, norm_stats=norm_stats)

    # Keep dataset action representation consistent with the trained model.
    # Only fall back to 6-dim velocity actions when the checkpoint actually
    # expects that -- forcing it unconditionally breaks checkpoints trained
    # on the dataset's native action vector (e.g. our nuscenes act_dim=13).
    if ds.D_act != expected_act_dim:
        if expected_act_dim == 6:
            ds.force_velocity_actions()
        if ds.D_act != expected_act_dim:
            raise ValueError(
                f"Action dim mismatch: checkpoint expects act_dim={expected_act_dim}, "
                f"dataset provides D_act={ds.D_act}."
            )
    print(f"Using action dim: {ds.D_act} (checkpoint expects {expected_act_dim})")

    if not args.no_val_filter:
        val_ids = set(eid for eid in ds._window_ids if eid % 5 == 0)
        ds._window_ids = [eid for eid in ds._window_ids if eid in val_ids]
        ds._window_rows = {k: v for k, v in ds._window_rows.items() if k in val_ids}
        print(f"Val episodes: {len(ds)}")
    else:
        print(f"All episodes (no val filter): {len(ds)}")

    summary = eval_mpc_recovery_replay(
        model=model,
        dataset=ds,
        norm_stats=norm_stats,
        device=args.device,
        horizon=args.horizon,
        num_episodes=args.num_episodes,
        cem_population=args.cem_population,
        cem_iters=args.cem_iters,
        cem_elites=args.cem_elites,
        memory_capacity=args.memory_capacity,
        reachability_threshold=args.reachability_threshold,
        progress_window=args.progress_window,
        progress_threshold=args.progress_threshold,
        plan_cost_threshold=args.plan_cost_threshold,
        trigger_cost_threshold=args.trigger_cost_threshold
        if args.trigger_cost_threshold is not None
        else args.plan_cost_threshold,
        recoverability_samples=args.recoverability_samples,
        recoverability_horizon=args.recoverability_horizon,
        top_k_memory=args.top_k_memory,
        goal_success_threshold=args.goal_success_threshold,
        collect_probe_rows=args.collect_probe_rows,
        collect_probe_max_states=args.collect_probe_max_states,
        collect_probe_bin_threshold=args.collect_probe_bin_threshold,
        collect_probe_optionality=args.collect_probe_optionality,
        use_optionality_metric=args.use_optionality_metric,
    )

    print("\n" + "=" * 60)
    print("MPC + Recoverability Replay Results")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, dict):
            if {"mean", "std", "min", "max"}.issubset(value.keys()):
                pct = ""
                if {"p25", "p50", "p75"}.issubset(value.keys()):
                    pct = (
                        f" p25={value['p25']:.4f} p50={value['p50']:.4f} "
                        f"p75={value['p75']:.4f}"
                    )
                print(
                    f"  {key}: mean={value['mean']:.4f} std={value['std']:.4f} "
                    f"min={value['min']:.4f} max={value['max']:.4f}{pct}"
                )
            else:
                print(f"  {key}:")
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, dict):
                        print(f"    {sub_k}: {json.dumps(sub_v)}")
                    else:
                        print(f"    {sub_k}: {sub_v}")
        else:
            print(f"  {key}: {value}")

    out_path = os.path.join(str(Path(args.checkpoint).parent), "mpc_recovery_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

    if args.collect_probe_rows and "collected_probe_rows" in summary:
        probe_path = os.path.join(
            str(Path(args.checkpoint).parent),
            "mpc_recovery_probe_rows.json",
        )
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(summary["collected_probe_rows"], f)
        print(f"Saved probe rows to {probe_path}")
        del summary["collected_probe_rows"]


if __name__ == "__main__":
    main()
