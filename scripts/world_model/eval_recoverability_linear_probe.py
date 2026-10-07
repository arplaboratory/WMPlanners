from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from world_model.architecture import WorldModel, checkpoint_backbone_kwargs
from world_model.data import HFWindowDataset
from world_model.data.normalization import NormStats
from world_model.mpc import CEMConfig
from world_model.recovery import RecoveryConfig, recoverability_score


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


def make_goal_tracking_cost(w_goal: float = 1.0, w_smooth: float = 0.1):
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


@torch.no_grad()
def build_probe_dataset(
    model,
    dataset: HFWindowDataset,
    device: str,
    max_states: int,
    recoverability_horizon: int,
    recoverability_samples: int,
    success_cost_threshold: float,
    use_optionality_metric: bool,
):
    cem_cfg = CEMConfig(
        horizon=max(1, recoverability_horizon),
        iters=1,
        population=max(8, recoverability_samples),
        elites=max(4, min(recoverability_samples // 2, recoverability_samples)),
        action_dim=dataset.D_act,
        action_min=-3.0,
        action_max=3.0,
        init_std=1.0,
        alpha=0.25,
    )
    rec_cfg = RecoveryConfig(
        recoverability_samples=recoverability_samples,
        recoverability_horizon=recoverability_horizon,
        success_cost_threshold=success_cost_threshold,
    )
    cost_fn = make_goal_tracking_cost()

    rows = []
    for ds_idx in range(len(dataset)):
        sample = dataset[ds_idx]
        eid = int(dataset._window_ids[ds_idx])

        rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(device)
        depth = torch.from_numpy(sample["depth"]).unsqueeze(0).to(device)
        odom = torch.from_numpy(sample["odom"]).unsqueeze(0).to(device)
        imu = torch.from_numpy(sample["imu"]).unsqueeze(0).to(device)

        total_frames = odom.shape[1]
        max_t = max(1, total_frames - recoverability_horizon - 1)

        for t in range(max_t):
            out = model.encode(rgb[:, t], depth[:, t], odom[:, t], imu[:, t])
            z_t = out.z.squeeze(0)

            goal_seq = odom[:, t + 1 : t + 1 + recoverability_horizon, :3]
            if goal_seq.shape[1] < recoverability_horizon:
                break

            r = recoverability_score(
                model=model,
                z=z_t,
                goal=goal_seq,
                cost_fn=cost_fn,
                cem_cfg=cem_cfg,
                cfg=rec_cfg,
                device=torch.device(device),
                use_optionality_metric=use_optionality_metric,
            )

            rows.append(
                {
                    "episode_id": eid,
                    "t": int(t),
                    "z": z_t.detach().cpu().numpy().astype(np.float32),
                    "recoverability": float(r),
                }
            )

            if len(rows) >= max_states:
                return rows

    return rows


def evaluate_probe(rows, bin_threshold: float):
    if not rows:
        raise RuntimeError("No probe rows were produced.")

    X = np.stack([r["z"] for r in rows], axis=0)
    y = np.array([r["recoverability"] for r in rows], dtype=np.float32)
    eid = np.array([r["episode_id"] for r in rows], dtype=np.int64)

    train_mask = (eid % 5) != 0
    test_mask = ~train_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("Train/test split is empty. Need both train and held-out episodes.")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    reg = make_pipeline(StandardScaler(), LinearRegression())
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    summary = {
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

    # Non-linear regressor probe (MLP)
    mlp_reg = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=2000,
            random_state=0,
            early_stopping=True,
        ),
    )
    mlp_reg.fit(X_train, y_train)
    y_pred_mlp = mlp_reg.predict(X_test)
    summary["nonlinear_regression"] = {
        "r2": float(r2_score(y_test, y_pred_mlp)),
        "mae": float(mean_absolute_error(y_test, y_pred_mlp)),
        "corr": float(np.corrcoef(y_test, y_pred_mlp)[0, 1]) if np.std(y_pred_mlp) > 0 else 0.0,
    }

    threshold = float(np.median(y_train)) if np.isnan(bin_threshold) else float(bin_threshold)

    yb_train = (y_train >= threshold).astype(np.int64)
    yb_test = (y_test >= threshold).astype(np.int64)

    if np.unique(yb_train).size > 1 and np.unique(yb_test).size > 1:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_train, yb_train)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)

        summary["linear_classification"] = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(yb_test, pred)),
            "auroc": float(roc_auc_score(yb_test, prob)),
            "positive_rate_test": float(yb_test.mean()),
        }

        mlp_clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=2000,
                random_state=0,
                early_stopping=True,
            ),
        )
        mlp_clf.fit(X_train, yb_train)
        prob_mlp = mlp_clf.predict_proba(X_test)[:, 1]
        pred_mlp = (prob_mlp >= 0.5).astype(np.int64)
        summary["nonlinear_classification"] = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(yb_test, pred_mlp)),
            "auroc": float(roc_auc_score(yb_test, prob_mlp)),
            "positive_rate_test": float(yb_test.mean()),
        }
    else:
        summary["linear_classification"] = {
            "threshold": threshold,
            "skipped": True,
            "reason": "Binary labels collapsed in train or test split",
            "positive_rate_test": float(yb_test.mean()),
        }
        summary["nonlinear_classification"] = {
            "threshold": threshold,
            "skipped": True,
            "reason": "Binary labels collapsed in train or test split",
            "positive_rate_test": float(yb_test.mean()),
        }

    return summary


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def parse_target_modes(raw: str) -> list[str]:
    valid = {"success_fraction", "optionality"}
    vals = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok:
            vals.append(tok)
    if not vals:
        raise ValueError("Expected at least one target mode")
    bad = [v for v in vals if v not in valid]
    if bad:
        raise ValueError(f"Unknown target mode(s): {bad}. Valid: {sorted(valid)}")
    return vals


def main():
    ap = argparse.ArgumentParser(description="Linear probe for recoverability from latent states")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_states", type=int, default=1000)
    ap.add_argument("--recoverability_horizon", type=int, default=8)
    ap.add_argument("--recoverability_samples", type=int, default=32)
    ap.add_argument("--success_cost_threshold", type=float, default=0.2)
    ap.add_argument(
        "--bin_threshold",
        type=float,
        default=float("nan"),
        help="Binary threshold for classification probe; defaults to median train recoverability if omitted.",
    )
    ap.add_argument("--use_optionality_metric", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument(
        "--sweep_horizons",
        type=str,
        default="8,16",
        help="Comma-separated recoverability_horizon values for sweep mode.",
    )
    ap.add_argument(
        "--sweep_samples",
        type=str,
        default="32,64",
        help="Comma-separated recoverability_samples values for sweep mode.",
    )
    ap.add_argument(
        "--sweep_target_modes",
        type=str,
        default="success_fraction,optionality",
        help="Comma-separated target modes: success_fraction,optionality",
    )
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, train_args, _ = load_model_and_stats(args.checkpoint, args.device)
    img_size = args.img_size or train_args.get("img_size", 224)

    print(f"Loading dataset: {args.dataset}")
    ds = HFWindowDataset(args.dataset, img_size=img_size, norm_stats=norm_stats)
    ds.force_velocity_actions()

    if not args.sweep:
        print("Building latent/recoverability probe dataset...")
        rows = build_probe_dataset(
            model=model,
            dataset=ds,
            device=args.device,
            max_states=args.max_states,
            recoverability_horizon=args.recoverability_horizon,
            recoverability_samples=args.recoverability_samples,
            success_cost_threshold=args.success_cost_threshold,
            use_optionality_metric=args.use_optionality_metric,
        )

        print(f"Collected states: {len(rows)}")
        summary = evaluate_probe(rows, bin_threshold=args.bin_threshold)

        print("\n" + "=" * 60)
        print("Recoverability Probe")
        print("=" * 60)
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

        out_path = os.path.join(str(Path(args.checkpoint).parent), "recoverability_linear_probe.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {out_path}")
        return

    horizons = parse_int_list(args.sweep_horizons)
    samples = parse_int_list(args.sweep_samples)
    target_modes = parse_target_modes(args.sweep_target_modes)

    sweep_results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "max_states": args.max_states,
        "success_cost_threshold": args.success_cost_threshold,
        "bin_threshold": None if np.isnan(args.bin_threshold) else float(args.bin_threshold),
        "configs": [],
    }

    combos = list(itertools.product(horizons, samples, target_modes))
    print(f"Running probe sweep with {len(combos)} configurations...")
    for i, (h, s, mode) in enumerate(combos, start=1):
        use_optionality = mode == "optionality"
        print(f"[{i}/{len(combos)}] horizon={h} samples={s} target={mode}")

        rows = build_probe_dataset(
            model=model,
            dataset=ds,
            device=args.device,
            max_states=args.max_states,
            recoverability_horizon=h,
            recoverability_samples=s,
            success_cost_threshold=args.success_cost_threshold,
            use_optionality_metric=use_optionality,
        )
        summary = evaluate_probe(rows, bin_threshold=args.bin_threshold)
        summary["config"] = {
            "recoverability_horizon": h,
            "recoverability_samples": s,
            "target_mode": mode,
        }
        sweep_results["configs"].append(summary)

    out_path = os.path.join(str(Path(args.checkpoint).parent), "recoverability_probe_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    main()
