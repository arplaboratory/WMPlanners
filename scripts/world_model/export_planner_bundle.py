"""Write a planner bundle so the ROS bridge can run a relative-pose checkpoint.

The bundle is a small YAML next to the checkpoint that records what the bridge
needs beyond the weights: the pose decoder, the dynamics step, and the CEM
action bounds. Bounds are the per-dimension min/max/mean/std of the
``[v_forward, v_lateral, yaw_rate]`` actions in ``--bounds_dataset`` (use data
from the robot that will run the planner), computed from odometry only.

Usage:
    python scripts/world_model/export_planner_bundle.py \\
        --checkpoint runs/wm_rel5hz_standard_pdrop50_varboth/best.pt \\
        --state_decoder runs/wm_rel5hz_standard_pdrop50_varboth/real_eval/real_state_decoder.pt \\
        --bounds_dataset /media/4t/datasets/wm-phoenix-real-run03_windowed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from world_model.data import open_window_dataset
from world_model.data.parquet_dataset import _body_velocity_actions

BUNDLE_FORMAT = "relative_pose_v1"


def action_stats(dataset) -> dict[str, list[float]]:
    dataset.recording_cache_size = 10_000  # windows are not grouped by recording
    actions = []
    for window in dataset.windows:
        field = lambda name, default=None: dataset._field(window, name, "windows", default)  # noqa: E731
        frames = dataset._recording_frames(str(field("recording_id")))
        first, step = int(field("first_frame_index")), int(field("frame_step", 1))
        rows = [frames[first + k * step] for k in range(int(field("length")))]
        odom = np.asarray([dataset._field(r, "odom", "frames") for r in rows], dtype=np.float32)
        stamps = np.asarray([int(dataset._field(r, "timestamp_ns", "frames", 0) or 0) for r in rows], dtype=np.int64)
        actions.append(_body_velocity_actions(odom, stamps))
    a = np.concatenate(actions)
    return {
        "low": a.min(0).tolist(), "high": a.max(0).tolist(),
        "init_mean": a.mean(0).tolist(), "init_std": np.maximum(a.std(0), 1e-3).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--state_decoder", required=True, help="PoseDecoder from train_state_decoder.py")
    ap.add_argument("--bounds_dataset", required=True, help="Parquet dataset for action bounds (the target robot's data)")
    ap.add_argument("--bounds_split", default="train")
    ap.add_argument("--horizon", type=int, default=None, help="Planning steps (default: the checkpoint's training horizon)")
    ap.add_argument("--output", default=None, help="Default: <checkpoint dir>/planner_bundle.yaml")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    schema_config = train_args.get("schema_config")
    if not schema_config:
        raise ValueError("checkpoint has no schema_config; it was not trained on relative-pose inputs")
    dataset = open_window_dataset(dataset_type="hf", path=args.bounds_dataset, split=args.bounds_split,
                                  schema_config=schema_config)
    sensors = dataset.sensors
    if sensors.get("odom_frame") != "window" or sensors.get("action_source") != "body_velocity":
        raise ValueError(f"{schema_config} is not a relative-pose config (odom_frame: window, action_source: body_velocity)")

    stamps = np.asarray([int(dataset._field(r, "timestamp_ns", "frames", 0) or 0)
                         for r in dataset._recording_frames(str(dataset._field(dataset.windows[0], "recording_id", "windows"))).values()])
    step_dt = float(np.median(np.diff(np.sort(stamps)))) / 1e9 * int(dataset._field(dataset.windows[0], "frame_step", "windows", 1))

    output = Path(args.output or Path(args.checkpoint).parent / "planner_bundle.yaml")
    rel = lambda p: os.path.relpath(Path(p).resolve(), output.parent.resolve())  # noqa: E731
    bundle = {
        "format": BUNDLE_FORMAT,
        "checkpoint": rel(args.checkpoint),
        "state_decoder": rel(args.state_decoder),
        "step_dt": round(step_dt, 4),
        "horizon": int(args.horizon or train_args.get("horizon", 16)),
        "sensors": {"depth": bool(sensors.get("depth", True)), "imu": bool(sensors.get("imu", True))},
        "action": {"names": ["v_forward", "v_lateral", "yaw_rate"], **action_stats(dataset)},
        "source": {"schema_config": schema_config, "bounds_dataset": str(args.bounds_dataset), "bounds_split": args.bounds_split},
    }
    output.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(bundle, sort_keys=False))
    print(f"Saved planner bundle to {output}")


if __name__ == "__main__":
    main()
