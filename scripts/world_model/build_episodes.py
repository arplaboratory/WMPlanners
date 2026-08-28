#!/usr/bin/env python3
"""Build .npz episodes from extracted bag data (output of extract_dataset.py).

Reads manifest.json + sharded images + CSVs produced by extract_dataset.py,
applies quality filtering (idle removal, timestamp gap detection),
time-synchronises all sensor modalities to image timestamps,
and slices into fixed-length episodes for world model training.

Usage:
    # After running extract_dataset.py:
    python -m scripts.world_model.build_episodes \
        --data_dir dataset_out \
        --out_dir data/race16 \
        --episode_len 64 \
        --stride 32 \
        --img_size 128

    # Then train:
    python -m scripts.world_model.train_world_model \
        --data_dir data/race16 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.world_model.sync_utils import nearest_merge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load an image, resize. Return uint8 RGB or float32 depth."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        # depth: 16-bit PNG → float metres (assume mm encoding)
        return img.astype(np.float32) / 1000.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resolve_path(rel_path: str, bag_dir: Path) -> str:
    """Resolve an image path from image_index.csv against the bag directory.

    Handles both absolute paths and relative paths like
    ``shard_0000/color_000000.jpg``.
    """
    if not rel_path or rel_path == "":
        return ""
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try directly under bag_dir
    candidate = bag_dir / rel_path
    if candidate.exists():
        return str(candidate)
    # Try just shard/filename (strip leading directory from CSV path)
    parts = p.parts
    if len(parts) >= 2:
        candidate = bag_dir / parts[-2] / parts[-1]
        if candidate.exists():
            return str(candidate)
    # Try filename in every shard
    for shard_dir in sorted(bag_dir.glob("shard_*")):
        candidate = shard_dir / p.name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot find image: {rel_path} under {bag_dir}")


# ---------------------------------------------------------------------------
# Quality filtering helpers
# ---------------------------------------------------------------------------

def _mark_idle(speed: np.ndarray, min_speed: float, idle_persistence: int) -> np.ndarray:
    """Return boolean array: True where frame is idle (consecutive run-length approach).

    A frame is marked idle only when it is part of a run of `idle_persistence`+
    consecutive frames all below `min_speed`.
    """
    n = len(speed)
    slow = speed < min_speed
    idle = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]:
                j += 1
            run_len = j - i
            if run_len >= idle_persistence:
                idle[i:j] = True
            i = j
        else:
            i += 1
    return idle


def _mark_gaps(timestamps: np.ndarray, gap_factor: float = 2.0) -> np.ndarray:
    """Return boolean array of length N: True at index i means a gap BEFORE frame i.

    A gap is defined as dt > gap_factor * median(dt). Index 0 is never marked.
    """
    if gap_factor <= 0:
        raise ValueError("gap_factor must be positive")

    n = len(timestamps)
    gap_before = np.zeros(n, dtype=bool)
    if n < 2:
        return gap_before
    dt = np.diff(timestamps.astype(np.float64))
    median_dt = np.median(dt)
    threshold = gap_factor * median_dt # Index 0 is never marked.
    gap_before[1:] = dt > threshold
    return gap_before


def _find_valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True in `valid`. Returns list of (start, end) exclusive."""
    runs = []
    n = len(valid)
    i = 0
    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _nearest_gap_ns(target_ts: np.ndarray, src_df: pd.DataFrame | None) -> np.ndarray:
    """Distance from each target timestamp to the nearest source timestamp."""
    if src_df is None or src_df.empty:
        return np.full(len(target_ts), np.inf, dtype=np.float64)

    src_ts = src_df["timestamp_ns"].values
    if len(src_ts) == 0:
        return np.full(len(target_ts), np.inf, dtype=np.float64)
    if len(src_ts) == 1:
        return np.abs(target_ts.astype(np.float64) - float(src_ts[0]))

    idx = np.searchsorted(src_ts, target_ts, side="left")
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left = np.abs(src_ts[idx - 1] - target_ts)
    right = np.abs(src_ts[idx] - target_ts)
    return np.minimum(left, right).astype(np.float64)


# ---------------------------------------------------------------------------
# Core build function
"""
An episode is one fixed-length, time-ordered chunk of robot data.
In this pipeline, build_episodes.py creates an episode by:
Taking a continuous run of valid camera frames—no large timestamp gaps, no idle section, and synchronized sensor data.
Selecting episode_len frames, optionally every frame_skip frames.
Saving the aligned data for those frames into one ep_XXXXX.npz file:RGB images
depth images
IMU readings
odometry / robot pose and velocity
action / command values

So with episode_len=50 and frame_skip=6, an episode contains 50 saved timesteps sampled from a 300-frame span of continuous robot operation.
The episode is the unit later scored, selected, uploaded, and used for world-model training.
"""

# ---------------------------------------------------------------------------

def build_episodes(
    data_dir: str,
    out_dir: str,
    episode_len: int = 64,
    stride: int = 32,
    img_size: int = 128,
    min_speed: float = 0.1,
    max_sync_gap_ms: float = 100.0,
    idle_persistence: int = 10,
    frame_skip: int = 1,
    image_gap_factor: float = 2.0,
) -> dict:
    """Build NPZ episodes from all bags in data_dir. Returns stats dict.

    With frame_skip > 1, every Nth image frame is selected, reducing the
    effective FPS and increasing episode temporal span.  Sensor readings
    are synced to image timestamps *before* subsampling, so each selected
    frame keeps its correctly paired sensor data.
    """
    data_dir = Path(data_dir)
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)

    max_gap_ns = int(max_sync_gap_ms * 1_000_000)
    img_sz = (img_size, img_size)

    # ── 1. Read manifest ──────────────────────────────────────────────────
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    bags = manifest.get("bags", [])

    configured_action_source = config.get("action", {}).get("source", "position_cmd")
    imu_dim = config.get("sensors", {}).get("imu_dim", 10)
    odom_dim = config.get("sensors", {}).get("odom_dim", 13)
    action_dim = config.get("action", {}).get("dim", 14)

    per_bag_counts: dict[str, int] = {}
    total_episodes = 0
    sensor_dims: dict[str, int] = {}
    ep_count = 0

    for bag_info in bags:
        bag_name = bag_info["name"]
        bag_rel = bag_info.get("path", f"{bag_name}/")
        bag_dir = data_dir / bag_rel.rstrip("/")

        logger.info("Processing bag: %s", bag_name)

        # ── 2. Load CSVs ─────────────────────────────────────────────────
        def _load_csv(name: str) -> pd.DataFrame | None:
            p = bag_dir / name
            if p.exists():
                df = pd.read_csv(p)
                if df.empty:
                    logger.warning("Empty %s in bag %s — will zero-fill", name, bag_name)
                    return None
                return df
            logger.warning("Missing %s in bag %s — will zero-fill", name, bag_name)
            return None

        img_index = _load_csv("image_index.csv")
        if img_index is None:
            logger.warning("No image_index.csv for bag %s — skipping", bag_name)
            per_bag_counts[bag_name] = 0
            continue

        odom_df = _load_csv("odom.csv")
        imu_df = _load_csv("imu.csv")
        action_df = _load_csv(f"{configured_action_source}.csv")

        # ── 3. Sort by timestamp ──────────────────────────────────────────
        img_index = img_index.sort_values("timestamp_ns").reset_index(drop=True)
        image_ts = img_index["timestamp_ns"].values
        N = len(image_ts)

        if odom_df is not None:
            odom_df = odom_df.sort_values("timestamp_ns").reset_index(drop=True)
        if imu_df is not None:
            imu_df = imu_df.sort_values("timestamp_ns").reset_index(drop=True)
        if action_df is not None:
            action_df = action_df.sort_values("timestamp_ns").reset_index(drop=True)

        # ── 4. Speed + idle detection ─────────────────────────────────────
        if odom_df is not None and all(
            c in odom_df.columns for c in ("linear_vel_x", "linear_vel_y", "linear_vel_z")
        ):
            # Sync odom to image timestamps to get per-frame speed
            odom_sync_all = nearest_merge(image_ts, odom_df, max_gap_ns=None)
            # Find column indices for linear_vel_x/y/z
            odom_val_cols = [c for c in odom_df.columns if c != "timestamp_ns"]
            try:
                vx_idx = odom_val_cols.index("linear_vel_x")
                vy_idx = odom_val_cols.index("linear_vel_y")
                vz_idx = odom_val_cols.index("linear_vel_z")
                speed = np.sqrt(
                    odom_sync_all[:, vx_idx] ** 2
                    + odom_sync_all[:, vy_idx] ** 2
                    + odom_sync_all[:, vz_idx] ** 2
                )
            except ValueError:
                speed = np.ones(N)  # can't compute speed; don't filter
        else:
            speed = np.ones(N)  # no odom → no speed filtering

        idle = _mark_idle(speed, min_speed, idle_persistence)

        # ── 5. Gap detection ──────────────────────────────────────────────
        # _mark_gaps finds unusually large time gaps between consecutive camera frames.
        # It:
        #   - Computes the time difference (dt) between each pair of timestamps.
        #   - Finds the typical interval using median(dt).
        #   - Marks frame i as having a gap before it when:
        #   - dt > gap_factor * median(dt)
        # With the default gap_factor=2.0, a frame is marked if it arrived more than twice as late as the typical frame interval.
        # Example: if camera frames normally arrive every 33 ms, any interval above 66 ms is a gap. 
        # The frame after that interval is marked invalid, preventing an episode from spanning missing/discontinuous data.
        gap_before = _mark_gaps(image_ts, gap_factor=image_gap_factor)

        # ── 6. Valid mask + contiguous runs ───────────────────────────────
        # A frame is invalid if it's idle OR if there's a gap right before it
        odom_synced = np.ones(N, dtype=bool)
        if odom_df is not None:
            odom_synced = _nearest_gap_ns(image_ts, odom_df) <= max_gap_ns

        action_synced = np.ones(N, dtype=bool)
        if action_df is not None:
            action_synced = _nearest_gap_ns(image_ts, action_df) <= max_gap_ns

        valid = ~idle & ~gap_before & odom_synced & action_synced
        runs = _find_valid_runs(valid)

        # ── 7. Sync all sensors to image timestamps (once for the bag) ────
        # Sensor streams often have different sampling rates.  Aligning them
        # here ensures every saved image frame has its corresponding action,
        # pose, and IMU values.
        if odom_df is not None:
            odom_sync = nearest_merge(image_ts, odom_df, max_gap_ns=max_gap_ns)
            sensor_dims["odom"] = odom_sync.shape[1]
        else:
            odom_sync = np.zeros((N, odom_dim), dtype=np.float32)
            sensor_dims["odom"] = odom_dim

        if imu_df is not None:
            imu_sync = nearest_merge(image_ts, imu_df, max_gap_ns=max_gap_ns)
            sensor_dims["imu"] = imu_sync.shape[1]
        else:
            imu_sync = np.zeros((N, imu_dim), dtype=np.float32)
            sensor_dims["imu"] = imu_dim

        if action_df is not None:
            action_sync = nearest_merge(image_ts, action_df, max_gap_ns=max_gap_ns)
            sensor_dims["action"] = action_sync.shape[1]
        else:
            action_sync = np.zeros((N, action_dim), dtype=np.float32)
            sensor_dims["action"] = action_dim

        color_paths = img_index["color_path"].values
        depth_paths = img_index["depth_path"].fillna("").values

        # ── 8. Slice episodes within each valid run ───────────────────────
        # With frame_skip, each episode spans (episode_len * frame_skip) raw
        # frames but only keeps every frame_skip-th one.  Stride is applied
        # in raw-frame space so the user can control overlap independently.
        raw_window = episode_len * frame_skip  # raw frames needed per episode
        bag_ep_count = 0
        for run_start, run_end in runs:
            run_len = run_end - run_start
            if run_len < raw_window:
                continue

            starts = range(run_start, run_end - raw_window + 1, stride * frame_skip)
            for start in starts:
                # Indices of the frames we actually keep
                frame_indices = list(range(start, start + raw_window, frame_skip))
                assert len(frame_indices) == episode_len

                rgb = np.zeros((episode_len, img_size, img_size, 3), dtype=np.uint8)
                depth = np.zeros((episode_len, img_size, img_size), dtype=np.float32)

                ok = True
                for t, frame_idx in enumerate(frame_indices):
                    try:
                        rgb[t] = _load_image(
                            _resolve_path(color_paths[frame_idx], bag_dir), img_sz
                        )
                        dp = depth_paths[frame_idx]
                        if dp:
                            depth[t] = _load_image(
                                _resolve_path(dp, bag_dir), img_sz
                            )
                    except FileNotFoundError as e:
                        logger.warning("Skipping episode at frame %d: %s", start, e)
                        ok = False
                        break
                if not ok:
                    continue

                # Sensor arrays: pick the same subsampled indices so RGB,
                # depth, action, odometry, and IMU remain frame-aligned.
                idx_arr = np.array(frame_indices)
                out_file = out_path / f"ep_{ep_count:05d}.npz"
                np.savez_compressed(
                    str(out_file),
                    rgb=rgb,                       # (T, H, W, 3) uint8
                    depth=depth,                   # (T, H, W) float32
                    imu=imu_sync[idx_arr],         # (T, D_imu) float32
                    odom=odom_sync[idx_arr],       # (T, D_odom) float32
                    action=action_sync[idx_arr],   # (T, D_act) float32
                )
                ep_count += 1
                bag_ep_count += 1

        per_bag_counts[bag_name] = bag_ep_count
        total_episodes += bag_ep_count
        logger.info("  Bag %s: wrote %d episodes", bag_name, bag_ep_count)

    # ── 9. Write metadata.json ────────────────────────────────────────────
    metadata = {
        "total_episodes": total_episodes,
        "per_bag_counts": per_bag_counts,
        "sensor_dims": sensor_dims,
        "config": config,
        "quality_params": {
            "min_speed": min_speed,
            "max_sync_gap_ms": max_sync_gap_ms,
            "idle_persistence": idle_persistence,
            "configured_action_source": configured_action_source,
            "image_gap_factor": image_gap_factor,
        },
        "episode_params": {
            "episode_len": episode_len,
            "stride": stride,
            "img_size": img_size,
            "frame_skip": frame_skip,
        },
    }
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {total_episodes} episodes to {out_dir}")
    return {"total_episodes": total_episodes, "per_bag_counts": per_bag_counts}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build .npz episodes from extracted bag data",
    )
    ap.add_argument(
        "--data_dir", required=True,
        help="Directory with manifest.json from extract_dataset.py",
    )
    ap.add_argument("--out_dir", required=True, help="Where to write .npz episodes")
    ap.add_argument("--episode_len", type=int, default=64,
                    help="Timesteps per episode (default: 64)")
    ap.add_argument("--stride", type=int, default=32,
                    help="Stride between episode starts (default: 32)")
    ap.add_argument("--img_size", type=int, default=128,
                    help="Resize images to (img_size, img_size) (default: 128)")
    ap.add_argument("--min_speed", type=float, default=0.1,
                    help="Minimum speed threshold to exclude idle frames (default: 0.1)")
    ap.add_argument("--max_sync_gap_ms", type=float, default=100.0,
                    help="Max sensor sync gap in milliseconds (default: 100.0)")
    ap.add_argument("--idle_persistence", type=int, default=10,
                    help="Min consecutive slow frames to mark as idle (default: 10)")
    ap.add_argument("--frame_skip", type=int, default=1,
                    help="Take every Nth frame, reducing FPS (default: 1 = no skip)")
    ap.add_argument("--image_gap_factor", type=float, default=2.0,
                    help=(
                        "Image stream gap threshold as a multiple of median image dt "
                        "(default: 2.0). Use larger values for jittery rover cameras."
                    ))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    stats = build_episodes(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        episode_len=args.episode_len,
        stride=args.stride,
        img_size=args.img_size,
        min_speed=args.min_speed,
        max_sync_gap_ms=args.max_sync_gap_ms,
        idle_persistence=args.idle_persistence,
        frame_skip=args.frame_skip,
        image_gap_factor=args.image_gap_factor,
    )
    print(f"Wrote {stats['total_episodes']} episodes")


if __name__ == "__main__":
    main()
