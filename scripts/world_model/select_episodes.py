"""Episode selector with generic and Spot-oriented diversity profiles.

Selects a diverse subset of NPZ episodes for world model training,
prioritising action/dynamics coverage over visual diversity.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler

GENERIC_PROFILE = "generic"
SPOT_PROFILE = "spot"
ROVER_PROFILE = "rover"


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------

def compute_episode_odom_delta(odom: np.ndarray) -> np.ndarray | None:
    """Compute start-to-end body-frame displacement from odom array.

    odom: (T, D_odom) where first 7 cols are pos_x,y,z, orient_x,y,z,w.
    Returns: (6,) array [dx, dy, dz, ax, ay, az] in body frame (axis-angle rotation).
             Returns None if start/end quaternions are invalid (zero norm).
    """
    pos_start, pos_end = odom[0, :3], odom[-1, :3]
    quat_start = odom[0, 3:7]  # (x, y, z, w) — scipy format
    quat_end = odom[-1, 3:7]

    # Check for zero-norm quaternions which cause scipy.spatial.transform.Rotation to fail
    if np.allclose(np.linalg.norm(quat_start), 0) or np.allclose(np.linalg.norm(quat_end), 0):
        return None

    try:
        r_start = Rotation.from_quat(quat_start)
        r_end = Rotation.from_quat(quat_end)
        dp_world = pos_end - pos_start
        dp_body = r_start.inv().apply(dp_world)
        r_delta = r_start.inv() * r_end
        axis_angle = r_delta.as_rotvec()
        return np.concatenate([dp_body, axis_angle])
    except ValueError:
        return None


def _safe_scale(block: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(block)


def _action_smoothness(action: np.ndarray) -> np.ndarray:
    if len(action) < 2:
        return np.zeros(2, dtype=np.float32)
    deltas = np.diff(action, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    return np.array([norms.mean(), norms.std()], dtype=np.float32)


def _spot_motion_features(odom: np.ndarray) -> np.ndarray:
    """Features that describe legged-robot base motion.

    Uses only odom columns already available in the current pipeline:
    - linear velocity: cols 7:10
    - angular velocity: cols 10:13
    """
    lin_vel = odom[:, 7:10] if odom.shape[1] >= 10 else np.zeros((len(odom), 3), dtype=np.float32)
    ang_vel = odom[:, 10:13] if odom.shape[1] >= 13 else np.zeros((len(odom), 3), dtype=np.float32)
    speed = np.linalg.norm(lin_vel, axis=1)
    yaw_rate = ang_vel[:, 2] if ang_vel.shape[1] >= 3 else np.zeros(len(odom), dtype=np.float32)
    return np.concatenate([
        lin_vel.mean(axis=0),
        lin_vel.std(axis=0),
        ang_vel.mean(axis=0),
        ang_vel.std(axis=0),
        np.array([speed.mean(), speed.std(), yaw_rate.mean(), yaw_rate.std()], dtype=np.float32),
    ])


def _spot_imu_features(imu: np.ndarray | None) -> np.ndarray:
    if imu is None or imu.ndim != 2 or imu.shape[1] < 10:
        return np.zeros(10, dtype=np.float32)
    gyro = imu[:, 4:7]
    accel = imu[:, 7:10]
    gyro_norm = np.linalg.norm(gyro, axis=1)
    accel_norm = np.linalg.norm(accel, axis=1)
    return np.concatenate([
        gyro.mean(axis=0),
        gyro.std(axis=0),
        np.array([gyro_norm.mean(), gyro_norm.std()], dtype=np.float32),
        np.array([accel_norm.mean(), accel_norm.std()], dtype=np.float32),
    ])


def _rover_motion_features(odom: np.ndarray, odom_delta: np.ndarray) -> np.ndarray:
    """Planar motion features for ground rover episode diversity."""
    lin_vel = odom[:, 7:10] if odom.shape[1] >= 10 else np.zeros((len(odom), 3), dtype=np.float32)
    ang_vel = odom[:, 10:13] if odom.shape[1] >= 13 else np.zeros((len(odom), 3), dtype=np.float32)

    forward = lin_vel[:, 0]
    lateral = lin_vel[:, 1]
    yaw_rate = ang_vel[:, 2]
    planar_speed = np.linalg.norm(lin_vel[:, :2], axis=1)
    planar_distance = float(np.linalg.norm(odom_delta[:2]))
    abs_yaw = abs(float(odom_delta[5]))
    curvature_proxy = abs_yaw / max(planar_distance, 1e-6)

    return np.array([
        forward.mean(),
        forward.std(),
        lateral.mean(),
        lateral.std(),
        planar_speed.mean(),
        planar_speed.std(),
        np.max(planar_speed) if len(planar_speed) else 0.0,
        yaw_rate.mean(),
        yaw_rate.std(),
        np.max(np.abs(yaw_rate)) if len(yaw_rate) else 0.0,
        float(np.mean(planar_speed < 0.05)),
        float(np.mean(forward < -0.05)),
        planar_distance,
        abs_yaw,
        curvature_proxy,
    ], dtype=np.float32)


def _weighted_concat(blocks: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(blocks) != len(weights):
        raise ValueError("blocks and weights must have the same length")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    scaled_blocks = []
    for block, weight in zip(blocks, weights):
        if block.size == 0:
            continue
        scaled_blocks.append(_safe_scale(block) * math.sqrt(weight / total_weight))
    return np.concatenate(scaled_blocks, axis=1)


def compute_spot_descriptors(data_dir: str) -> Tuple[np.ndarray, List[str]]:
    """Compute Spot-oriented descriptors for farthest-point sampling."""
    pattern = os.path.join(data_dir, "ep_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz files found in {data_dir!r}")

    action_blocks = []
    odom_blocks = []
    motion_blocks = []
    smooth_blocks = []
    imu_blocks = []
    filenames: list[str] = []
    skipped = 0

    for fpath in files:
        data = np.load(fpath)
        action = data["action"]
        odom = data["odom"]
        imu = data["imu"] if "imu" in data.files else None

        odom_delta = compute_episode_odom_delta(odom)
        if odom_delta is None:
            skipped += 1
            continue

        action_blocks.append(np.concatenate([action.mean(axis=0), action.std(axis=0)]))
        odom_blocks.append(odom_delta.astype(np.float32))
        motion_blocks.append(_spot_motion_features(odom).astype(np.float32))
        smooth_blocks.append(_action_smoothness(action))

        imu_feat = _spot_imu_features(imu)
        if imu_feat.size > 0:
            imu_blocks.append(imu_feat.astype(np.float32))

        filenames.append(os.path.basename(fpath))

    if skipped > 0:
        print(f"Skipped {skipped} episodes due to invalid odom/quaternion data.")
    if not filenames:
        raise ValueError(f"No valid episodes found in {data_dir!r} (all skipped).")

    blocks = [np.stack(action_blocks), np.stack(odom_blocks), np.stack(motion_blocks), np.stack(smooth_blocks)]
    weights = [0.22, 0.20, 0.30, 0.18]
    if imu_blocks:
        blocks.append(np.stack(imu_blocks))
        weights.append(0.10)
    return _weighted_concat(blocks, weights), filenames


def compute_rover_descriptors(data_dir: str) -> Tuple[np.ndarray, List[str]]:
    """Compute planar rover descriptors for farthest-point sampling."""
    pattern = os.path.join(data_dir, "ep_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz files found in {data_dir!r}")

    action_blocks = []
    planar_delta_blocks = []
    motion_blocks = []
    smooth_blocks = []
    filenames: list[str] = []
    skipped = 0

    for fpath in files:
        data = np.load(fpath)
        action = data["action"]
        odom = data["odom"]

        odom_delta = compute_episode_odom_delta(odom)
        if odom_delta is None:
            skipped += 1
            continue

        action_blocks.append(np.concatenate([action.mean(axis=0), action.std(axis=0)]))
        planar_delta_blocks.append(np.array([
            odom_delta[0],
            odom_delta[1],
            odom_delta[5],
            np.linalg.norm(odom_delta[:2]),
        ], dtype=np.float32))
        motion_blocks.append(_rover_motion_features(odom, odom_delta))
        smooth_blocks.append(_action_smoothness(action))
        filenames.append(os.path.basename(fpath))

    if skipped > 0:
        print(f"Skipped {skipped} episodes due to invalid odom/quaternion data.")
    if not filenames:
        raise ValueError(f"No valid episodes found in {data_dir!r} (all skipped).")

    blocks = [
        np.stack(action_blocks),
        np.stack(planar_delta_blocks),
        np.stack(motion_blocks),
        np.stack(smooth_blocks),
    ]
    weights = [0.25, 0.25, 0.35, 0.15]
    return _weighted_concat(blocks, weights), filenames


def compute_descriptors(
    data_dir: str,
    encoder: str = "none",
    profile: str = GENERIC_PROFILE,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-episode descriptors.

    Returns: (N, D) descriptor array and list of episode filenames.

    For encoder="none":
      descriptor = concat(mean(action), std(action), odom_delta)
      shape = (2*D_act + 6,)
    """
    if encoder != "none":
        raise NotImplementedError(
            f"Visual encoder '{encoder}' is not yet implemented. Use encoder='none'."
        )

    if profile == SPOT_PROFILE:
        return compute_spot_descriptors(data_dir)
    if profile == ROVER_PROFILE:
        return compute_rover_descriptors(data_dir)
    if profile != GENERIC_PROFILE:
        raise ValueError(f"Unknown selection profile: {profile!r}")

    pattern = os.path.join(data_dir, "ep_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz files found in {data_dir!r}")

    descriptors = []
    filenames = []
    skipped = 0
    for fpath in files:
        data = np.load(fpath)
        action = data["action"]   # (T, D_act)
        odom = data["odom"]       # (T, D_odom)

        action_stats = np.concatenate([action.mean(axis=0), action.std(axis=0)])  # (2*D_act,)
        odom_delta = compute_episode_odom_delta(odom)                             # (6,)

        if odom_delta is None:
            skipped += 1
            continue

        descriptor = np.concatenate([action_stats, odom_delta])

        descriptors.append(descriptor)
        filenames.append(os.path.basename(fpath))

    if skipped > 0:
        print(f"Skipped {skipped} episodes due to invalid odom/quaternion data.")

    if not descriptors:
        raise ValueError(f"No valid episodes found in {data_dir!r} (all skipped).")

    return np.stack(descriptors, axis=0), filenames


# ---------------------------------------------------------------------------
# FPS core
# ---------------------------------------------------------------------------

def _farthest_point_sampling(descriptors: np.ndarray, target_count: int) -> List[int]:
    n = len(descriptors)
    if n == 0:
        return []
    if target_count >= n:
        return list(range(n))
    centroid = descriptors.mean(axis=0)
    dists_to_centroid = np.linalg.norm(descriptors - centroid, axis=1)
    first_idx = int(np.argmax(dists_to_centroid))
    selected = [first_idx]
    min_dists = np.linalg.norm(descriptors - descriptors[first_idx], axis=1)
    min_dists[first_idx] = -1.0
    while len(selected) < target_count:
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            break
        selected.append(next_idx)
        new_dists = np.linalg.norm(descriptors - descriptors[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[next_idx] = -1.0
    return selected


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def select_episodes(
    data_dir: str,
    out_dir: str,
    target_count: int,
    encoder: str = "none",
    dynamics_weight: float = 0.6,
    symlink: bool = False,
    profile: str = GENERIC_PROFILE,
) -> dict:
    """Select diverse episodes via dynamics-weighted FPS. Returns stats dict."""

    # 1. Compute descriptors
    raw_descs, filenames = compute_descriptors(data_dir, encoder, profile)
    n = len(filenames)

    # 2. Run FPS. Robot-specific descriptors are already block-scaled and weighted.
    if profile in {SPOT_PROFILE, ROVER_PROFILE}:
        selected_indices = _farthest_point_sampling(raw_descs, target_count)
        weighted = raw_descs
    else:
        #    For encoder="none": layout is [action_mean | action_std | odom_delta]
        #    odom_delta is always the last 6 columns
        odom_dim = 6
        action_stats_block = raw_descs[:, :-odom_dim]   # (N, 2*D_act)
        odom_block = raw_descs[:, -odom_dim:]            # (N, 6)

        # 3. Z-score normalise each block independently
        action_scaled = _safe_scale(action_stats_block)
        odom_scaled = _safe_scale(odom_block)

        # 4. Apply weights via sqrt(weight) multiplication
        if encoder == "none":
            action_weight = 0.75
            odom_weight = 0.25
            weighted = np.concatenate([
                action_scaled * math.sqrt(action_weight),
                odom_scaled * math.sqrt(odom_weight),
            ], axis=1)
        else:
            action_weight = dynamics_weight
            odom_weight = (1.0 - dynamics_weight) / 2.0
            visual_weight = (1.0 - dynamics_weight) / 2.0
            # visual block would be appended here; encoder != "none" raises earlier
            weighted = np.concatenate([
                action_scaled * math.sqrt(action_weight),
                odom_scaled * math.sqrt(odom_weight),
            ], axis=1)

        # 5 & 6. Run FPS
        selected_indices = _farthest_point_sampling(weighted, target_count)

    # 7. Create out_dir and copy/symlink selected episodes
    #    Remove stale episodes from a previous run first.
    if os.path.isdir(out_dir):
        for old in glob.glob(os.path.join(out_dir, "ep_*.npz")):
            os.remove(old)
    os.makedirs(out_dir, exist_ok=True)
    for new_idx, src_idx in enumerate(selected_indices):
        src_path = os.path.join(data_dir, filenames[src_idx])
        dst_name = f"ep_{new_idx:05d}.npz"
        dst_path = os.path.join(out_dir, dst_name)
        if symlink:
            if os.path.exists(dst_path) or os.path.islink(dst_path):
                os.remove(dst_path)
            os.symlink(os.path.abspath(src_path), dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # 8. Copy metadata.json if present, augmenting with selection params
    src_meta = os.path.join(data_dir, "metadata.json")
    dst_meta = os.path.join(out_dir, "metadata.json")
    if os.path.exists(src_meta):
        with open(src_meta) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["selection_params"] = {
        "data_dir": data_dir,
        "target_count": target_count,
        "encoder": encoder,
        "profile": profile,
        "dynamics_weight": dynamics_weight,
        "symlink": symlink,
        "selected_count": len(selected_indices),
        "total_count": n,
    }
    with open(dst_meta, "w") as f:
        json.dump(meta, f, indent=2)

    # 9. Compute FPS spread statistics and write selection_stats.json
    #    Distance of each selected point from all others (min-distance spread)
    selected_descs = weighted[selected_indices]
    if len(selected_indices) > 1:
        fps_distances = []
        for i, idx in enumerate(selected_indices):
            others = np.delete(selected_descs, i, axis=0)
            min_d = float(np.linalg.norm(selected_descs[i] - others, axis=1).min())
            fps_distances.append(min_d)
        fps_mean = float(np.mean(fps_distances))
        fps_std = float(np.std(fps_distances))
    else:
        fps_mean = 0.0
        fps_std = 0.0

    stats_path = os.path.join(out_dir, "selection_stats.json")
    selection_stats = {
        "selected": len(selected_indices),
        "total": n,
        "fps_distance_mean": fps_mean,
        "fps_distance_std": fps_std,
        "encoder": encoder,
        "profile": profile,
        "dynamics_weight": dynamics_weight,
        "selected_filenames": [filenames[i] for i in selected_indices],
    }
    with open(stats_path, "w") as f:
        json.dump(selection_stats, f, indent=2)

    # 10. Return summary stats
    return {"selected": len(selected_indices), "total": n}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select diverse NPZ episodes via dynamics-weighted FPS"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument(
        "--encoder", default="none", choices=["none", "cnn", "dinov2", "vjepa2"]
    )
    parser.add_argument(
        "--profile",
        default=GENERIC_PROFILE,
        choices=[GENERIC_PROFILE, SPOT_PROFILE, ROVER_PROFILE],
        help="Selection profile. Use 'rover' for planar ground-robot diversity.",
    )
    parser.add_argument("--dynamics-weight", type=float, default=0.6)
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()

    stats = select_episodes(
        args.data_dir,
        args.out_dir,
        args.target_count,
        args.encoder,
        args.dynamics_weight,
        args.symlink,
        args.profile,
    )
    print(f"Selected {stats['selected']}/{stats['total']} episodes")


if __name__ == "__main__":
    main()
