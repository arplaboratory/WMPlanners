"""Dynamics-weighted Farthest Point Sampling episode selector.

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


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------

def compute_episode_odom_delta(odom: np.ndarray) -> np.ndarray:
    """Compute start-to-end body-frame displacement from odom array.

    odom: (T, D_odom) where first 7 cols are pos_x,y,z, orient_x,y,z,w.
    Returns: (6,) array [dx, dy, dz, ax, ay, az] in body frame (axis-angle rotation).
    """
    pos_start, pos_end = odom[0, :3], odom[-1, :3]
    quat_start = odom[0, 3:7]  # (x, y, z, w) — scipy format
    quat_end = odom[-1, 3:7]
    r_start = Rotation.from_quat(quat_start)
    r_end = Rotation.from_quat(quat_end)
    dp_world = pos_end - pos_start
    dp_body = r_start.inv().apply(dp_world)
    r_delta = r_start.inv() * r_end
    axis_angle = r_delta.as_rotvec()
    return np.concatenate([dp_body, axis_angle])


def compute_descriptors(
    data_dir: str,
    encoder: str = "none",
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

    pattern = os.path.join(data_dir, "ep_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz files found in {data_dir!r}")

    descriptors = []
    filenames = []
    for fpath in files:
        data = np.load(fpath)
        action = data["action"]   # (T, D_act)
        odom = data["odom"]       # (T, D_odom)

        action_stats = np.concatenate([action.mean(axis=0), action.std(axis=0)])  # (2*D_act,)
        odom_delta = compute_episode_odom_delta(odom)                             # (6,)
        descriptor = np.concatenate([action_stats, odom_delta])

        descriptors.append(descriptor)
        filenames.append(os.path.basename(fpath))

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
) -> dict:
    """Select diverse episodes via dynamics-weighted FPS. Returns stats dict."""

    # 1. Compute descriptors
    raw_descs, filenames = compute_descriptors(data_dir, encoder)
    n = len(filenames)

    # 2. Split descriptor blocks
    #    For encoder="none": layout is [action_mean | action_std | odom_delta]
    #    odom_delta is always the last 6 columns
    odom_dim = 6
    action_stats_block = raw_descs[:, :-odom_dim]   # (N, 2*D_act)
    odom_block = raw_descs[:, -odom_dim:]            # (N, 6)

    # 3. Z-score normalise each block independently
    def _safe_scale(block: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        # Avoid zero-variance columns blowing up — sklearn handles this with
        # with_std=True but clips near-zero stds; use default behaviour.
        return scaler.fit_transform(block)

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
    )
    print(f"Selected {stats['selected']}/{stats['total']} episodes")


if __name__ == "__main__":
    main()
