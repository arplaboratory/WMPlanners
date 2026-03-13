"""Stage 2 -- DINOv2 CLS tokens + body-frame odom deltas -> transition descriptors.

Supports two modes:
  - dino-wm: compact 390-dim transition descriptors (CLS + odom delta)
  - td-jepa: rich 416-dim per-frame embeddings (CLS + depth + proprio) + 10-dim actions
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import torch

from .odom_delta import compute_odom_deltas_for_frames
from .utils import (
    ACTION_DIM,
    DEPTH_DIM,
    DEPTH_FAR_THRESH,
    DEPTH_HIST_BINS,
    DEPTH_HIST_RANGE,
    DEPTH_NEAR_THRESH,
    DESCRIPTOR_DIM,
    FRAME_EMBEDDING_DIM,
    ODOM_COLS,
    PCMD_ACTION_COLS,
    PROPRIO_DIM,
    SO3_ACTION_COLS,
    SO3_PROPRIO_COLS,
    VISUAL_DIM,
    build_dino_transform,
    cleanup_model,
    interpolate_to_timestamps,
    load_csv,
    load_dino_model,
    load_image_batch,
    resolve_path,
    select_device,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 1000


# ---------------------------------------------------------------------------
# DINOv2 CLS token extraction
# ---------------------------------------------------------------------------

def _compute_cls_tokens(
    color_paths: list[str],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> np.ndarray:
    """Compute DINOv2 CLS-token embeddings (384-dim) for a list of images.

    Supports checkpointing for resumption.
    """
    n = len(color_paths)
    cls_tokens = np.zeros((n, VISUAL_DIM), dtype=np.float32)
    start_idx = 0

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        done = int(ckpt["done"])
        cls_tokens[:done] = ckpt["embeddings"][:done]
        start_idx = done
        print(f"[stage2] Resuming CLS tokens from frame {start_idx}/{n}")

    for batch_start in range(start_idx, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_tensor = load_image_batch(
            color_paths[batch_start:batch_end], transform,
        ).to(device)
        with torch.no_grad():
            feats = model(batch_tensor)
        cls_tokens[batch_start:batch_end] = feats.cpu().numpy()

        if checkpoint_path and (batch_end % CHECKPOINT_INTERVAL < BATCH_SIZE or batch_end == n):
            np.savez_compressed(checkpoint_path, embeddings=cls_tokens, done=np.array(batch_end))

        if (batch_end - start_idx) % (BATCH_SIZE * 10) == 0 or batch_end == n:
            print(f"[stage2]   CLS tokens: {batch_end}/{n}")

    return cls_tokens


# ---------------------------------------------------------------------------
# Depth features (12 dims) -- TD-JEPA
# ---------------------------------------------------------------------------

def _compute_depth_features(depth_paths: list[str]) -> np.ndarray:
    """Extract 12-dim depth features per frame.

    Features: mean_depth, std_depth, pct_near(<1m), pct_far(>5m), 8-bin histogram (0-10m).
    """
    n = len(depth_paths)
    features = np.zeros((n, DEPTH_DIM), dtype=np.float32)

    for i, path in enumerate(depth_paths):
        if not path or not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        depth_m = img.astype(np.float32) / 1000.0
        valid = depth_m > 0
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        valid_depths = depth_m[valid]
        features[i, 0] = valid_depths.mean()
        features[i, 1] = valid_depths.std()
        features[i, 2] = (valid_depths < DEPTH_NEAR_THRESH).sum() / n_valid
        features[i, 3] = (valid_depths > DEPTH_FAR_THRESH).sum() / n_valid

        hist, _ = np.histogram(valid_depths, bins=DEPTH_HIST_BINS, range=DEPTH_HIST_RANGE)
        features[i, 4:12] = hist.astype(np.float32) / n_valid

    return features


# ---------------------------------------------------------------------------
# Proprioceptive state (20 dims) -- TD-JEPA
# ---------------------------------------------------------------------------

def _compute_proprio(
    frame_ts: np.ndarray,
    odom_df: pd.DataFrame,
    so3_df: pd.DataFrame | None,
) -> np.ndarray:
    """Build 20-dim proprioceptive state vector per frame.

    Odometry (13): pos(3) + orient_quat(4) + lin_vel(3) + ang_vel(3)
    SO3 cmd (7): force(3) + cmd_orient_quat(4)  -- zeros if so3_df unavailable
    """
    odom_vals = interpolate_to_timestamps(
        odom_df, frame_ts, ODOM_COLS,
    )[ODOM_COLS].values.astype(np.float32)  # (N, 13)

    if so3_df is not None and len(so3_df) > 0:
        so3_vals = interpolate_to_timestamps(
            so3_df, frame_ts, SO3_PROPRIO_COLS,
        )[SO3_PROPRIO_COLS].values.astype(np.float32)  # (N, 7)
    else:
        so3_vals = np.zeros((len(frame_ts), 7), dtype=np.float32)

    return np.concatenate([odom_vals, so3_vals], axis=1)  # (N, 20)


# ---------------------------------------------------------------------------
# Action representation (10 dims) -- TD-JEPA
# ---------------------------------------------------------------------------

def _compute_actions(
    frame_ts: np.ndarray,
    pcmd_df: pd.DataFrame | None,
    so3_df: pd.DataFrame | None,
) -> np.ndarray:
    """Build 10-dim action vector per frame.

    Position cmd (8): velocity(3) + acceleration(3) + yaw + yaw_dot
    SO3 cmd derived (2): force_magnitude + force_azimuth
    """
    n = len(frame_ts)

    if pcmd_df is not None and len(pcmd_df) > 0:
        pcmd_vals = interpolate_to_timestamps(
            pcmd_df, frame_ts, PCMD_ACTION_COLS,
        )[PCMD_ACTION_COLS].values.astype(np.float32)  # (N, 8)
    else:
        pcmd_vals = np.zeros((n, 8), dtype=np.float32)

    if so3_df is not None and len(so3_df) > 0:
        so3_interp = interpolate_to_timestamps(so3_df, frame_ts, SO3_ACTION_COLS)
        fx = so3_interp["force_x"].values.astype(np.float32)
        fy = so3_interp["force_y"].values.astype(np.float32)
        fz = so3_interp["force_z"].values.astype(np.float32)
        force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
        force_azimuth = np.arctan2(fy, fx)
        so3_derived = np.stack([force_mag, force_azimuth], axis=1)  # (N, 2)
    else:
        so3_derived = np.zeros((n, 2), dtype=np.float32)

    return np.concatenate([pcmd_vals, so3_derived], axis=1)  # (N, 10)


# ---------------------------------------------------------------------------
# Transition descriptor construction
# ---------------------------------------------------------------------------

def build_transition_descriptors(
    cls_tokens: np.ndarray,
    odom_deltas: np.ndarray,
    valid_transition: np.ndarray,
) -> np.ndarray:
    """Build 390-dim transition descriptors from CLS tokens and odom deltas.

    Parameters
    ----------
    cls_tokens : (N, 384) float32
    odom_deltas : (N-1, 6) float64
    valid_transition : (N-1,) bool -- True where transition is valid

    Returns
    -------
    (N-1, 390) float32 transition descriptors (zeroed for invalid transitions)
    """
    n = len(cls_tokens)
    assert len(odom_deltas) == n - 1
    assert len(valid_transition) == n - 1

    descriptors = np.zeros((n - 1, DESCRIPTOR_DIM), dtype=np.float32)

    valid_indices = np.where(valid_transition)[0]
    descriptors[valid_indices, :VISUAL_DIM] = cls_tokens[valid_indices]
    descriptors[valid_indices, VISUAL_DIM:] = odom_deltas[valid_indices].astype(np.float32)

    return descriptors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage2(
    dataset_dir: str,
    metrics_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    output_dir: str,
    base_dir: str | None = None,
    device: str | None = None,
) -> dict | None:
    """Run Stage 2: compute CLS tokens + odom deltas for transition descriptors.

    Parameters
    ----------
    dataset_dir : str
        Directory containing odometry.csv.
    metrics_df : pd.DataFrame
        Per-frame metrics with frame_id, timestamp_ns, color_path, depth_path, valid_transition.
    segments_df : pd.DataFrame
        Segment info with start_idx, end_idx, keep columns.
    output_dir : str
        Where to write the output .npz file.
    base_dir : str | None
        Root for resolving relative image paths.
    device : str | None
        Torch device string.

    Returns
    -------
    dict with keys: cls_tokens, odom_deltas, descriptors, etc. Or None if no kept frames.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    dev = select_device(device)
    os.makedirs(output_dir, exist_ok=True)
    bag_name = os.path.basename(os.path.normpath(dataset_dir))

    # Identify kept frames
    kept_segments = segments_df[segments_df["keep"] == True]  # noqa: E712
    if kept_segments.empty:
        print("[stage2] No kept segments -- skipping.")
        return None

    kept_mask = np.zeros(len(metrics_df), dtype=bool)
    for _, seg in kept_segments.iterrows():
        s, e = int(seg["start_idx"]), int(seg["end_idx"])
        kept_mask[s:e] = True

    kept_df = metrics_df[kept_mask].reset_index(drop=True)
    n_frames = len(kept_df)
    print(f"[stage2] {n_frames} kept frames from {len(kept_segments)} segments.")

    if n_frames == 0:
        return None

    frame_ids = kept_df["frame_id"].values
    timestamps = kept_df["timestamp_ns"].values
    color_paths = [resolve_path(p, base_dir) for p in kept_df["color_path"].values]
    depth_paths = [resolve_path(p, base_dir) for p in kept_df["depth_path"].values]

    # Valid transition mask for kept frames
    if "valid_transition" in kept_df.columns:
        valid_transition = kept_df["valid_transition"].values[:-1].astype(bool)
    else:
        valid_transition = np.ones(n_frames - 1, dtype=bool)

    # CLS tokens
    checkpoint_path = os.path.join(output_dir, f"{bag_name}_checkpoint.npz")
    model = load_dino_model(dev)
    transform = build_dino_transform()
    cls_tokens = _compute_cls_tokens(color_paths, model, transform, dev, checkpoint_path)
    cleanup_model(model, dev)

    # Body-frame odom deltas
    print("[stage2] Computing body-frame odom deltas ...")
    odom_df = load_csv(os.path.join(dataset_dir, "odometry.csv"))
    odom_deltas = compute_odom_deltas_for_frames(odom_df, timestamps)

    # Zero out cross-segment deltas (invalid transitions produce garbage values)
    odom_deltas[~valid_transition] = 0.0

    # Build transition descriptors (DINO-WM: 390-dim)
    print("[stage2] Building transition descriptors ...")
    descriptors = build_transition_descriptors(cls_tokens, odom_deltas, valid_transition)

    # TD-JEPA: rich per-frame features (depth + proprio + actions)
    print("[stage2] Computing depth features ...")
    depth_feats = _compute_depth_features(depth_paths)

    # Load telemetry CSVs for proprioceptive/action features
    so3_path = os.path.join(dataset_dir, "so3_cmd.csv")
    so3_df = load_csv(so3_path) if os.path.isfile(so3_path) else None
    pcmd_path = os.path.join(dataset_dir, "position_cmd.csv")
    pcmd_df = load_csv(pcmd_path) if os.path.isfile(pcmd_path) else None

    print("[stage2] Computing proprioceptive state ...")
    proprio = _compute_proprio(timestamps, odom_df, so3_df)

    print("[stage2] Computing actions ...")
    actions = _compute_actions(timestamps, pcmd_df, so3_df)

    # TD-JEPA combined frame embedding: CLS (384) + depth (12) + proprio (20) = 416
    frame_embeddings = np.concatenate([cls_tokens, depth_feats, proprio], axis=1)

    # Save output — includes both DINO-WM transition descriptors and TD-JEPA embeddings
    output_path = os.path.join(output_dir, f"{bag_name}_transitions_compact.npz")
    np.savez_compressed(
        output_path,
        # DINO-WM fields
        descriptors=descriptors,                # (N-1, 390) float32
        odom_deltas=odom_deltas,                # (N-1, 6) float64
        cls_tokens=cls_tokens,                  # (N, 384) float32
        frame_t_ids=frame_ids[:-1],             # (N-1,) int
        frame_t1_ids=frame_ids[1:],             # (N-1,) int
        timestamps_t=timestamps[:-1],           # (N-1,) int64
        timestamps_t1=timestamps[1:],           # (N-1,) int64
        valid_mask=valid_transition,            # (N-1,) bool
        bag_name=np.array(bag_name),            # scalar string
        color_paths=np.array(color_paths),      # (N,)
        depth_paths=np.array(depth_paths),      # (N,)
        # TD-JEPA fields
        frame_embeddings=frame_embeddings,      # (N, 416) float32
        actions=actions,                        # (N, 10) float32
        proprio=proprio,                        # (N, 20) float32
        depth_features=depth_feats,             # (N, 12) float32
        frame_ids=frame_ids,                    # (N,) int
        timestamps=timestamps,                  # (N,) int64
    )

    print(f"[stage2] Saved {output_path}")
    print(f"[stage2]   CLS tokens:        {cls_tokens.shape}")
    print(f"[stage2]   Odom deltas:        {odom_deltas.shape}")
    print(f"[stage2]   Descriptors:        {descriptors.shape}")
    print(f"[stage2]   Frame embeddings:   {frame_embeddings.shape}")
    print(f"[stage2]   Actions:            {actions.shape}")
    print(f"[stage2]   Proprio:            {proprio.shape}")

    # Clean up checkpoint
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass

    return {
        "cls_tokens": cls_tokens,
        "odom_deltas": odom_deltas,
        "descriptors": descriptors,
        "frame_embeddings": frame_embeddings,
        "actions": actions,
        "proprio": proprio,
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "valid_mask": valid_transition,
    }
