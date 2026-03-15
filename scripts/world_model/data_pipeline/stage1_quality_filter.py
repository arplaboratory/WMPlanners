"""Stage 1 -- per-frame quality metrics and segment-level filtering."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .utils import interpolate_to_timestamps, load_csv, resolve_path

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 379          # Laplacian variance below this = blurry
SPEED_IDLE_THRESHOLD = 0.1    # m/s
DEPTH_VALID_PIXEL_THRESH = 0  # pixels strictly > this count as valid
DEPTH_VALID_FRAC_THRESH = 0.5 # frame is "depth-invalid" when valid fraction < this
SEGMENT_PERSIST_FRAMES = 10   # condition must persist this many frames to split
MIN_SEGMENT_DURATION_S = 1.0  # drop segments shorter than this


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def _laplacian_variance(image_path: str) -> float:
    """Return Laplacian variance (blur metric) for a colour image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def _depth_valid_fraction(depth_path: str) -> float:
    """Fraction of non-zero pixels in a 16-bit depth image."""
    if not depth_path:
        return np.nan
    img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.nan
    return float(np.count_nonzero(img) / img.size)


def compute_frame_metrics(dataset_dir: str, base_dir: str) -> pd.DataFrame:
    """Compute per-frame quality metrics for every frame in *dataset_dir*.

    Parameters
    ----------
    dataset_dir : str
        Directory containing image_index.csv, odometry.csv, position_cmd.csv.
    base_dir : str
        Root directory used to resolve relative image paths in image_index.csv.

    Returns
    -------
    pd.DataFrame  with columns:
        frame_id, timestamp_ns, color_path, depth_path,
        speed, angular_speed, blur, visual_diff, depth_valid_frac, action_mag
    """
    imgs = load_csv(os.path.join(dataset_dir, "image_index.csv"))
    odom = load_csv(os.path.join(dataset_dir, "odometry.csv"))
    pcmd = load_csv(os.path.join(dataset_dir, "position_cmd.csv"))

    frame_ts = imgs["timestamp_ns"].values

    # --- speed & angular speed from odometry ---
    odom["speed"] = np.sqrt(
        odom["linear_vel_x"] ** 2
        + odom["linear_vel_y"] ** 2
        + odom["linear_vel_z"] ** 2
    )
    odom["angular_speed"] = np.sqrt(
        odom["angular_vel_x"] ** 2
        + odom["angular_vel_y"] ** 2
        + odom["angular_vel_z"] ** 2
    )
    interp_odom = interpolate_to_timestamps(
        odom, frame_ts, ["speed", "angular_speed"]
    )
    imgs["speed"] = interp_odom["speed"].values
    imgs["angular_speed"] = interp_odom["angular_speed"].values

    # --- action magnitude from position_cmd velocity ---
    pcmd["action_mag"] = np.sqrt(
        pcmd["velocity_x"] ** 2
        + pcmd["velocity_y"] ** 2
        + pcmd["velocity_z"] ** 2
    )
    interp_pcmd = interpolate_to_timestamps(pcmd, frame_ts, ["action_mag"])
    imgs["action_mag"] = interp_pcmd["action_mag"].values

    # --- blur + visual diff in a single pass (each color image loaded once) ---
    blur_vals = []
    diffs = [0.0]  # first frame has no predecessor
    prev_gray = None
    for p in imgs["color_path"]:
        abs_p = resolve_path(p, base_dir)
        gray = cv2.imread(abs_p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            blur_vals.append(0.0)
            diffs.append(0.0)
            prev_gray = None
            continue
        blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))))
        else:
            if len(diffs) < len(imgs):
                diffs.append(0.0)
        prev_gray = gray
    imgs["blur"] = blur_vals
    imgs["visual_diff"] = diffs[: len(imgs)]

    # --- depth validity ---
    depth_vals = []
    for p in imgs["depth_path"]:
        abs_p = resolve_path(p, base_dir)
        depth_vals.append(_depth_valid_fraction(abs_p))
    imgs["depth_valid_frac"] = depth_vals

    return imgs


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_frames(metrics: pd.DataFrame) -> pd.DataFrame:
    """Split the frame sequence into segments at detected boundaries.

    A boundary is placed when one of the following conditions persists for
    >= SEGMENT_PERSIST_FRAMES consecutive frames:
      - idle: speed < SPEED_IDLE_THRESHOLD
      - blurry: blur < BLUR_THRESHOLD
      - gap: timestamp delta > 2x median frame interval
    """
    n = len(metrics)
    if n == 0:
        return pd.DataFrame(columns=["segment_id", "start_idx", "end_idx"])

    ts = metrics["timestamp_ns"].values
    dt = np.diff(ts)
    median_dt = np.median(dt) if len(dt) > 0 else 1

    # Boolean flags per frame
    is_idle = (metrics["speed"].values < SPEED_IDLE_THRESHOLD)
    is_blur = (metrics["blur"].values < BLUR_THRESHOLD)
    is_gap = np.concatenate([[False], dt > 2 * median_dt])

    # A boundary flag is set on frames where a condition has persisted for
    # SEGMENT_PERSIST_FRAMES consecutive frames.  We mark the *first* frame
    # of each new persistent run as a boundary.
    boundary = np.zeros(n, dtype=bool)

    for flag_arr in [is_idle, is_blur]:
        run_len = 0
        for i in range(n):
            if flag_arr[i]:
                run_len += 1
                if run_len == SEGMENT_PERSIST_FRAMES:
                    # Mark the start of this run as a boundary
                    boundary[i - SEGMENT_PERSIST_FRAMES + 1] = True
            else:
                run_len = 0

    # Timestamp gaps: immediate split (no persistence required)
    boundary |= is_gap

    # Build segments from boundary positions
    boundary[0] = True  # first frame always starts a segment
    split_indices = np.where(boundary)[0]

    segments = []
    for k, start in enumerate(split_indices):
        end = split_indices[k + 1] if k + 1 < len(split_indices) else n
        segments.append({"segment_id": k, "start_idx": start, "end_idx": end})

    return pd.DataFrame(segments)


# ---------------------------------------------------------------------------
# Evaluate & label segments
# ---------------------------------------------------------------------------

def evaluate_segments(
    metrics: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """Attach summary stats to each segment and decide keep/drop."""
    records = []
    for _, seg in segments.iterrows():
        s, e = int(seg["start_idx"]), int(seg["end_idx"])
        chunk = metrics.iloc[s:e]

        n_frames = len(chunk)
        ts = chunk["timestamp_ns"].values
        duration_s = (ts[-1] - ts[0]) / 1e9 if n_frames > 1 else 0.0
        mean_speed = chunk["speed"].mean()
        frac_blurry = (chunk["blur"] < BLUR_THRESHOLD).mean()

        depth_fracs = chunk["depth_valid_frac"].dropna()
        if len(depth_fracs) > 0:
            frac_depth_invalid = (depth_fracs < DEPTH_VALID_FRAC_THRESH).mean()
        else:
            frac_depth_invalid = 0.0

        # Decision
        keep = True
        reason = "ok"
        if mean_speed < SPEED_IDLE_THRESHOLD:
            keep, reason = False, "idle"
        elif frac_blurry > 0.5:
            keep, reason = False, "blurry"
        elif duration_s < MIN_SEGMENT_DURATION_S:
            keep, reason = False, "too_short"
        elif frac_depth_invalid > 0.3:
            keep, reason = False, "depth_invalid"

        records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "start_idx": s,
                "end_idx": e,
                "n_frames": n_frames,
                "duration_s": round(duration_s, 3),
                "mean_speed": round(mean_speed, 4),
                "frac_blurry": round(frac_blurry, 4),
                "frac_depth_invalid": round(frac_depth_invalid, 4),
                "keep": keep,
                "reason": reason,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage1(
    dataset_dir: str,
    output_dir: str,
    base_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage 1 quality filter end-to-end.

    Parameters
    ----------
    dataset_dir : str
        Path to the extracted dataset (contains image_index.csv, etc.).
    output_dir : str
        Where to write per-bag results.
    base_dir : str | None
        Root directory for resolving relative image paths.

    Returns
    -------
    (metrics, segments)  DataFrames.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    # Derive a bag name from the dataset directory for output filenames
    bag_name = os.path.basename(os.path.normpath(dataset_dir))

    print(f"[stage1] Computing per-frame metrics for {dataset_dir} ...")
    metrics = compute_frame_metrics(dataset_dir, base_dir)
    print(f"[stage1] {len(metrics)} frames processed.")

    print("[stage1] Segmenting ...")
    segments = segment_frames(metrics)

    print("[stage1] Evaluating segments ...")
    segments = evaluate_segments(metrics, segments)

    # --- Compute valid_transition flag ---
    print("[stage1] Computing valid_transition flags ...")
    ts = metrics["timestamp_ns"].values
    dt = np.diff(ts)
    median_dt = np.median(dt) if len(dt) > 0 else 1

    valid_transition = np.ones(len(metrics), dtype=bool)

    # Last frame overall cannot be a transition source
    valid_transition[-1] = False

    # Frame before a gap (dt > 2x median) is invalid
    gap_mask = dt > 2 * median_dt
    valid_transition[:-1][gap_mask] = False

    # Last frame of each kept segment is invalid
    for _, seg in segments.iterrows():
        end_idx = int(seg["end_idx"]) - 1  # last frame in segment
        if end_idx >= 0 and end_idx < len(valid_transition):
            valid_transition[end_idx] = False

    metrics["valid_transition"] = valid_transition

    # Save outputs
    metrics.to_csv(os.path.join(output_dir, f"{bag_name}_frame_scores.csv"), index=False)
    segments.to_csv(os.path.join(output_dir, f"{bag_name}_segments.csv"), index=False)

    kept = segments["keep"].sum()
    print(f"[stage1] {len(segments)} segments, {kept} kept, {len(segments) - kept} dropped.")

    return metrics, segments
