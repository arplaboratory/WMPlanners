"""Stage 3 -- Window-level FPS diversity selection for DINO-WM training."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VISUAL_DIM = 384
ODOM_DELTA_DIM = 6
DEFAULT_WINDOW_SIZE = 16  # frames per window (H+1 where H=15)
DEFAULT_WINDOW_STRIDE = 16  # non-overlapping by default


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Z-score normalize CLS and odom_delta components independently."""
    cls_part = descriptors[:, :VISUAL_DIM].copy()
    odom_part = descriptors[:, VISUAL_DIM:].copy()

    cls_normed = np.nan_to_num(StandardScaler().fit_transform(cls_part), nan=0.0)
    odom_normed = np.nan_to_num(StandardScaler().fit_transform(odom_part), nan=0.0)

    return np.concatenate([cls_normed, odom_normed], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(
    descriptors: np.ndarray,
    target_count: int,
) -> list[int]:
    """Select diverse items via FPS on descriptors."""
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
# Window extraction
# ---------------------------------------------------------------------------

def extract_windows(
    cls_tokens: np.ndarray,
    odom_deltas: np.ndarray,
    valid_mask: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    bag_name: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, list[dict]]:
    """Extract sliding windows from contiguous valid transition runs.

    Parameters
    ----------
    cls_tokens : (N, 384) CLS tokens per frame
    odom_deltas : (N-1, 6) odom deltas per transition
    valid_mask : (N-1,) bool valid transitions
    frame_ids : (N,) frame IDs
    timestamps : (N,) timestamps
    bag_name : str
    window_size : int — number of frames per window
    window_stride : int — stride between windows

    Returns
    -------
    (window_descriptors, window_metadata)
    window_descriptors: (W, 390) float32
    window_metadata: list of dicts with bag_name, window fields
    """
    n_transitions = len(valid_mask)
    if n_transitions == 0:
        return np.zeros((0, VISUAL_DIM + ODOM_DELTA_DIM), dtype=np.float32), []

    # Find contiguous runs of valid transitions
    # A window of size W needs W-1 consecutive valid transitions
    min_run_length = window_size - 1

    runs = []
    run_start = None
    for i in range(n_transitions):
        if valid_mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_run_length:
                    runs.append((run_start, i))
                run_start = None
    # Handle final run
    if run_start is not None:
        run_len = n_transitions - run_start
        if run_len >= min_run_length:
            runs.append((run_start, n_transitions))

    descriptors_list = []
    metadata_list = []
    window_id = 0

    for run_start, run_end in runs:
        for t_start in range(run_start, run_end - min_run_length + 1, window_stride):
            f_start = t_start
            f_end = t_start + window_size  # exclusive

            win_cls = cls_tokens[f_start:f_end]  # (window_size, 384)
            win_odom = odom_deltas[t_start:t_start + window_size - 1]  # (window_size-1, 6)

            # Window descriptor: mean CLS + mean odom delta
            mean_cls = win_cls.mean(axis=0)  # (384,)
            mean_odom = win_odom.mean(axis=0)  # (6,)
            desc = np.concatenate([mean_cls, mean_odom]).astype(np.float32)

            descriptors_list.append(desc)
            metadata_list.append({
                "bag_name": bag_name,
                "window_id": window_id,
                "start_frame_id": int(frame_ids[f_start]),
                "end_frame_id": int(frame_ids[f_end - 1]),
                "start_timestamp": int(timestamps[f_start]),
                "end_timestamp": int(timestamps[f_end - 1]),
                "n_frames": window_size,
                "frame_ids": frame_ids[f_start:f_end].tolist(),
                "timestamps": timestamps[f_start:f_end].tolist(),
                "odom_deltas": win_odom.tolist(),
                "descriptor": desc.tolist(),
                "odom_dx_mean": float(mean_odom[0]),
                "odom_dy_mean": float(mean_odom[1]),
                "odom_dz_mean": float(mean_odom[2]),
                "odom_ax_mean": float(mean_odom[3]),
                "odom_ay_mean": float(mean_odom[4]),
                "odom_az_mean": float(mean_odom[5]),
            })
            window_id += 1

    if descriptors_list:
        descriptors = np.stack(descriptors_list)
    else:
        descriptors = np.zeros((0, VISUAL_DIM + ODOM_DELTA_DIM), dtype=np.float32)

    return descriptors, metadata_list


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------

def generate_gap_analysis(
    descriptors: np.ndarray,
    odom_means: np.ndarray,
    selected_indices: list[int],
    bag_names: np.ndarray,
    output_dir: str,
    n_clusters: int = 40,
) -> dict:
    """Generate gap analysis reports for window selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    os.makedirs(output_dir, exist_ok=True)

    n = len(descriptors)
    is_selected = np.zeros(n, dtype=bool)
    is_selected[selected_indices] = True

    k = min(n_clusters, max(2, n // 2))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(descriptors)

    # Cluster report
    cluster_records = []
    for c in range(k):
        mask_c = labels == c
        total = int(mask_c.sum())
        selected = int((mask_c & is_selected).sum())
        c_odom = odom_means[mask_c]

        if len(c_odom) > 0:
            mean_delta = np.abs(c_odom.mean(axis=0))
            dim_names = ["forward", "lateral", "vertical", "roll", "pitch", "yaw"]
            dominant_idx = int(np.argmax(mean_delta))
            sign = "+" if c_odom.mean(axis=0)[dominant_idx] >= 0 else "-"
            label = f"{dim_names[dominant_idx]} ({sign}{mean_delta[dominant_idx]:.3f})"
        else:
            label = "empty"

        selection_pct = round(selected / total * 100, 1) if total > 0 else 0.0
        cluster_records.append({
            "cluster": c, "label": label, "total": total,
            "selected": selected, "selection_pct": selection_pct,
        })

    cluster_df = pd.DataFrame(cluster_records)
    cluster_df.to_csv(os.path.join(output_dir, "cluster_report.csv"), index=False)

    # Action histograms
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    dim_names = ["dx", "dy", "dz", "ax", "ay", "az"]
    for i, (ax, name) in enumerate(zip(axes.flat, dim_names)):
        all_vals = odom_means[:, i]
        sel_vals = odom_means[is_selected, i]
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)
        ax.hist(all_vals, bins=bins, alpha=0.5, label="all", density=True)
        ax.hist(sel_vals, bins=bins, alpha=0.7, label="selected", density=True)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Action Space Histograms (corpus vs selected windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "action_histograms.png"), dpi=150)
    plt.close(fig)

    # Visual diversity PCA
    from sklearn.decomposition import PCA
    cls_part = descriptors[:, :VISUAL_DIM]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cls_part)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[~is_selected, 0], coords[~is_selected, 1],
               c="gray", alpha=0.3, s=5, label="not selected")
    ax.scatter(coords[is_selected, 0], coords[is_selected, 1],
               c="red", alpha=0.6, s=10, label="selected")
    ax.legend()
    ax.set_title("Visual Diversity (PCA of window mean CLS tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "visual_diversity.png"), dpi=150)
    plt.close(fig)

    # Per-bag contribution
    unique_bags = np.unique(bag_names)
    bag_records = []
    for bag in unique_bags:
        bag_mask = bag_names == bag
        total_w = int(bag_mask.sum())
        selected_w = int((bag_mask & is_selected).sum())
        clusters_covered = len(set(labels[bag_mask & is_selected]))
        selection_pct = round(selected_w / total_w * 100, 1) if total_w > 0 else 0.0
        bag_records.append({
            "bag_name": str(bag), "total_windows": total_w,
            "selected_windows": selected_w, "selection_pct": selection_pct,
            "clusters_covered": f"{clusters_covered}/{k}",
        })
    bag_df = pd.DataFrame(bag_records)
    bag_df.to_csv(os.path.join(output_dir, "bag_contribution.csv"), index=False)

    # Recommendations
    avg_pct = cluster_df["selection_pct"].mean()
    recommendations = []
    for _, row in cluster_df.iterrows():
        if row["selection_pct"] < avg_pct * 0.5:
            recommendations.append(
                f"Under-represented: cluster {row['cluster']} ({row['label']}), "
                f"{row['selected']} selected vs corpus avg {avg_pct:.0f}%"
            )
    with open(os.path.join(output_dir, "recommendations.txt"), "w") as f:
        f.write("\n".join(recommendations) if recommendations else "No gaps detected.")

    return {
        "cluster_report": cluster_records,
        "environment_report": bag_records,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage3(
    per_bag_dir: str,
    output_dir: str,
    target_count: int = 5_000,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[pd.DataFrame, dict]:
    """Run Stage 3: window-level diversity selection across all bags.

    Parameters
    ----------
    per_bag_dir : str
        Directory containing per-bag subdirectories with *_transitions_compact.npz.
    output_dir : str
        Where to write selected_windows.csv, corpus_stats.json, gap_analysis/.
    target_count : int
        Target number of windows to select.
    window_size : int
        Number of frames per trajectory window.
    window_stride : int
        Stride between consecutive windows.

    Returns
    -------
    (selected_df, corpus_stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Discover bags
    npz_files = sorted(glob.glob(os.path.join(per_bag_dir, "*", "*_transitions_compact.npz")))
    if not npz_files:
        npz_files = sorted(glob.glob(os.path.join(per_bag_dir, "*_transitions_compact.npz")))
    if not npz_files:
        print("[stage3] No transition files found.")
        return pd.DataFrame(), {}

    print(f"[stage3] Found {len(npz_files)} bag(s)")

    # Extract windows from all bags
    all_descriptors = []
    all_metadata = []

    for npz_path in npz_files:
        data = dict(np.load(npz_path, allow_pickle=True))
        bag_name = str(data["bag_name"])

        cls_tokens = data["cls_tokens"]
        odom_deltas = data["odom_deltas"]
        valid_mask = data["valid_mask"].astype(bool)

        # Reconstruct full frame IDs and timestamps arrays (N frames)
        frame_t_ids = data["frame_t_ids"]
        frame_t1_ids = data["frame_t1_ids"]
        timestamps_t = data["timestamps_t"]
        timestamps_t1 = data["timestamps_t1"]

        # Reconstruct N-length arrays
        n_frames = len(cls_tokens)
        frame_ids = np.zeros(n_frames, dtype=np.int64)
        timestamps = np.zeros(n_frames, dtype=np.int64)
        frame_ids[:-1] = frame_t_ids
        frame_ids[-1] = frame_t1_ids[-1] if len(frame_t1_ids) > 0 else 0
        timestamps[:-1] = timestamps_t
        timestamps[-1] = timestamps_t1[-1] if len(timestamps_t1) > 0 else 0

        win_desc, win_meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, bag_name,
            window_size=window_size,
            window_stride=window_stride,
        )

        if len(win_desc) > 0:
            all_descriptors.append(win_desc)
            all_metadata.extend(win_meta)

    if not all_descriptors:
        print("[stage3] No windows extracted.")
        return pd.DataFrame(), {}

    descriptors = np.concatenate(all_descriptors)
    n_windows = len(descriptors)
    bag_names_arr = np.array([m["bag_name"] for m in all_metadata])
    odom_means = descriptors[:, VISUAL_DIM:]  # last 6 dims are mean odom

    print(f"[stage3] {n_windows} windows extracted (size={window_size}, stride={window_stride})")

    # Normalize and FPS
    normed = normalize_descriptors(descriptors)
    print(f"[stage3] Running FPS (target={target_count}) ...")
    selected_indices = farthest_point_sampling(normed, target_count)
    print(f"[stage3] Selected {len(selected_indices)} windows")

    # Build selected_windows.csv (drop frame_ids list for CSV compatibility)
    selected_records = []
    for idx in selected_indices:
        meta = all_metadata[idx]
        record = {k: v for k, v in meta.items() if k != "frame_ids"}
        selected_records.append(record)

    selected_df = pd.DataFrame(selected_records)
    selected_df.to_csv(os.path.join(output_dir, "selected_windows.csv"), index=False)

    # Also save full metadata with frame_ids as JSON for HF upload
    selected_full = [all_metadata[idx] for idx in selected_indices]
    with open(os.path.join(output_dir, "selected_windows_full.json"), "w") as f:
        json.dump(selected_full, f, indent=2)

    # Corpus stats
    stats = {
        "total_windows": n_windows,
        "selected_windows": len(selected_indices),
        "target_count": target_count,
        "window_size": window_size,
        "window_stride": window_stride,
        "n_bags": len(npz_files),
        "total_selected_frames": len(selected_indices) * window_size,
    }

    # Gap analysis
    gap_dir = os.path.join(output_dir, "gap_analysis")
    print("[stage3] Generating gap analysis ...")
    gap_results = generate_gap_analysis(
        normed, odom_means, selected_indices,
        bag_names_arr, gap_dir,
    )
    stats["gap_analysis"] = gap_results

    with open(os.path.join(output_dir, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("[stage3] Stage 3 complete.")
    return selected_df, stats
