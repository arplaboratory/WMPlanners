"""Stage 3 -- FPS diversity selection for DINO-WM and TD-JEPA training.

Supports two modes:
  - dino-wm: window-level selection using 390-dim transition descriptors
  - td-jepa: segment-level selection using 832-dim (mean+std of 416-dim frame embeddings)
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import (
    FRAME_EMBEDDING_DIM,
    ODOM_DELTA_DIM,
    VISUAL_DIM,
    extract_track_id,
    interpolate_to_timestamps,
    load_csv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE = 16  # frames per window (H+1 where H=15)
DEFAULT_WINDOW_STRIDE = 16  # non-overlapping by default


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Z-score normalize CLS and odom_delta components independently."""
    cls_part = descriptors[:, :VISUAL_DIM]
    odom_part = descriptors[:, VISUAL_DIM:]

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


# ═══════════════════════════════════════════════════════════════════════════
# TD-JEPA: Segment-level selection
# ═══════════════════════════════════════════════════════════════════════════


def build_segment_summaries(
    segments_df: pd.DataFrame,
    embeddings_data: dict,
    dataset_dir: str,
    bag_name: str,
) -> pd.DataFrame:
    """Compute 832-dim summary embedding and metadata for each kept segment.

    Summary = concat(mean, std) of 416-dim per-frame embeddings.
    """
    kept_segments = segments_df[segments_df["keep"] == True].reset_index(drop=True)  # noqa: E712
    if kept_segments.empty:
        return pd.DataFrame()

    all_embeddings = embeddings_data["frame_embeddings"]  # (N_kept, 416)
    all_timestamps = embeddings_data["timestamps"]
    all_frame_ids = embeddings_data["frame_ids"]

    # Load odometry for altitude and velocity info
    odom_path = os.path.join(dataset_dir, "odometry.csv")
    odom_df = load_csv(odom_path) if os.path.isfile(odom_path) else None

    track_id = extract_track_id(bag_name)

    records = []
    emb_offset = 0

    for _, seg in kept_segments.iterrows():
        n_frames = int(seg["n_frames"])
        seg_embeddings = all_embeddings[emb_offset: emb_offset + n_frames]
        seg_timestamps = all_timestamps[emb_offset: emb_offset + n_frames]
        seg_frame_ids = all_frame_ids[emb_offset: emb_offset + n_frames]

        # Summary embedding: concat of mean and std -> 832 dims
        emb_mean = seg_embeddings.mean(axis=0)
        emb_std = seg_embeddings.std(axis=0)
        summary_embedding = np.concatenate([emb_mean, emb_std])

        duration_s = float(seg.get("duration_s", 0.0))

        # Classify flying vs driving and compute speed stats
        mean_speed = 0.0
        mean_angular_speed = 0.0
        is_flying = False

        if odom_df is not None and len(odom_df) > 0:
            odom_interp = interpolate_to_timestamps(
                odom_df, seg_timestamps,
                ["pos_z", "linear_vel_z", "linear_vel_x", "linear_vel_y",
                 "angular_vel_x", "angular_vel_y", "angular_vel_z"],
            )
            mean_altitude = odom_interp["pos_z"].mean()
            vel_z_std = odom_interp["linear_vel_z"].std()
            is_flying = bool(mean_altitude > 0.5 and vel_z_std > 0.1)

            speed = np.sqrt(
                odom_interp["linear_vel_x"].values ** 2
                + odom_interp["linear_vel_y"].values ** 2
                + odom_interp["linear_vel_z"].values ** 2
            )
            angular_speed = np.sqrt(
                odom_interp["angular_vel_x"].values ** 2
                + odom_interp["angular_vel_y"].values ** 2
                + odom_interp["angular_vel_z"].values ** 2
            )
            mean_speed = float(speed.mean())
            mean_angular_speed = float(angular_speed.mean())

        records.append({
            "bag_name": bag_name,
            "segment_id": int(seg["segment_id"]),
            "start_frame": int(seg_frame_ids[0]),
            "end_frame": int(seg_frame_ids[-1]),
            "n_frames": n_frames,
            "duration_s": duration_s,
            "mean_speed": round(mean_speed, 4),
            "mean_angular_speed": round(mean_angular_speed, 4),
            "track_id": track_id,
            "is_flying": is_flying,
            "summary_embedding": summary_embedding,
        })
        emb_offset += n_frames

    return pd.DataFrame(records)


def farthest_point_sampling_segments(
    segment_summaries: pd.DataFrame,
    target_frames: int,
) -> list[int]:
    """Select diverse segments via FPS on 832-dim summaries.

    Stops when cumulative selected frames >= target_frames.
    """
    n = len(segment_summaries)
    if n == 0:
        return []

    embeddings = np.stack(segment_summaries["summary_embedding"].values)
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    centroid = embeddings.mean(axis=0)
    dists_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    first_idx = int(np.argmax(dists_to_centroid))

    selected = [first_idx]
    cumulative_frames = int(segment_summaries.iloc[first_idx]["n_frames"])

    if cumulative_frames >= target_frames:
        return selected

    min_dists = np.linalg.norm(embeddings - embeddings[first_idx], axis=1)
    min_dists[first_idx] = -1.0

    while cumulative_frames < target_frames and len(selected) < n:
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] <= 0:
            break
        selected.append(next_idx)
        cumulative_frames += int(segment_summaries.iloc[next_idx]["n_frames"])
        new_dists = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[next_idx] = -1.0

    return selected


def generate_segment_gap_analysis(
    segment_summaries: pd.DataFrame,
    selected_indices: list[int],
    output_dir: str,
) -> None:
    """Generate gap-analysis plots for TD-JEPA segment selection.

    Includes UMAP projection, cluster analysis, speed distributions,
    and coverage heatmap by track and mode.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    os.makedirs(output_dir, exist_ok=True)

    n = len(segment_summaries)
    is_selected = np.zeros(n, dtype=bool)
    is_selected[selected_indices] = True

    embeddings = np.stack(segment_summaries["summary_embedding"].values)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 1. UMAP projection (optional)
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[~is_selected, 0], coords[~is_selected, 1],
                   c="gray", alpha=0.4, s=15, label="not selected")
        ax.scatter(coords[is_selected, 0], coords[is_selected, 1],
                   c="red", alpha=0.8, s=25, label="selected")
        ax.legend()
        ax.set_title("UMAP Projection of Segment Embeddings")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "umap_projection.png"), dpi=150)
        plt.close(fig)
        print("[stage3] UMAP projection saved.")
    except ImportError:
        print("[stage3] umap-learn not installed — skipping UMAP projection.")

    # 2. Cluster analysis (k-means)
    k = min(30, max(2, n // 2))
    if n < k:
        print(f"[stage3] Too few segments ({n}) for clustering — skipping cluster analysis.")
        return
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_scaled)

    cluster_records = []
    for c in range(k):
        mask_c = labels == c
        total_segs = int(mask_c.sum())
        selected_segs = int((mask_c & is_selected).sum())
        selection_pct = round(selected_segs / total_segs * 100, 1) if total_segs > 0 else 0.0

        c_df = segment_summaries[mask_c]
        total_frames = int(c_df["n_frames"].sum())
        selected_frames = int(segment_summaries[mask_c & is_selected]["n_frames"].sum())
        mean_speed = round(float(c_df["mean_speed"].mean()), 4)
        pct_flying = round(float(c_df["is_flying"].mean()) * 100, 1)
        tracks = ",".join(sorted(c_df["track_id"].unique()))

        cluster_records.append({
            "cluster": c,
            "total_segments": total_segs,
            "selected_segments": selected_segs,
            "selection_pct": selection_pct,
            "total_frames": total_frames,
            "selected_frames": selected_frames,
            "mean_speed": mean_speed,
            "pct_flying": pct_flying,
            "tracks": tracks,
        })

    cluster_df = pd.DataFrame(cluster_records)
    avg_pct = cluster_df["selection_pct"].mean()
    cluster_df["under_represented"] = cluster_df["selection_pct"] < avg_pct
    cluster_df.to_csv(os.path.join(output_dir, "cluster_report.csv"), index=False)
    print(f"[stage3] Cluster report saved (k={k}).")

    # 3. Distribution comparison — speed and angular speed
    for col, label in [("mean_speed", "Speed (m/s)"), ("mean_angular_speed", "Angular Speed (rad/s)")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = segment_summaries[col].values
        sel_vals = segment_summaries.iloc[selected_indices][col].values
        bins = np.linspace(all_vals.min(), all_vals.max(), 30)
        ax.hist(all_vals, bins=bins, alpha=0.5, label="all", density=True)
        ax.hist(sel_vals, bins=bins, alpha=0.7, label="selected", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {label}")
        ax.legend()
        fig.tight_layout()
        fname = col.replace("mean_", "") + "_distribution.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
    print("[stage3] Distribution plots saved.")

    # 4. Coverage heatmap by track and mode
    summaries = segment_summaries.copy()
    summaries["mode"] = summaries["is_flying"].map({True: "flying", False: "driving"})
    summaries["selected"] = is_selected

    pivot_total = summaries.pivot_table(
        index="track_id", columns="mode", values="n_frames", aggfunc="sum", fill_value=0,
    )
    pivot_selected = summaries[summaries["selected"]].pivot_table(
        index="track_id", columns="mode", values="n_frames", aggfunc="sum", fill_value=0,
    )
    pivot_selected = pivot_selected.reindex_like(pivot_total).fillna(0)
    coverage = pivot_selected / pivot_total.replace(0, np.nan)
    coverage = coverage.fillna(0)

    fig, ax = plt.subplots(figsize=(max(6, len(coverage.columns) * 2), max(4, len(coverage) * 0.6)))
    im = ax.imshow(coverage.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns)
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Track")
    ax.set_title("Frame Selection Coverage by Track and Mode")
    for i in range(len(coverage.index)):
        for j in range(len(coverage.columns)):
            val = coverage.values[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Selection fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "coverage_heatmap.png"), dpi=150)
    plt.close(fig)
    print("[stage3] Coverage heatmap saved.")


def run_stage3_tdjepa(
    per_bag_dir: str,
    output_dir: str,
    target_frames: int = 500_000,
) -> tuple[pd.DataFrame, dict]:
    """Run Stage 3 in TD-JEPA mode: segment-level diversity selection.

    Uses 832-dim segment summaries (mean + std of 416-dim per-frame embeddings)
    and farthest-point sampling to select diverse segments.

    Parameters
    ----------
    per_bag_dir : str
        Directory containing per-bag *_transitions_compact.npz and *_segments.csv.
    output_dir : str
        Where to write selected_segments.csv, selected_frames.csv, corpus_stats.json.
    target_frames : int
        Target number of frames to select.

    Returns
    -------
    (selected_segments_df, corpus_stats_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.dirname(os.path.normpath(per_bag_dir))

    # Discover bags
    npz_files = sorted(glob.glob(os.path.join(per_bag_dir, "*", "*_transitions_compact.npz")))
    if not npz_files:
        npz_files = sorted(glob.glob(os.path.join(per_bag_dir, "*_transitions_compact.npz")))
    if not npz_files:
        print("[stage3] No embedding files found.")
        return pd.DataFrame(), {}

    print(f"[stage3] TD-JEPA mode: found {len(npz_files)} bag(s)")

    # Build segment summaries for each bag
    all_summaries = []
    bag_embeddings_data = {}

    for npz_path in npz_files:
        bag_name = os.path.basename(npz_path).replace("_transitions_compact.npz", "")
        npz_dir = os.path.dirname(npz_path)
        seg_path = os.path.join(npz_dir, f"{bag_name}_segments.csv")

        if not os.path.isfile(seg_path):
            print(f"[stage3] WARNING: no segments CSV for {bag_name}, skipping.")
            continue

        segments_df = load_csv(seg_path)
        embeddings_data = dict(np.load(npz_path, allow_pickle=True))
        bag_embeddings_data[bag_name] = embeddings_data

        bag_dataset_dir = os.path.join(dataset_dir, bag_name)
        if not os.path.isdir(bag_dataset_dir):
            bag_dataset_dir = npz_dir

        print(f"[stage3] Building segment summaries for {bag_name} ...")
        summaries = build_segment_summaries(segments_df, embeddings_data, bag_dataset_dir, bag_name)
        if not summaries.empty:
            all_summaries.append(summaries)

    if not all_summaries:
        print("[stage3] No segment summaries produced.")
        return pd.DataFrame(), {}

    segment_summaries = pd.concat(all_summaries, ignore_index=True)
    print(f"[stage3] Total segments: {len(segment_summaries)}, "
          f"total frames: {segment_summaries['n_frames'].sum()}")

    # Farthest Point Sampling
    print(f"[stage3] Running FPS (target_frames={target_frames}) ...")
    selected_indices = farthest_point_sampling_segments(segment_summaries, target_frames)
    selected_df = segment_summaries.iloc[selected_indices].copy()
    total_selected_frames = int(selected_df["n_frames"].sum())
    print(f"[stage3] Selected {len(selected_indices)} segments, {total_selected_frames} frames.")

    # Save selected_segments.csv
    selected_out = selected_df.drop(columns=["summary_embedding"])
    selected_out.to_csv(os.path.join(output_dir, "selected_segments.csv"), index=False)

    # Build selected_frames.csv
    frame_records = []
    for _, row in selected_df.iterrows():
        bag_name = row["bag_name"]
        emb_data = bag_embeddings_data[bag_name]
        npz_dir_search = os.path.join(per_bag_dir, bag_name)

        seg_csv_path = os.path.join(npz_dir_search, f"{bag_name}_segments.csv")
        if not os.path.isfile(seg_csv_path):
            seg_csv_path = os.path.join(per_bag_dir, f"{bag_name}_segments.csv")
        seg_df = load_csv(seg_csv_path)
        kept_segs = seg_df[seg_df["keep"] == True].reset_index(drop=True)  # noqa: E712

        offset = 0
        target_seg_id = row["segment_id"]
        for _, ks in kept_segs.iterrows():
            if int(ks["segment_id"]) == target_seg_id:
                break
            offset += int(ks["n_frames"])

        n = int(row["n_frames"])
        fids = emb_data["frame_ids"][offset: offset + n]
        ts = emb_data["timestamps"][offset: offset + n]
        cpaths = emb_data["color_paths"][offset: offset + n]
        dpaths = emb_data["depth_paths"][offset: offset + n]
        proprio = emb_data["proprio"][offset: offset + n]
        actions = emb_data["actions"][offset: offset + n]

        for i in range(n):
            frame_records.append({
                "bag_name": bag_name,
                "frame_id": int(fids[i]),
                "timestamp_ns": int(ts[i]),
                "color_path": str(cpaths[i]),
                "depth_path": str(dpaths[i]),
                "proprio": ",".join(f"{v:.6f}" for v in proprio[i]),
                "action": ",".join(f"{v:.6f}" for v in actions[i]),
            })

    frames_df = pd.DataFrame(frame_records)
    frames_df.to_csv(os.path.join(output_dir, "selected_frames.csv"), index=False)
    print(f"[stage3] Saved selected_frames.csv ({len(frames_df)} frames)")

    # Corpus stats
    total_segments = len(segment_summaries)
    total_frames = int(segment_summaries["n_frames"].sum())

    by_mode = {}
    for mode_val, mode_label in [(True, "flying"), (False, "driving")]:
        mode_mask = segment_summaries["is_flying"] == mode_val
        sel_mode_mask = selected_df["is_flying"] == mode_val
        by_mode[mode_label] = {
            "total_segments": int(mode_mask.sum()),
            "selected_segments": int(sel_mode_mask.sum()),
            "total_frames": int(segment_summaries.loc[mode_mask, "n_frames"].sum()),
            "selected_frames": int(selected_df.loc[sel_mode_mask, "n_frames"].sum()),
        }

    by_track = {}
    for track in sorted(segment_summaries["track_id"].unique()):
        track_mask = segment_summaries["track_id"] == track
        sel_track_mask = selected_df["track_id"] == track
        by_track[track] = {
            "total_segments": int(track_mask.sum()),
            "selected_segments": int(sel_track_mask.sum()),
            "total_frames": int(segment_summaries.loc[track_mask, "n_frames"].sum()),
            "selected_frames": int(selected_df.loc[sel_track_mask, "n_frames"].sum()),
        }

    stats = {
        "mode": "td-jepa",
        "total_segments": total_segments,
        "selected_segments": len(selected_indices),
        "total_frames": total_frames,
        "selected_frames": total_selected_frames,
        "target_frames": target_frames,
        "by_mode": by_mode,
        "by_track": by_track,
    }

    with open(os.path.join(output_dir, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Gap analysis
    gap_dir = os.path.join(output_dir, "gap_analysis")
    print("[stage3] Generating gap analysis ...")
    generate_segment_gap_analysis(segment_summaries, selected_indices, gap_dir)

    print("[stage3] TD-JEPA Stage 3 complete.")
    return selected_out, stats


# ═══════════════════════════════════════════════════════════════════════════
# DINO-WM: Window-level selection (existing)
# ═══════════════════════════════════════════════════════════════════════════


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
