"""Tests for Stage 3b patch token export."""

import os

import numpy as np
import pandas as pd
import pytest

from scripts.world_model.data_pipeline.stage3b_export import run_stage3b


class TestRunStage3b:
    """Tests for patch token export."""

    def test_output_shards_created(self, tmp_dir):
        """Should create shard files in selected_patches/."""
        selected_csv, bags_root = _setup_stage3b_inputs(tmp_dir, n_selected=5)
        output_dir = os.path.join(tmp_dir, "output")
        run_stage3b(
            selected_csv_path=selected_csv,
            bags_root=bags_root,
            output_dir=output_dir,
            device="cpu",
        )
        patches_dir = os.path.join(output_dir, "selected_patches")
        assert os.path.isdir(patches_dir)
        shards = [f for f in os.listdir(patches_dir) if f.endswith(".npz")]
        assert len(shards) > 0

    def test_shard_contents(self, tmp_dir):
        """Each shard should have patches_t, patches_t1, odom_deltas, metadata."""
        selected_csv, bags_root = _setup_stage3b_inputs(tmp_dir, n_selected=3)
        output_dir = os.path.join(tmp_dir, "output")
        run_stage3b(
            selected_csv_path=selected_csv,
            bags_root=bags_root,
            output_dir=output_dir,
            device="cpu",
        )
        patches_dir = os.path.join(output_dir, "selected_patches")
        shard_files = sorted([f for f in os.listdir(patches_dir) if f.endswith(".npz")])
        data = np.load(os.path.join(patches_dir, shard_files[0]), allow_pickle=True)

        assert "patches_t" in data
        assert "patches_t1" in data
        assert "odom_deltas" in data
        assert data["patches_t"].dtype == np.float16
        assert data["patches_t1"].dtype == np.float16
        # DINOv2 ViT-S/14 at 224px: 256 patches x 384 dims
        assert data["patches_t"].shape[1] == 256
        assert data["patches_t"].shape[2] == 384


def _setup_stage3b_inputs(base_dir, n_selected=5):
    """Create inputs for Stage 3b testing."""
    import cv2

    bag_name = "test_bag"
    bags_root = base_dir
    bag_dir = os.path.join(bags_root, bag_name)
    os.makedirs(bag_dir, exist_ok=True)

    # Create image_index.csv and dummy images
    n_frames = n_selected + 1
    ts = np.arange(n_frames, dtype=np.int64) * 33_000_000 + 1_000_000_000
    frame_ids = np.arange(n_frames)

    shard_dir = os.path.join(bag_dir, "shard_0000")
    os.makedirs(shard_dir, exist_ok=True)

    color_paths = []
    for i in range(n_frames):
        color_p = f"shard_0000/color_{i:06d}.jpg"
        img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(bag_dir, color_p), img)
        color_paths.append(color_p)

    pd.DataFrame({
        "frame_id": frame_ids,
        "timestamp_ns": ts,
        "color_path": color_paths,
    }).to_csv(os.path.join(bag_dir, "image_index.csv"), index=False)

    # Create selected_transitions.csv
    records = []
    for i in range(n_selected):
        records.append({
            "bag_name": bag_name,
            "frame_t_id": i,
            "frame_t1_id": i + 1,
            "timestamp_t": int(ts[i]),
            "timestamp_t1": int(ts[i + 1]),
            "odom_dx": 0.1, "odom_dy": 0.0, "odom_dz": 0.0,
            "odom_ax": 0.0, "odom_ay": 0.0, "odom_az": 0.0,
        })
    selected_csv = os.path.join(base_dir, "selected_transitions.csv")
    pd.DataFrame(records).to_csv(selected_csv, index=False)

    return selected_csv, bags_root
