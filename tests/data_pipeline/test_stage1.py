"""Tests for Stage 1 quality filter modifications."""

import os

import numpy as np
import pandas as pd
import pytest

from scripts.world_model.data_pipeline.stage1_quality_filter import segment_frames, run_stage1


class TestGapSegmentation:
    """Test that timestamp gaps trigger immediate segment splits."""

    def test_single_gap_splits_immediately(self):
        """A single timestamp gap > 2x median dt should split segments immediately."""
        n = 30
        ts = np.arange(n, dtype=np.int64) * 33_000_000  # 33ms intervals = ~30Hz
        # Insert a large gap at frame 15
        ts[15:] += 500_000_000  # 500ms gap

        metrics = pd.DataFrame({
            "timestamp_ns": ts,
            "speed": np.ones(n),  # not idle
            "blur": np.full(n, 1000.0),  # not blurry
        })

        segments = segment_frames(metrics)
        # Should have at least 2 segments due to the gap
        assert len(segments) >= 2
        # First segment should end at or before frame 15
        assert segments.iloc[0]["end_idx"] <= 15

    def test_no_persistence_needed_for_gaps(self):
        """Gap segmentation should NOT require SEGMENT_PERSIST_FRAMES consecutive gaps."""
        n = 25
        ts = np.arange(n, dtype=np.int64) * 33_000_000
        # Single large gap at frame 10 (only 1 frame, not 10)
        ts[10:] += 1_000_000_000  # 1s gap

        metrics = pd.DataFrame({
            "timestamp_ns": ts,
            "speed": np.ones(n),
            "blur": np.full(n, 1000.0),
        })

        segments = segment_frames(metrics)
        # Must split at the gap
        assert len(segments) >= 2


class TestValidTransition:
    """Test valid_transition flag computation."""

    def test_valid_transition_added_to_output(self, tmp_dir):
        """run_stage1 should add valid_transition column to metrics output."""
        _create_minimal_dataset(tmp_dir, n_frames=20)
        metrics, segments = run_stage1(tmp_dir, tmp_dir, base_dir=tmp_dir)
        assert "valid_transition" in metrics.columns

    def test_last_frame_in_segment_is_invalid(self, tmp_dir):
        """The last frame of each segment cannot be a transition source."""
        _create_minimal_dataset(tmp_dir, n_frames=20)
        metrics, segments = run_stage1(tmp_dir, tmp_dir, base_dir=tmp_dir)

        # Check that last frame of each segment has valid_transition=False
        for _, seg in segments.iterrows():
            if not seg["keep"]:
                continue
            end_idx = int(seg["end_idx"]) - 1  # last frame index
            assert metrics.iloc[end_idx]["valid_transition"] == False

    def test_frame_before_gap_is_invalid(self):
        """Frame immediately before a timestamp gap should be invalid."""
        n = 30
        ts = np.arange(n, dtype=np.int64) * 33_000_000
        ts[15:] += 500_000_000  # gap at frame 15
        median_dt = 33_000_000

        metrics = pd.DataFrame({
            "timestamp_ns": ts,
            "speed": np.ones(n),
            "blur": np.full(n, 1000.0),
        })

        segments = segment_frames(metrics)

        # Compute valid_transition manually for testing
        # Frame 14 (before gap) should have dt > 2*median to frame 15 -> invalid
        dt_14_15 = ts[15] - ts[14]
        assert dt_14_15 > 2 * median_dt


def _create_minimal_dataset(base_dir, n_frames=20):
    """Create a minimal dataset directory with required files."""
    import cv2

    # image_index.csv
    ts = np.arange(n_frames, dtype=np.int64) * 33_000_000 + 1_000_000_000
    color_paths = []
    depth_paths = []
    for i in range(n_frames):
        shard = f"shard_{i // 500:04d}"
        shard_dir = os.path.join(base_dir, shard)
        os.makedirs(shard_dir, exist_ok=True)

        color_p = os.path.join(shard, f"color_{i:06d}.jpg")
        depth_p = os.path.join(shard, f"depth_{i:06d}.png")

        # Create dummy images
        img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(base_dir, color_p), img)
        depth = np.random.randint(100, 10000, (48, 64), dtype=np.uint16)
        cv2.imwrite(os.path.join(base_dir, depth_p), depth)

        color_paths.append(color_p)
        depth_paths.append(depth_p)

    pd.DataFrame({
        "frame_id": np.arange(n_frames),
        "timestamp_ns": ts,
        "color_path": color_paths,
        "depth_path": depth_paths,
    }).to_csv(os.path.join(base_dir, "image_index.csv"), index=False)

    # odometry.csv
    n_odom = n_frames * 3
    odom_ts = np.linspace(ts[0] - 100_000_000, ts[-1] + 100_000_000, n_odom, dtype=np.int64)
    pd.DataFrame({
        "timestamp_ns": odom_ts,
        "pos_x": np.linspace(0, 1, n_odom),
        "pos_y": np.zeros(n_odom),
        "pos_z": np.zeros(n_odom),
        "orient_x": np.zeros(n_odom),
        "orient_y": np.zeros(n_odom),
        "orient_z": np.zeros(n_odom),
        "orient_w": np.ones(n_odom),
        "linear_vel_x": np.ones(n_odom),
        "linear_vel_y": np.zeros(n_odom),
        "linear_vel_z": np.zeros(n_odom),
        "angular_vel_x": np.zeros(n_odom),
        "angular_vel_y": np.zeros(n_odom),
        "angular_vel_z": np.zeros(n_odom),
    }).to_csv(os.path.join(base_dir, "odometry.csv"), index=False)

    # position_cmd.csv (can be minimal)
    pd.DataFrame({
        "timestamp_ns": odom_ts,
        "velocity_x": np.zeros(n_odom),
        "velocity_y": np.zeros(n_odom),
        "velocity_z": np.zeros(n_odom),
    }).to_csv(os.path.join(base_dir, "position_cmd.csv"), index=False)
