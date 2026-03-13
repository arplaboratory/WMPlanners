"""Tests for Stage 2 transition-level embedding."""

import os

import numpy as np
import pandas as pd
import pytest

from scripts.world_model.data_pipeline.stage2_embed import run_stage2, build_transition_descriptors, VISUAL_DIM


class TestBuildTransitionDescriptors:
    """Test transition descriptor construction."""

    def test_output_shape(self):
        """N frames should produce N-1 descriptors of 390 dims."""
        n = 10
        cls_tokens = np.random.randn(n, 384).astype(np.float32)
        odom_deltas = np.random.randn(n - 1, 6).astype(np.float64)
        valid_transition = np.ones(n - 1, dtype=bool)

        descriptors = build_transition_descriptors(cls_tokens, odom_deltas, valid_transition)
        assert descriptors.shape == (n - 1, 390)
        assert descriptors.dtype == np.float32

    def test_descriptor_is_concat_of_cls_and_delta(self):
        """Descriptor should be concat(CLS_t, odom_delta_t)."""
        cls_tokens = np.array([[1.0] * 384, [2.0] * 384], dtype=np.float32)
        odom_deltas = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float64)
        valid = np.array([True])

        desc = build_transition_descriptors(cls_tokens, odom_deltas, valid)
        assert desc.shape == (1, 390)
        np.testing.assert_allclose(desc[0, :384], 1.0)  # CLS_t (first frame)
        np.testing.assert_allclose(desc[0, 384:], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)

    def test_invalid_transitions_zeroed(self):
        """Invalid transitions should produce zero descriptors."""
        cls_tokens = np.ones((3, 384), dtype=np.float32)
        odom_deltas = np.ones((2, 6), dtype=np.float64)
        valid = np.array([True, False])

        desc = build_transition_descriptors(cls_tokens, odom_deltas, valid)
        # First transition valid, second invalid
        assert np.all(desc[0] != 0)
        np.testing.assert_allclose(desc[1], 0.0)


class TestRunStage2:
    """Integration tests for run_stage2."""

    def test_output_file_created(self, tmp_dir):
        """run_stage2 should create transitions_compact.npz."""
        dataset_dir, metrics_df, segments_df = _setup_stage2_inputs(tmp_dir)
        result = run_stage2(
            dataset_dir=dataset_dir,
            metrics_df=metrics_df,
            segments_df=segments_df,
            output_dir=tmp_dir,
            base_dir=tmp_dir,
            device="cpu",
        )
        assert result is not None
        bag_name = os.path.basename(dataset_dir)
        npz_path = os.path.join(tmp_dir, f"{bag_name}_transitions_compact.npz")
        assert os.path.isfile(npz_path)

    def test_npz_keys(self, tmp_dir):
        """Output npz should have required keys."""
        dataset_dir, metrics_df, segments_df = _setup_stage2_inputs(tmp_dir)
        run_stage2(
            dataset_dir=dataset_dir,
            metrics_df=metrics_df,
            segments_df=segments_df,
            output_dir=tmp_dir,
            base_dir=tmp_dir,
            device="cpu",
        )
        bag_name = os.path.basename(dataset_dir)
        data = np.load(os.path.join(tmp_dir, f"{bag_name}_transitions_compact.npz"),
                       allow_pickle=True)
        required_keys = [
            "descriptors", "odom_deltas", "cls_tokens",
            "frame_t_ids", "frame_t1_ids",
            "timestamps_t", "timestamps_t1",
            "valid_mask", "bag_name",
            "color_paths", "depth_paths",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_descriptor_dimensions(self, tmp_dir):
        """Descriptors should be (N-1, 390) and odom_deltas (N-1, 6)."""
        dataset_dir, metrics_df, segments_df = _setup_stage2_inputs(tmp_dir, n_frames=15)
        run_stage2(
            dataset_dir=dataset_dir,
            metrics_df=metrics_df,
            segments_df=segments_df,
            output_dir=tmp_dir,
            base_dir=tmp_dir,
            device="cpu",
        )
        bag_name = os.path.basename(dataset_dir)
        data = np.load(os.path.join(tmp_dir, f"{bag_name}_transitions_compact.npz"),
                       allow_pickle=True)
        n_kept = int(segments_df[segments_df["keep"]]["n_frames"].sum())
        assert data["descriptors"].shape[1] == 390
        assert data["odom_deltas"].shape[1] == 6
        assert data["cls_tokens"].shape[1] == VISUAL_DIM


def _setup_stage2_inputs(base_dir, n_frames=10):
    """Create minimal inputs for Stage 2 testing."""
    import cv2

    dataset_dir = os.path.join(base_dir, "test_bag")
    os.makedirs(dataset_dir, exist_ok=True)

    ts = np.arange(n_frames, dtype=np.int64) * 33_000_000 + 1_000_000_000
    color_paths = []
    depth_paths = []
    shard_dir = os.path.join(dataset_dir, "shard_0000")
    os.makedirs(shard_dir, exist_ok=True)

    for i in range(n_frames):
        color_p = f"shard_0000/color_{i:06d}.jpg"
        depth_p = f"shard_0000/depth_{i:06d}.png"
        # Create dummy images (small for speed)
        img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dataset_dir, color_p), img)
        depth = np.random.randint(100, 10000, (48, 64), dtype=np.uint16)
        cv2.imwrite(os.path.join(dataset_dir, depth_p), depth)
        color_paths.append(color_p)
        depth_paths.append(depth_p)

    metrics_df = pd.DataFrame({
        "frame_id": np.arange(n_frames),
        "timestamp_ns": ts,
        "color_path": color_paths,
        "depth_path": depth_paths,
        "speed": np.ones(n_frames),
        "blur": np.full(n_frames, 1000.0),
        "valid_transition": np.append(np.ones(n_frames - 1, dtype=bool), [False]),
    })

    segments_df = pd.DataFrame({
        "segment_id": [0],
        "start_idx": [0],
        "end_idx": [n_frames],
        "n_frames": [n_frames],
        "duration_s": [float(n_frames) / 30.0],
        "keep": [True],
    })

    # Create odometry.csv
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
    }).to_csv(os.path.join(dataset_dir, "odometry.csv"), index=False)

    return dataset_dir, metrics_df, segments_df
