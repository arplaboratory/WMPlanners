"""Tests for body-frame odometry delta computation."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from scripts.world_model.data_pipeline.odom_delta import compute_odom_deltas_for_frames


class TestComputeOdomDeltas:
    """Tests for compute_odom_deltas_for_frames."""

    def test_output_shape(self, sample_odom_df, sample_frame_timestamps):
        """N frames should produce N-1 deltas of 6 dims."""
        result = compute_odom_deltas_for_frames(sample_odom_df, sample_frame_timestamps)
        n = len(sample_frame_timestamps)
        assert result.shape == (n - 1, 6)
        assert result.dtype == np.float64

    def test_stationary_produces_zero_deltas(self):
        """A robot sitting still should produce near-zero deltas."""
        n_odom = 50
        ts_odom = np.linspace(0, 1_000_000_000, n_odom, dtype=np.int64)
        odom_df = pd.DataFrame({
            "timestamp_ns": ts_odom,
            "pos_x": np.zeros(n_odom),
            "pos_y": np.zeros(n_odom),
            "pos_z": np.zeros(n_odom),
            "orient_x": np.zeros(n_odom),
            "orient_y": np.zeros(n_odom),
            "orient_z": np.zeros(n_odom),
            "orient_w": np.ones(n_odom),
        })
        frame_ts = np.linspace(100_000_000, 900_000_000, 10, dtype=np.int64)
        result = compute_odom_deltas_for_frames(odom_df, frame_ts)
        assert result.shape == (9, 6)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_pure_forward_motion(self):
        """Moving +1m/s in world-x with identity orientation -> dx=dt in body frame."""
        n_odom = 100
        ts_odom = np.linspace(0, 1_000_000_000, n_odom, dtype=np.int64)
        t_sec = (ts_odom - ts_odom[0]) / 1e9
        odom_df = pd.DataFrame({
            "timestamp_ns": ts_odom,
            "pos_x": t_sec * 1.0,
            "pos_y": np.zeros(n_odom),
            "pos_z": np.zeros(n_odom),
            "orient_x": np.zeros(n_odom),
            "orient_y": np.zeros(n_odom),
            "orient_z": np.zeros(n_odom),
            "orient_w": np.ones(n_odom),
        })
        # 5 frames at 10 Hz -> dt = 0.1s between frames
        frame_ts = np.array([0, 100_000_000, 200_000_000, 300_000_000, 400_000_000],
                            dtype=np.int64)
        result = compute_odom_deltas_for_frames(odom_df, frame_ts)
        # With identity orientation, body-frame dx should equal world-frame dx
        # dt = 0.1s, speed = 1 m/s -> dx ~ 0.1 per step
        assert result.shape == (4, 6)
        np.testing.assert_allclose(result[:, 0], 0.1, atol=0.02)  # dx ~ 0.1
        np.testing.assert_allclose(result[:, 1], 0.0, atol=0.02)  # dy ~ 0
        np.testing.assert_allclose(result[:, 2], 0.0, atol=0.02)  # dz ~ 0
        np.testing.assert_allclose(result[:, 3:], 0.0, atol=0.02)  # no rotation

    def test_pure_yaw_rotation(self):
        """Pure 90-degree yaw should produce axis-angle along z-axis."""
        n_odom = 100
        ts_odom = np.linspace(0, 1_000_000_000, n_odom, dtype=np.int64)
        odom_df = pd.DataFrame({
            "timestamp_ns": ts_odom,
            "pos_x": np.zeros(n_odom),
            "pos_y": np.zeros(n_odom),
            "pos_z": np.zeros(n_odom),
            "orient_x": np.zeros(n_odom),
            "orient_y": np.zeros(n_odom),
            "orient_z": np.zeros(n_odom),
            "orient_w": np.ones(n_odom),
        })
        # Two frames: identity then 90-deg yaw
        frame_ts = np.array([0, 500_000_000], dtype=np.int64)
        # Override orient at t=500ms to 90-deg yaw: q = (0, 0, sin(pi/4), cos(pi/4))
        # Set the last half of odom samples to the rotated quaternion
        half = n_odom // 2
        odom_df.loc[half:, "orient_z"] = np.sin(np.pi / 4)
        odom_df.loc[half:, "orient_w"] = np.cos(np.pi / 4)

        result = compute_odom_deltas_for_frames(odom_df, frame_ts)
        assert result.shape == (1, 6)
        # Position deltas should be zero
        np.testing.assert_allclose(result[0, :3], 0.0, atol=1e-6)
        # Rotation should be ~90 deg around z-axis -> axis-angle = (0, 0, pi/2)
        np.testing.assert_allclose(result[0, 3], 0.0, atol=0.1)  # ax ~ 0
        np.testing.assert_allclose(result[0, 4], 0.0, atol=0.1)  # ay ~ 0
        assert abs(result[0, 5]) > 0.5  # az should be significant (near pi/2)

    def test_body_frame_rotation(self):
        """Forward motion while rotated 90-deg yaw should map to body dy, not dx."""
        n_odom = 100
        ts_odom = np.linspace(0, 1_000_000_000, n_odom, dtype=np.int64)
        t_sec = (ts_odom - ts_odom[0]) / 1e9
        # Robot is facing +y (90-deg yaw), moving in world +y
        odom_df = pd.DataFrame({
            "timestamp_ns": ts_odom,
            "pos_x": np.zeros(n_odom),
            "pos_y": t_sec * 1.0,  # moving in world +y
            "pos_z": np.zeros(n_odom),
            "orient_x": np.zeros(n_odom),
            "orient_y": np.zeros(n_odom),
            "orient_z": np.full(n_odom, np.sin(np.pi / 4)),  # 90-deg yaw
            "orient_w": np.full(n_odom, np.cos(np.pi / 4)),
        })
        frame_ts = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        result = compute_odom_deltas_for_frames(odom_df, frame_ts)
        # When robot faces +y (90-deg yaw), world +y motion = body +x motion
        assert result.shape == (2, 6)
        np.testing.assert_allclose(result[:, 0], 0.1, atol=0.02)  # body dx ~ 0.1
        np.testing.assert_allclose(result[:, 1], 0.0, atol=0.02)  # body dy ~ 0
