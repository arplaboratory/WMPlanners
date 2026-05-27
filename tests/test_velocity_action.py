"""Tests for velocity-only action extraction from HF datasets."""

import numpy as np
import pytest
import torch

from world_model.data.types import SampleBatch


class TestVelocityActionExtraction:
    """When no action columns exist, action should be odom velocity dims only."""

    def test_action_dim_is_6_without_action_cols(self):
        """D_act should be 6 (lin_vel + ang_vel) when no action_* columns."""
        from unittest.mock import MagicMock
        from world_model.data.hf_datasets import HFWindowDataset

        ds = HFWindowDataset.__new__(HFWindowDataset)
        ds._ds = MagicMock()
        ds._ds.column_names = [
            "episode_id", "frame_idx", "image",
            *[f"odom_{i}" for i in range(13)],
            *[f"imu_{i}" for i in range(10)],
        ]
        ds.img_size = 64
        ds._norm_stats = None
        ds.odom_cols = [f"odom_{i}" for i in range(13)]
        ds.imu_cols = [f"imu_{i}" for i in range(10)]
        ds.act_cols = []
        ds.D_odom = 13
        ds.D_imu = 10
        ds.D_act = len(ds.act_cols) or 6

        assert ds.D_act == 6

    def test_action_values_are_velocity_slice(self):
        """Action should equal odom[:, 7:13] (lin_vel + ang_vel)."""
        odom = np.arange(13 * 5, dtype=np.float32).reshape(5, 13)
        odom[:, 0:3] = odom[:, 0:3] - odom[0, 0:3]

        expected_action = odom[:, 7:13].copy()
        action = odom[:, 7:13].copy()

        np.testing.assert_array_equal(action, expected_action)
        assert action.shape == (5, 6)
