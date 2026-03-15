"""Shared fixtures for pipeline tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_odom_df():
    """Odometry DataFrame with 100 samples spanning 1 second."""
    n = 100
    ts = np.linspace(1_000_000_000, 2_000_000_000, n, dtype=np.int64)
    # Straight-line motion: +1m/s in x, slight yaw
    t_sec = (ts - ts[0]) / 1e9
    df = pd.DataFrame({
        "timestamp_ns": ts,
        "pos_x": t_sec * 1.0,  # 1 m/s forward
        "pos_y": np.zeros(n),
        "pos_z": np.zeros(n),
        "orient_x": np.zeros(n),
        "orient_y": np.zeros(n),
        "orient_z": np.sin(t_sec * 0.1),  # slow yaw
        "orient_w": np.cos(t_sec * 0.1),
        "linear_vel_x": np.ones(n),
        "linear_vel_y": np.zeros(n),
        "linear_vel_z": np.zeros(n),
        "angular_vel_x": np.zeros(n),
        "angular_vel_y": np.zeros(n),
        "angular_vel_z": np.full(n, 0.2),
    })
    return df


@pytest.fixture
def sample_frame_timestamps():
    """30 frame timestamps at ~30 Hz spanning 1 second."""
    return np.linspace(1_000_000_000, 2_000_000_000, 30, dtype=np.int64)
