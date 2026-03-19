import numpy as np
import pandas as pd
import pytest

from scripts.world_model.sync_utils import nearest_merge


def _make_sensor_df(timestamps, values):
    """Helper to build a sensor DataFrame with timestamp_ns + one value column."""
    return pd.DataFrame({"timestamp_ns": timestamps, "val": values})


def test_exact_match():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([100, 200, 300])
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])


def test_nearest_neighbor():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([110, 260, 290])  # 110→100, 260→300, 290→300
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [1.0, 3.0, 3.0])


def test_single_source_row():
    src = _make_sensor_df([100], [5.0])
    target = np.array([90, 100, 200])
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [5.0, 5.0, 5.0])


def test_max_gap_zero_fill():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([100, 500])  # 500 is 200ns away from nearest (300)
    result = nearest_merge(target, src, max_gap_ns=150)
    assert result[0, 0] == 1.0  # within gap
    assert result[1, 0] == 0.0  # exceeds gap — zero-filled


def test_max_gap_none_no_zeroing():
    src = _make_sensor_df([100, 200], [1.0, 2.0])
    target = np.array([100, 9999])
    result = nearest_merge(target, src, max_gap_ns=None)
    assert result[1, 0] == 2.0  # no gap limit, takes nearest


def test_multi_column():
    src = pd.DataFrame({
        "timestamp_ns": [100, 200, 300],
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    target = np.array([100, 300])
    result = nearest_merge(target, src)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[0], [1.0, 10.0])
    np.testing.assert_array_equal(result[1], [3.0, 30.0])
