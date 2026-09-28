"""Shared time-synchronization utilities for the data pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_merge(
    target_ts: np.ndarray,
    src_df: pd.DataFrame,
    max_gap_ns: int | None = None,
) -> np.ndarray:
    """For each target timestamp, pick the nearest row in src_df.

    Assumes *src_df* has a ``timestamp_ns`` column. All other columns are
    treated as value columns and returned as an ``(N, D)`` float32 array.

    This is useful when sensor streams arrive at different rates: it aligns
    steering, throttle, pose, and other measurements to camera-frame
    timestamps.

    Parameters
    ----------
    target_ts : (N,) int64 array of target timestamps in nanoseconds.
    src_df : DataFrame with ``timestamp_ns`` and D value columns.
    max_gap_ns : If set, zero-fill rows where the nearest source timestamp
        is farther than this many nanoseconds from the target.
    """
    src_ts = src_df["timestamp_ns"].values
    vals = src_df.drop(columns=["timestamp_ns"]).values.astype(np.float32)
    n_target = len(target_ts)
    n_src = len(src_ts)

    if n_src == 0:
        return np.zeros((n_target, vals.shape[1]), dtype=np.float32)
    if n_src == 1:
        result = np.tile(vals[0], (n_target, 1))
        if max_gap_ns is not None:
            gaps = np.abs(src_ts[0] - target_ts)
            result = result.copy()
            result[gaps > max_gap_ns] = 0.0
        return result

    idx = np.searchsorted(src_ts, target_ts, side="left")
    idx = np.clip(idx, 1, n_src - 1)

    left = np.abs(src_ts[idx - 1] - target_ts)
    right = np.abs(src_ts[idx] - target_ts)
    closer = np.where(left <= right, idx - 1, idx)

    result = vals[closer]

    if max_gap_ns is not None:
        gaps = np.minimum(left, right)
        exceed = gaps > max_gap_ns
        if exceed.any():
            result = result.copy()
            result[exceed] = 0.0

    return result
