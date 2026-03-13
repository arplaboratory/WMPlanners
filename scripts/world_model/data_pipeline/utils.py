"""Shared utilities for the data-selection pipeline."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV, cast timestamp_ns to int64, and fill NaN depth_path with ''."""
    df = pd.read_csv(path)
    if "timestamp_ns" in df.columns:
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
    if "depth_path" in df.columns:
        df["depth_path"] = df["depth_path"].fillna("")
    return df


def interpolate_to_timestamps(
    source_df: pd.DataFrame,
    target_timestamps_ns: np.ndarray,
    value_columns: list[str],
) -> pd.DataFrame:
    """Linearly interpolate *value_columns* from source_df onto target timestamps.

    Returns a DataFrame indexed by the target timestamps with one column per
    requested value column.
    """
    src_ts = source_df["timestamp_ns"].values.astype(np.float64)
    tgt_ts = np.asarray(target_timestamps_ns, dtype=np.float64)

    result = pd.DataFrame({"timestamp_ns": target_timestamps_ns})
    for col in value_columns:
        result[col] = np.interp(tgt_ts, src_ts, source_df[col].values.astype(np.float64))
    return result


def extract_track_id(bag_name: str) -> str:
    """Extract the track ID from a bag filename like ``{track_id}_bag_YYYYMMDD_HHMMSS``."""
    m = re.match(r"^(.+?)_bag_\d{8}_\d{6}", bag_name)
    if m:
        return m.group(1)
    return bag_name


def resolve_path(relative_path: str, base_dir: str) -> str:
    """Resolve a relative image path against *base_dir*."""
    if not relative_path:
        return ""
    return os.path.join(base_dir, relative_path)
