"""Body-frame odometry delta computation.

Computes 6-dim deltas (dx, dy, dz, ax, ay, az) between consecutive frames
in the body frame of reference at time t.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .utils import interpolate_to_timestamps


# Columns needed from odometry.csv
ODOM_POSE_COLS = [
    "pos_x", "pos_y", "pos_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
]


def compute_odom_deltas_for_frames(
    odom_df,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """Compute body-frame odometry deltas between consecutive frames.

    Parameters
    ----------
    odom_df : pd.DataFrame
        Odometry data with timestamp_ns and pose columns.
    frame_timestamps : (N,) int64
        Frame timestamps in nanoseconds.

    Returns
    -------
    (N-1, 6) float64 array of body-frame deltas:
        (dx, dy, dz, ax, ay, az) where (dx,dy,dz) is position change
        in body frame and (ax,ay,az) is axis-angle rotation.
    """
    n = len(frame_timestamps)
    if n < 2:
        return np.zeros((0, 6), dtype=np.float64)

    # Interpolate pose to frame timestamps
    interp = interpolate_to_timestamps(odom_df, frame_timestamps, ODOM_POSE_COLS)

    pos = interp[["pos_x", "pos_y", "pos_z"]].values  # (N, 3)
    quat = interp[["orient_x", "orient_y", "orient_z", "orient_w"]].values  # (N, 4)

    # Vectorized computation using batched scipy Rotation
    dp_world = pos[1:] - pos[:-1]  # (N-1, 3)

    # scipy Rotation uses (x, y, z, w) format — matches our column order
    r_t = Rotation.from_quat(quat[:-1])
    r_t1 = Rotation.from_quat(quat[1:])

    # Position delta in body frame at time t
    dp_body = r_t.inv().apply(dp_world)  # (N-1, 3)

    # Orientation delta: q_delta = q_t^{-1} * q_{t+1}, as axis-angle
    r_delta = r_t.inv() * r_t1
    axis_angle = r_delta.as_rotvec()  # (N-1, 3)

    return np.concatenate([dp_body, axis_angle], axis=1)  # (N-1, 6)
