"""Coordinate conversions shared by inference and its geometry tests."""
import numpy as np


def body_to_world_rotation(orientation):
    q = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Odometry requires a finite, nonzero quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)
