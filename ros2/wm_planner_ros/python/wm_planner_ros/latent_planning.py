"""Goal observations and action-position paths for latent MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoalObservation:
    """A synchronized sensor observation captured at the destination."""

    rgb: Any
    depth: Any | None
    odom: Any
    imu: Any


def positions_from_actions(actions: np.ndarray, dimensions: tuple[int, int, int],
                           origin: np.ndarray) -> np.ndarray:
    """Turn relative position setpoints into an absolute path, including t0."""
    if len(dimensions) != 3 or len(set(dimensions)) != 3 or any(
        dim < 0 or dim >= actions.shape[1] for dim in dimensions
    ):
        raise ValueError("action_position_dims must select three distinct action dimensions")
    positions = actions[:, dimensions].astype(np.float64) + origin[np.newaxis, :]
    return np.concatenate((origin[np.newaxis, :], positions), axis=0)
