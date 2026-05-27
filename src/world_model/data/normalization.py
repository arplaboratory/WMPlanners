"""Normalization statistics for odom, imu, and action data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch


class NormStats:
    """Stores per-channel mean/std for odom, imu, and action, and provides
    normalize / denormalize helpers that work on arbitrary-device tensors."""

    def __init__(
        self,
        odom_mean: torch.Tensor,
        odom_std: torch.Tensor,
        imu_mean: torch.Tensor,
        imu_std: torch.Tensor,
        act_mean: torch.Tensor,
        act_std: torch.Tensor,
    ) -> None:
        self.odom_mean = odom_mean
        self.odom_std = odom_std
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.act_mean = act_mean
        self.act_std = act_std

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        odom: np.ndarray,
        imu: np.ndarray,
        action: np.ndarray,
        eps: float = 1e-6,
    ) -> "NormStats":
        """Compute mean/std from numpy arrays of shape (N, D).

        Standard deviations are clamped to a minimum of *eps* so that
        constant channels never produce NaN during normalization.
        """

        def _stats(arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
            std = torch.tensor(arr.std(axis=0), dtype=torch.float32).clamp(min=eps)
            return mean, std

        odom_mean, odom_std = _stats(odom)
        imu_mean, imu_std = _stats(imu)
        act_mean, act_std = _stats(action)
        return cls(odom_mean, odom_std, imu_mean, imu_std, act_mean, act_std)

    # ------------------------------------------------------------------
    # Normalize / Denormalize
    # ------------------------------------------------------------------

    def normalize_odom(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.odom_mean.to(x.device)) / self.odom_std.to(x.device)

    def denormalize_odom(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.odom_std.to(x.device) + self.odom_mean.to(x.device)

    def normalize_imu(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.imu_mean.to(x.device)) / self.imu_std.to(x.device)

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.act_mean.to(x.device)) / self.act_std.to(x.device)

    def denormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act_std.to(x.device) + self.act_mean.to(x.device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "odom_mean": self.odom_mean,
                "odom_std": self.odom_std,
                "imu_mean": self.imu_mean,
                "imu_std": self.imu_std,
                "act_mean": self.act_mean,
                "act_std": self.act_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormStats":
        d = torch.load(path, map_location="cpu")
        return cls(
            odom_mean=d["odom_mean"],
            odom_std=d["odom_std"],
            imu_mean=d["imu_mean"],
            imu_std=d["imu_std"],
            act_mean=d["act_mean"],
            act_std=d["act_std"],
        )
