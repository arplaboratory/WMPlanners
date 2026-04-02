"""Backbone base class and protocol for vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    """Container returned by every backbone builder."""

    model: nn.Module
    preprocess: Callable
    feature_dim: Optional[int]
    modality: str  # ``"image"`` or ``"video"``
    tubelet_size: int = 1


class BaseBackbone(ABC, nn.Module):
    """Abstract base class for custom backbone implementations.

    Subclasses should implement :meth:`forward` to return a flat feature
    tensor of shape ``(B, feature_dim)``.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input and return ``(B, feature_dim)`` features."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the output feature dimensionality."""
        ...
