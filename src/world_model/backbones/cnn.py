"""Simple CNN vision encoder backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone, ModelBundle
from .registry import register


class VisionEncoder(BaseBackbone):
    """Lightweight CNN encoder for RGB(+depth) observations.

    This is a simple baseline backbone that can be swapped with
    heavier pre-trained encoders (DINO, V-JEPA, MAE, …).
    """

    def __init__(self, in_ch: int = 4, z_dim: int = 128, image_size: int = 224):
        super().__init__()
        self._z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, H, W)`` → ``(B, z_dim)``."""
        h = self.net(x).flatten(1)
        z = self.proj(h)
        z = F.layer_norm(z, z.shape[-1:])
        return z

    @property
    def feature_dim(self) -> int:
        return self._z_dim


@register("cnn")
def build_cnn(in_ch: int = 4, z_dim: int = 128, image_size: int = 224, device: str = "cpu", **_kwargs) -> ModelBundle:
    """Build the default CNN backbone."""
    model = VisionEncoder(in_ch=in_ch, z_dim=z_dim, image_size=image_size).to(device)
    return ModelBundle(
        model=model,
        preprocess=lambda x: x,
        feature_dim=z_dim,
        modality="image",
    )
