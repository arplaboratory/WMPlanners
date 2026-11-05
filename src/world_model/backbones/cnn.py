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

    def __init__(self, in_ch: int = 4, z_dim: int = 128, image_size: int = 224,
                 pool_features: bool = False):
        super().__init__()
        self._z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Older CNN checkpoints project globally pooled feature maps; newer
        # checkpoints project the flattened map. Keep the default unchanged.
        self.pool = nn.AdaptiveAvgPool2d(1) if pool_features else nn.Identity()
        with torch.no_grad():
            probe = torch.zeros(1, in_ch, image_size, image_size)
            image_proj_dim = self.pool(self.net(probe)).flatten(1).shape[-1]
        self.proj = nn.Linear(image_proj_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, H, W)`` → ``(B, z_dim)``."""
        h = self.pool(self.net(x)).flatten(1)
        z = self.proj(h)
        z = F.layer_norm(z, z.shape[-1:])
        return z

    @property
    def feature_dim(self) -> int:
        return self._z_dim


@register("cnn")
def build_cnn(in_ch: int = 4, z_dim: int = 128, image_size: int = 224,
              device: str = "cpu", pool_features: bool = False, **_kwargs) -> ModelBundle:
    """Build the default CNN backbone."""
    model = VisionEncoder(in_ch=in_ch, z_dim=z_dim, image_size=image_size,
                          pool_features=pool_features).to(device)
    return ModelBundle(
        model=model,
        preprocess=lambda x: x,
        feature_dim=z_dim,
        modality="image",
    )
