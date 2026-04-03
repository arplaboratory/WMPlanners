"""DINOv2 backbone loaded from the local ``external/dino`` submodule."""

from __future__ import annotations

import torch
from pathlib import Path
from logging import getLogger

from .base import ModelBundle
from .registry import register

logger = getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]


def default_preprocess(img, size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )(img)


class DINOv2Encoder(torch.nn.Module):
    """Thin nn.Module wrapper around a loaded DINOv2 model.

    Returns patch-level token features of shape ``(B, num_patches, emb_dim)``.
    """

    def __init__(self, model, emb_dim: int, patch_size: int = 14, image_size: int = 224):
        super().__init__()
        self.model = model
        self.emb_dim = emb_dim
        self.latent_ndim = 2  # output is (num_patches, emb_dim) per frame
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  (B, num_patches, emb_dim) patch tokens."""
        features = self.model.get_intermediate_layers(x, n=1)[0]  # (B, num_patches, D)
        return features


@register("dinov2")
def build_dinov2(
    repo_dir: str = str(_ROOT / "external/dino"),
    model_name: str = "dinov2_vitb14",
    pretrained: bool = True,
    device: str = "cpu",
    image_size: int = 224,
    **_kwargs,
) -> "DINOv2Encoder":
    """Load DINOv2 from the local ``dino`` submodule.

    Supported *model_name* values: ``dinov2_vits14``, ``dinov2_vitb14``,
    ``dinov2_vitl14``, ``dinov2_vitg14``.
    """
    try:
        model = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            pretrained=pretrained,
        )
    except TypeError as e:
        if "unsupported operand type" in str(e) and "|" in str(e):
            logger.error("DINOv2 requires Python 3.10+")
            raise RuntimeError("DINOv2 requires Python 3.10+.") from e
        raise

    model.eval().to(device)

    feature_dim = getattr(model, "embed_dim", None)
    if feature_dim is None:
        feature_dim_map = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        feature_dim = feature_dim_map.get(model_name)

    logger.info(f"Loaded DINOv2 model: {model_name}, feature dimension: {feature_dim}")
    patch_size = getattr(model, "patch_size", 14)
    return DINOv2Encoder(model=model, emb_dim=feature_dim, patch_size=patch_size, image_size=image_size)

