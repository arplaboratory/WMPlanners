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


@register("dinov2")
def build_dinov2(
    repo_dir: str = str(_ROOT / "external/dino"),
    model_name: str = "dinov2_vitb14",
    pretrained: bool = True,
    device: str = "cpu",
    **_kwargs,
) -> ModelBundle:
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
    return ModelBundle(
        model=model,
        preprocess=default_preprocess,
        feature_dim=feature_dim,
        modality="image",
    )
