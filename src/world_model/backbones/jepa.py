"""V-JEPA2 backbone loaded from the ``external/vjepa2`` submodule."""

from __future__ import annotations

import torch
from logging import getLogger

from .base import ModelBundle
from .registry import register

logger = getLogger(__name__)


@register("vjepa2")
def build_vjepa2(
    repo_dir: str = "external/vjepa2",
    model_name: str = "vjepa2_vitg16",
    pretrained: bool = True,
    device: str = "cpu",
    **_kwargs,
) -> ModelBundle:
    """Load V-JEPA2 from the local ``vjepa2`` submodule."""
    model = torch.hub.load(
        repo_dir,
        model_name,
        source="local",
        pretrained=pretrained,
    )

    model.eval().to(device)

    feature_dim = getattr(model, "embed_dim", None)

    logger.info(f"Loaded VJEPA2 model: {model_name}, feature dimension: {feature_dim}")
    return ModelBundle(
        model=model,
        preprocess=lambda x: x,  # Identity – VJEPA2 expects raw video
        feature_dim=feature_dim,
        modality="video",
    )
