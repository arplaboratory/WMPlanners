"""DINOv3 backbone loaded from the local ``external/dinov3`` submodule."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path

import torch

from .base import ModelBundle
from .dino import default_preprocess
from .registry import register

logger = getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_FEATURE_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}


class DINOv3Encoder(torch.nn.Module):
    """Normalize RGB images and return DINOv3 patch tokens."""

    def __init__(
        self, model: torch.nn.Module, emb_dim: int, patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        self.model = model
        self.emb_dim = emb_dim
        self.latent_ndim = 2
        self.num_patches = (image_size // patch_size) ** 2
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(
            dtype=x.dtype
        )
        return self.model.get_intermediate_layers(x, n=1)[0]


@register("dinov3")
def build_dinov3(
    repo_dir: str = str(_ROOT / "external/dinov3"),
    model_name: str = "dinov3_vitb16",
    pretrained: bool = True,
    device: str = "cpu",
    image_size: int = 224,
    weights: str | None = None,
    **_kwargs,
) -> ModelBundle:
    """Load a DINOv3 ViT from the local submodule.

    ``weights`` may be a local checkpoint path or URL supplied by Meta. ConvNeXt
    variants are intentionally excluded because they do not provide patch tokens.
    """
    if model_name not in _FEATURE_DIMS:
        raise ValueError(
            f"Unsupported DINOv3 ViT {model_name!r}; choose one of "
            f"{sorted(_FEATURE_DIMS)}"
        )

    load_kwargs = {
        "source": "local",
        "pretrained": pretrained,
    }
    if weights is not None:
        load_kwargs["weights"] = weights
    model = torch.hub.load(repo_dir, model_name, **load_kwargs)
    model.eval().to(device)

    feature_dim = getattr(model, "embed_dim", _FEATURE_DIMS[model_name])
    patch_size = getattr(model, "patch_size", 16)
    encoder = DINOv3Encoder(
        model=model,
        emb_dim=feature_dim,
        patch_size=patch_size,
        image_size=image_size,
    )
    logger.info(
        "Loaded DINOv3 model: %s, feature dimension: %s",
        model_name,
        feature_dim,
    )
    return ModelBundle(
        model=encoder,
        preprocess=default_preprocess,
        feature_dim=feature_dim,
        modality="image",
    )
