"""V-JEPA2 backbone loaded from the ``external/vjepa2`` submodule."""

from __future__ import annotations

import torch
from pathlib import Path
from logging import getLogger
import sys
import contextlib
import types

from .base import ModelBundle
from .registry import register

logger = getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]

@contextlib.contextmanager
def _vjepa_import_context(repo_dir: str):

    old_src = sys.modules.get("src")

    vjepa_src = types.ModuleType("src")
    vjepa_src.__path__ = [str(Path(repo_dir) / "src")]
    sys.modules["src"] = vjepa_src

    try:
        yield
    finally:
        if old_src is not None:
            sys.modules["src"] = old_src
        else:
            sys.modules.pop("src", None)


class VJEPAWrapper(torch.nn.Module):
    """Wrapper to handle temporal repetition and pooling for V-JEPA2."""

    def __init__(self, encoder: torch.nn.Module, tubelet_size: int = 2):
        super().__init__()
        self.encoder = encoder
        self.tubelet_size = tubelet_size
        self.embed_dim = getattr(encoder, "embed_dim", 1408)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects (B, C, T, H, W) or (B, C, H, W).
        If 4D, repeats frame into a tubelet.
        """
        if x.ndim == 4:
            # (B, C, H, W) -> (B, C, 1, H, W) -> (B, C, tubelet, H, W)
            x = x.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)
        elif x.ndim == 5 and x.shape[2] < self.tubelet_size:
            # Pad temporal dim if too short
            repeats = (self.tubelet_size + x.shape[2] - 1) // x.shape[2]
            x = x.repeat(1, 1, repeats, 1, 1)[:, :, : self.tubelet_size]

        z = self.encoder(x)  # (B, N, D)
        if isinstance(z, (list, tuple)):
            z = z[0]

        # Global average pooling over tokens
        return z.mean(dim=1)


@register("vjepa2")
def build_vjepa2(
    repo_dir: str = str(_ROOT / "external/vjepa2"),
    model_name: str = "vjepa2_vit_giant",
    pretrained: bool = True,
    device: str = "cpu",
    **_kwargs,
) -> ModelBundle:
    """Load V-JEPA2 from the local ``vjepa2`` submodule."""
    with _vjepa_import_context(repo_dir):
        # torch.hub.load will prepend repo_dir to sys.path
        model_bundle = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            pretrained=pretrained,
        )

    # vjepa2 returns (encoder, predictor)
    if isinstance(model_bundle, (tuple, list)):
        encoder = model_bundle[0]
    else:
        encoder = model_bundle

    tubelet_size = getattr(encoder, "tubelet_size", 2)
    model = VJEPAWrapper(encoder, tubelet_size=tubelet_size)
    model.eval().to(device)

    feature_dim = model.embed_dim

    logger.info(f"Loaded VJEPA2 model: {model_name}, feature dimension: {feature_dim}, tubelet: {tubelet_size}")
    return ModelBundle(
        model=model,
        preprocess=lambda x: x,  # Identity – VJEPA2 expects raw video
        feature_dim=feature_dim,
        modality="video",
        tubelet_size=tubelet_size,
    )
