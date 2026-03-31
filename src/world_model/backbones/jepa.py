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
        model = model_bundle[0]
    else:
        model = model_bundle

    model.eval().to(device)

    feature_dim = getattr(model, "embed_dim", None)

    logger.info(f"Loaded VJEPA2 model: {model_name}, feature dimension: {feature_dim}")
    return ModelBundle(
        model=model,
        preprocess=lambda x: x,  # Identity – VJEPA2 expects raw video
        feature_dim=feature_dim,
        modality="video",
    )
