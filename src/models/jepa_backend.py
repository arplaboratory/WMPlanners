
import torch
from .registry import register, ModelBundle
from logging import getLogger

logger = getLogger(__name__)

@register("vjepa2")
def build_vjepa2(
    repo_dir="external/vjepa2",
    model_name="vjepa2_vitg16",
    pretrained=True,
    device="cpu",
):
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
        preprocess=lambda x: x, # Identity preprocess since VJEPA2 expects raw video
        feature_dim=feature_dim,
        modality="video",
    )
