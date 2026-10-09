"""DINOv3 backbone loaded from the local ``external/dinov3`` submodule."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
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
_HF_CHECKPOINTS = {
    "dinov3_vitb16": (
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitb16_pretrain_lvd1689m-hf-native-v1.pth",
    ),
}


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Serialize checkpoint conversion across local torchrun workers."""
    with path.open("a+") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            yield
        finally:
            flock(lock_file.fileno(), LOCK_UN)


def _convert_hf_vitb16_state_dict(
    hf_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the Transformers DINOv3 ViT-B checkpoint to Meta's layout."""
    converted = {
        "cls_token": hf_state_dict["embeddings.cls_token"],
        "storage_tokens": hf_state_dict["embeddings.register_tokens"],
        "mask_token": hf_state_dict["embeddings.mask_token"].squeeze(1),
        "patch_embed.proj.weight": hf_state_dict[
            "embeddings.patch_embeddings.weight"
        ],
        "patch_embed.proj.bias": hf_state_dict[
            "embeddings.patch_embeddings.bias"
        ],
    }

    norm_prefix = next(
        (
            prefix
            for prefix in ("layernorm", "norm")
            if f"{prefix}.weight" in hf_state_dict
        ),
        None,
    )
    if norm_prefix is None:
        raise KeyError("Hugging Face checkpoint does not contain the final layer norm")
    converted["norm.weight"] = hf_state_dict[f"{norm_prefix}.weight"]
    converted["norm.bias"] = hf_state_dict[f"{norm_prefix}.bias"]

    for index in range(12):
        hf_layer = f"layer.{index}"
        native_layer = f"blocks.{index}"
        for norm in ("norm1", "norm2"):
            converted[f"{native_layer}.{norm}.weight"] = hf_state_dict[
                f"{hf_layer}.{norm}.weight"
            ]
            converted[f"{native_layer}.{norm}.bias"] = hf_state_dict[
                f"{hf_layer}.{norm}.bias"
            ]
        converted[f"{native_layer}.ls1.gamma"] = hf_state_dict[
            f"{hf_layer}.layer_scale1.lambda1"
        ]
        converted[f"{native_layer}.ls2.gamma"] = hf_state_dict[
            f"{hf_layer}.layer_scale2.lambda1"
        ]

        q_weight = hf_state_dict[f"{hf_layer}.attention.q_proj.weight"]
        k_weight = hf_state_dict[f"{hf_layer}.attention.k_proj.weight"]
        v_weight = hf_state_dict[f"{hf_layer}.attention.v_proj.weight"]
        converted[f"{native_layer}.attn.qkv.weight"] = torch.cat(
            (q_weight, k_weight, v_weight), dim=0
        )
        q_bias = hf_state_dict[f"{hf_layer}.attention.q_proj.bias"]
        k_bias = hf_state_dict.get(
            f"{hf_layer}.attention.k_proj.bias", torch.zeros_like(q_bias)
        )
        v_bias = hf_state_dict[f"{hf_layer}.attention.v_proj.bias"]
        converted[f"{native_layer}.attn.qkv.bias"] = torch.cat(
            (q_bias, k_bias, v_bias), dim=0
        )
        converted[f"{native_layer}.attn.proj.weight"] = hf_state_dict[
            f"{hf_layer}.attention.o_proj.weight"
        ]
        converted[f"{native_layer}.attn.proj.bias"] = hf_state_dict[
            f"{hf_layer}.attention.o_proj.bias"
        ]
        converted[f"{native_layer}.mlp.fc1.weight"] = hf_state_dict[
            f"{hf_layer}.mlp.up_proj.weight"
        ]
        converted[f"{native_layer}.mlp.fc1.bias"] = hf_state_dict[
            f"{hf_layer}.mlp.up_proj.bias"
        ]
        converted[f"{native_layer}.mlp.fc2.weight"] = hf_state_dict[
            f"{hf_layer}.mlp.down_proj.weight"
        ]
        converted[f"{native_layer}.mlp.fc2.bias"] = hf_state_dict[
            f"{hf_layer}.mlp.down_proj.bias"
        ]
    return converted


def _prepare_hf_vitb16_checkpoint() -> Path:
    """Download and cache an HF checkpoint converted for Meta's implementation."""
    repo_id, filename = _HF_CHECKPOINTS["dinov3_vitb16"]
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / filename
    if output_path.is_file():
        return output_path

    with _exclusive_lock(output_path.with_suffix(f"{output_path.suffix}.lock")):
        if output_path.is_file():
            return output_path

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Automatic DINOv3 weight download requires huggingface-hub and "
                "safetensors"
            ) from exc

        logger.info("Downloading DINOv3 weights from Hugging Face: %s", repo_id)
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            token=os.environ.get("HF_TOKEN"),
        )
        converted = _convert_hf_vitb16_state_dict(load_file(safetensors_path))
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(converted, temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        logger.info("Cached converted DINOv3 weights at %s", output_path)
    return output_path


def _load_hf_vitb16(repo_dir: str) -> torch.nn.Module:
    checkpoint = _prepare_hf_vitb16_checkpoint()
    model = torch.hub.load(
        repo_dir, "dinov3_vitb16", source="local", pretrained=False
    )
    incompatible = model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if key != "rope_embed.periods" and not key.endswith(".attn.qkv.bias_mask")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Converted DINOv3 checkpoint is incompatible with the native model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


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

    ViT-B weights are downloaded from Hugging Face when ``weights`` is omitted.
    ``weights`` may also be a local native checkpoint path or URL supplied by
    Meta. ConvNeXt variants are excluded because they do not provide patch tokens.
    """
    if model_name not in _FEATURE_DIMS:
        raise ValueError(
            f"Unsupported DINOv3 ViT {model_name!r}; choose one of "
            f"{sorted(_FEATURE_DIMS)}"
        )

    if pretrained and weights is None and model_name == "dinov3_vitb16":
        model = _load_hf_vitb16(repo_dir)
    else:
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
