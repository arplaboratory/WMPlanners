"""DINOv3 registration and local backbone integration."""

from pathlib import Path

import pytest
import torch

from world_model.architecture import WorldModel, checkpoint_backbone_kwargs
from world_model.backbones import create_model
from world_model.backbones import dinov3 as dinov3_module
from world_model.backbones.dinov3 import (
    DINOv3Encoder,
    _convert_hf_vitb16_state_dict,
)
from world_model.backbones.registry import _BUILDERS


@pytest.fixture(scope="module")
def dinov3_bundle():
    hubconf = (
        Path(__file__).resolve().parents[1] / "external" / "dinov3" / "hubconf.py"
    )
    if not hubconf.is_file():
        pytest.skip("external/dinov3 submodule is not installed")
    return create_model(
        "dinov3", model_name="dinov3_vits16", pretrained=False, device="cpu"
    )


def test_dinov3_registered():
    assert "dinov3" in _BUILDERS


def test_dinov3_loads_from_local_submodule(dinov3_bundle):
    assert dinov3_bundle.modality == "image"
    assert dinov3_bundle.feature_dim == 384


def test_dinov3_forward_pass(dinov3_bundle):
    with torch.no_grad():
        output = dinov3_bundle.model(torch.zeros(1, 3, 224, 224))

    assert output.shape == (1, 196, dinov3_bundle.feature_dim)
    assert torch.isfinite(output).all()


def test_dinov3_encoder_applies_imagenet_normalization():
    class CaptureModel(torch.nn.Module):
        def get_intermediate_layers(self, image, n):
            self.image = image
            return [torch.zeros(image.shape[0], 1, 4)]

    wrapped = CaptureModel()
    encoder = DINOv3Encoder(wrapped, emb_dim=4)
    encoder(torch.ones(1, 3, 2, 2))

    expected = (
        torch.ones(1, 3, 2, 2)
        - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ) / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    torch.testing.assert_close(wrapped.image, expected)


def test_dinov3_encoder_rejects_nonfinite_features():
    class NonfiniteModel(torch.nn.Module):
        def get_intermediate_layers(self, image, n):
            return [torch.full((image.shape[0], 1, 4), float("nan"))]

    encoder = DINOv3Encoder(NonfiniteModel(), emb_dim=4)
    with pytest.raises(RuntimeError, match="non-finite patch features"):
        encoder(torch.zeros(1, 3, 2, 2))


def test_dinov3_rejects_non_token_backbone():
    with pytest.raises(ValueError, match="Unsupported DINOv3 ViT"):
        create_model("dinov3", model_name="dinov3_convnext_tiny", pretrained=False)


def test_hf_vitb16_conversion_maps_native_state_dict():
    hf_state = {
        "embeddings.cls_token": torch.tensor([1]),
        "embeddings.register_tokens": torch.tensor([2]),
        "embeddings.mask_token": torch.tensor([[[3]]]),
        "embeddings.patch_embeddings.weight": torch.tensor([4]),
        "embeddings.patch_embeddings.bias": torch.tensor([5]),
        "layernorm.weight": torch.tensor([6]),
        "layernorm.bias": torch.tensor([7]),
    }
    for index in range(12):
        prefix = f"layer.{index}"
        hf_state.update(
            {
                f"{prefix}.norm1.weight": torch.tensor([8]),
                f"{prefix}.norm1.bias": torch.tensor([9]),
                f"{prefix}.norm2.weight": torch.tensor([10]),
                f"{prefix}.norm2.bias": torch.tensor([11]),
                f"{prefix}.layer_scale1.lambda1": torch.tensor([12]),
                f"{prefix}.layer_scale2.lambda1": torch.tensor([13]),
                f"{prefix}.attention.q_proj.weight": torch.tensor([[14]]),
                f"{prefix}.attention.k_proj.weight": torch.tensor([[15]]),
                f"{prefix}.attention.v_proj.weight": torch.tensor([[16]]),
                f"{prefix}.attention.q_proj.bias": torch.tensor([17]),
                f"{prefix}.attention.v_proj.bias": torch.tensor([18]),
                f"{prefix}.attention.o_proj.weight": torch.tensor([19]),
                f"{prefix}.attention.o_proj.bias": torch.tensor([20]),
                f"{prefix}.mlp.up_proj.weight": torch.tensor([21]),
                f"{prefix}.mlp.up_proj.bias": torch.tensor([22]),
                f"{prefix}.mlp.down_proj.weight": torch.tensor([23]),
                f"{prefix}.mlp.down_proj.bias": torch.tensor([24]),
            }
        )

    converted = _convert_hf_vitb16_state_dict(hf_state)

    assert converted["mask_token"].shape == (1, 1)
    torch.testing.assert_close(
        converted["blocks.0.attn.qkv.weight"], torch.tensor([[14], [15], [16]])
    )
    torch.testing.assert_close(
        converted["blocks.0.attn.qkv.bias"], torch.tensor([17, 0, 18])
    )
    torch.testing.assert_close(converted["norm.weight"], torch.tensor([6]))


def test_vitb16_uses_automatic_hf_weights(monkeypatch):
    class FakeModel(torch.nn.Module):
        embed_dim = 768
        patch_size = 16

    calls = []

    def fake_load(repo_dir):
        calls.append(repo_dir)
        return FakeModel()

    monkeypatch.setattr(dinov3_module, "_load_hf_vitb16", fake_load)
    bundle = create_model(
        "dinov3",
        repo_dir="/tmp/dinov3",
        model_name="dinov3_vitb16",
        pretrained=True,
        device="cpu",
    )

    assert calls == ["/tmp/dinov3"]
    assert bundle.feature_dim == 768


def test_dinov3_world_model_checkpoint_round_trip():
    model_kwargs = {
        "odom_dim": 13,
        "imu_dim": 10,
        "act_dim": 7,
        "z_vis_dim": 32,
        "z_prop_dim": 16,
        "z_dim": 48,
        "backbone": "dinov3",
        "backbone_kwargs": {
            "model_name": "dinov3_vits16",
            "pretrained": False,
            "device": "cpu",
        },
        "freeze_backbone": True,
    }
    source = WorldModel(**model_kwargs)
    checkpoint = {
        "args": {
            "backbone": "dinov3",
            "backbone_name": "dinov3_vits16",
        },
        "model": source.state_dict(),
    }
    model_kwargs["backbone_kwargs"] = checkpoint_backbone_kwargs(
        checkpoint, "cpu"
    )
    restored = WorldModel(**model_kwargs)

    restored.load_state_dict(checkpoint["model"], strict=True)
