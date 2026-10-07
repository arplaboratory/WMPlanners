"""DINOv3 registration and local backbone integration."""

from pathlib import Path

import pytest
import torch

from world_model.architecture import WorldModel, checkpoint_backbone_kwargs
from world_model.backbones import create_model
from world_model.backbones.dinov3 import DINOv3Encoder
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


def test_dinov3_rejects_non_token_backbone():
    with pytest.raises(ValueError, match="Unsupported DINOv3 ViT"):
        create_model("dinov3", model_name="dinov3_convnext_tiny", pretrained=False)


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
