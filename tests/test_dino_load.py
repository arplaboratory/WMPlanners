"""DINOv2 registration and local backbone integration."""

from pathlib import Path

import pytest
import torch

from world_model.backbones import create_model
from world_model.backbones.dino import DINOv2Encoder
from world_model.backbones.registry import _BUILDERS


@pytest.fixture(scope="module")
def dino_bundle():
    hubconf = Path(__file__).resolve().parents[1] / "external" / "dino" / "hubconf.py"
    if not hubconf.is_file():
        pytest.skip("external/dino submodule is not installed")
    return create_model(
        "dinov2", model_name="dinov2_vits14", pretrained=False, device="cpu"
    )


def test_dinov2_registered():
    assert "dinov2" in _BUILDERS


def test_dinov2_loads_from_local_submodule(dino_bundle):
    assert dino_bundle.modality == "image"
    assert dino_bundle.feature_dim == 384


def test_dinov2_forward_pass(dino_bundle):
    with torch.no_grad():
        output = dino_bundle.model(torch.zeros(1, 3, 224, 224))

    assert output.ndim == 3
    assert output.shape[0] == 1
    assert output.shape[-1] == dino_bundle.feature_dim


def test_dinov2_encoder_applies_imagenet_normalization():
    class CaptureModel(torch.nn.Module):
        def get_intermediate_layers(self, image, n):
            self.image = image
            return [torch.zeros(image.shape[0], 1, 4)]

    wrapped = CaptureModel()
    encoder = DINOv2Encoder(wrapped, emb_dim=4)
    encoder(torch.ones(1, 3, 2, 2))

    expected = (
        torch.ones(1, 3, 2, 2)
        - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ) / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    torch.testing.assert_close(wrapped.image, expected)
