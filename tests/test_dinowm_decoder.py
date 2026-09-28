"""Decoder modes must satisfy the DinoWM image reconstruction contract."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.world_model.train_world_model_dinowm import Trainer, build_parser
from world_model.backbones.wm import WM
from world_model.metrics import eval_images


@pytest.mark.parametrize("decoder_type", ["vqvae", "transposed_conv"])
def test_decoder_factory_returns_trainable_images(decoder_type):
    args = build_parser().parse_args([
        "--data_path", "unused", "--img_size", "32",
        "--decoder_type", decoder_type,
        "--decoder_channel", "16", "--decoder_n_embed", "32",
        "--decoder_n_res_block", "1", "--decoder_n_res_channel", "8",
        "--tconv_depth", "8", "--decoder_quantize",
    ])
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    trainer.encoder = SimpleNamespace(emb_dim=12)
    decoder = trainer._build_decoder()

    tokens = torch.randn(2, 1, 4, 12, requires_grad=True)
    images, commitment = decoder(tokens)
    assert images.shape == (2, 3, 32, 32)
    (images.square().mean() + commitment.mean()).backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_image_metrics_run_without_optional_lpips():
    images = torch.rand(2, 3, 32, 32)
    scores = eval_images(images, images)
    assert set(scores) == {"l1", "l2", "ssim", "mse", "psnr"}
    assert all(torch.isfinite(value) for value in scores.values())
    assert scores["mse"].item() == 0.0


@pytest.mark.parametrize("decoder_type", ["vqvae", "transposed_conv"])
def test_world_model_loss_trains_decoder(decoder_type):
    class TinyEncoder(nn.Module):
        emb_dim = 12

        def __init__(self):
            super().__init__()
            self.project = nn.Linear(3, 12)

        def forward(self, frames):
            pooled = frames.mean(dim=(-2, -1))
            return self.project(pooled).unsqueeze(1).expand(-1, 4, -1)

    args = build_parser().parse_args([
        "--data_path", "unused", "--img_size", "32",
        "--decoder_type", decoder_type,
        "--decoder_channel", "16", "--decoder_n_embed", "32",
        "--decoder_n_res_block", "1", "--decoder_n_res_channel", "8",
        "--tconv_depth", "8",
    ])
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    trainer.encoder = TinyEncoder()
    decoder = trainer._build_decoder()
    model = WM(
        image_size=32, num_hist=2, num_pred=1,
        encoder=trainer.encoder, proprio_encoder=nn.Linear(4, 12),
        action_encoder=nn.Linear(3, 12), predictor=nn.Identity(),
        decoder=decoder, proprio_dim=12, action_dim=12, raw_action_dim=3,
        concat_dim=0, num_action_repeat=1, num_proprio_repeat=1,
        train_decoder=True, train_predictor=True,
    )
    observations = {
        "visual": torch.rand(2, 3, 3, 32, 32),
        "proprio": torch.rand(2, 3, 4),
    }
    actions = torch.rand(2, 3, 3)
    _, prediction, reconstruction, loss, parts = model(observations, actions)
    assert prediction.shape == (2, 2, 3, 32, 32)
    assert reconstruction.shape == (2, 3, 3, 32, 32)
    assert "decoder_loss_pred" in parts
    loss.backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())
