import pytest
import torch

from world_model.data.types import SampleBatch
from world_model.engine.trainer import TrainConfig, rollout_loss


class _FiniteModel(torch.nn.Module):
    def forward(self, rgb, depth, odom, imu, action, **kwargs):
        assert torch.isfinite(depth).all()
        batch = rgb.shape[0]
        latents = torch.zeros(batch, 2, 4, device=rgb.device)
        return {
            "z_targets": latents,
            "z_pred": latents,
            "state_pred": torch.zeros(batch, 2, 1, device=rgb.device),
            "risk_pred": torch.full((batch, 2, 1), 0.5, device=rgb.device),
        }


def _batch() -> SampleBatch:
    return SampleBatch(
        rgb=torch.zeros(1, 2, 3, 2, 2),
        depth=torch.full((1, 2, 1, 2, 2), float("nan")),
        imu=torch.zeros(1, 2, 1),
        odom=torch.zeros(1, 2, 1),
        action=torch.zeros(1, 2, 1),
        collision=torch.zeros(1, 2, 1),
    )


def test_rollout_loss_replaces_nonfinite_depth():
    loss, _ = rollout_loss(
        _FiniteModel(), _batch(),
        TrainConfig(device="cpu", horizon=1, objective="jepa-lite"),
    )
    assert torch.isfinite(loss)


def test_rollout_loss_rejects_nonfinite_actions():
    batch = _batch()
    batch.action[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite action"):
        rollout_loss(
            _FiniteModel(), batch,
            TrainConfig(device="cpu", horizon=1, objective="jepa-lite"),
        )
