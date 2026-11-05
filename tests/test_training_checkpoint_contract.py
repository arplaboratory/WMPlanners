from types import SimpleNamespace

import torch

from scripts.world_model.train_world_model import (
    restore_target_model,
    training_contract,
)


def test_training_contract_preserves_pose_action_semantics():
    dataset = SimpleNamespace(
        manifest={
            "policy": {
                "action_contract": {
                    "source": "future_odom_pose",
                    "dimension": 7,
                    "semantic": "realized_pose_waypoint",
                }
            },
            "sensor_semantics": {
                "action": {
                    "components": ["x", "y", "z", "qx", "qy", "qz", "qw"],
                    "frame": "odom",
                }
            },
        }
    )

    contract = training_contract(dataset, SimpleNamespace(horizon=36, num_hist=1))

    assert contract["trained_horizon"] == 36
    assert contract["action"]["source"] == "future_odom_pose"
    assert contract["action"]["position_origin"] == "window_start"
    assert contract["action"]["frame"] == "odom"


def test_restore_target_model_uses_saved_ema_weights():
    online = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    with torch.no_grad():
        ema.weight.fill_(3.0)
        ema.bias.fill_(4.0)

    restored = restore_target_model(
        target, {"model": online.state_dict(), "target_model": ema.state_dict()}
    )

    assert restored
    assert torch.equal(target.weight, ema.weight)
    assert torch.equal(target.bias, ema.bias)


def test_restore_target_model_supports_legacy_checkpoint():
    online = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)

    restored = restore_target_model(target, {"model": online.state_dict()})

    assert not restored
    assert torch.equal(target.weight, online.weight)
    assert torch.equal(target.bias, online.bias)
