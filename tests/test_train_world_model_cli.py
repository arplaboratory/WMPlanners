"""Standard world-model CLI contracts."""

import copy

import torch

from scripts.world_model.train_world_model import build_parser, update_ema_target


def test_dinov3_cli_uses_backend_specific_model_default():
    args = build_parser().parse_args(
        ["--data_path", "dataset", "--backbone", "dinov3"]
    )

    assert args.backbone == "dinov3"
    assert args.backbone_name is None


def test_dinov3_cli_accepts_weights_path():
    args = build_parser().parse_args(
        [
            "--data_path",
            "dataset",
            "--backbone",
            "dinov3",
            "--backbone_weights",
            "/models/dinov3.pth",
        ]
    )

    assert args.backbone_weights == "/models/dinov3.pth"


def test_ema_update_does_not_mutate_shared_frozen_parameters():
    online = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    for parameter in online[0].parameters():
        parameter.requires_grad = False
    target = copy.deepcopy(online, {id(online[0]): online[0]})
    frozen_before = online[0].weight.detach().clone()
    target_before = target[1].weight.detach().clone()
    with torch.no_grad():
        online[1].weight.add_(2.0)

    update_ema_target(online, target, tau=0.5)

    torch.testing.assert_close(online[0].weight, frozen_before)
    torch.testing.assert_close(target[1].weight, target_before + 1.0)
