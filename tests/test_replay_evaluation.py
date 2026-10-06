"""Both replay entry points share scoring while keeping risk behavior distinct."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.world_model.eval_mpc_replay import eval_mpc_replay
from scripts.world_model.eval_mpc_replay_no_risk import eval_mpc_replay_no_risk
from world_model.mpc import CEMConfig, constrain_actions
from world_model.replay import make_goal_tracking_cost, trajectory_tracking_cost


class TinyReplayModel:
    def __init__(self):
        self.state_head = nn.Identity()
        self.risk_calls = 0

    def encode(self, rgb, depth, odom, imu):
        return SimpleNamespace(z=torch.zeros(rgb.shape[0], 3, device=rgb.device))

    def rollout(self, z0, actions):
        return torch.cat((z0.unsqueeze(1), z0.unsqueeze(1) + actions.cumsum(1)), 1)

    def risk_head(self, states):
        self.risk_calls += 1
        return torch.full((*states.shape[:2], 1), 0.5, device=states.device)


class TinyReplayDataset:
    D_act = 3

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        steps = 4
        return {
            "rgb": np.zeros((steps, 3, 8, 8), dtype=np.float32),
            "depth": np.zeros((steps, 1, 8, 8), dtype=np.float32),
            "odom": np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32
            ),
            "imu": np.zeros((steps, 3), dtype=np.float32),
            "action": np.tile(np.array([1, 0, 0], dtype=np.float32), (steps, 1)),
        }


def test_short_horizon_tracking_cost_is_finite():
    states = torch.zeros(2, 3, 3)
    goal = torch.zeros(1, 2, 3)
    assert torch.isfinite(trajectory_tracking_cost(states, goal)).all()
    assert torch.isfinite(make_goal_tracking_cost()(states, None, goal)).all()


def test_pose_action_constraint_normalizes_quaternion():
    config = CEMConfig(action_dim=7, action_unit_norm_dims=(3, 4, 5, 6))
    actions = torch.tensor([[20.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    constrained = constrain_actions(
        actions, config, torch.full((7,), -10.0), torch.full((7,), 10.0))

    assert constrained[0, 0] == 10.0
    assert torch.linalg.vector_norm(constrained[0, 3:]).item() == pytest.approx(1.0)


@pytest.mark.parametrize("with_risk", [False, True])
def test_replay_modes_share_metrics_but_only_risk_mode_calls_head(with_risk):
    model = TinyReplayModel()
    dataset = TinyReplayDataset()
    evaluator = eval_mpc_replay if with_risk else eval_mpc_replay_no_risk
    summary = evaluator(
        model,
        dataset,
        None,
        "cpu",
        horizon=2,
        num_episodes=1,
        cem_population=8,
        cem_iters=2,
        cem_elites=2,
    )

    assert summary["trajectory_ade"]["mean"] == pytest.approx(0.0)
    assert summary["action_mse"]["mean"] == pytest.approx(0.0)
    assert summary["goal_success"]["mean"] == pytest.approx(0.0)
    assert (model.risk_calls > 0) is with_risk
    assert ("plan_cost" in summary) is not with_risk


def test_risk_cost_is_opt_in():
    states = torch.zeros(2, 3, 3)
    goal = torch.zeros(1, 2, 3)
    risk = torch.full((2, 3), 0.5)
    assert make_goal_tracking_cost()(states, risk, goal).sum().item() == 0.0
    assert make_goal_tracking_cost(risk_weight=2)(states, risk, goal).tolist() == [
        1.0,
        1.0,
    ]
