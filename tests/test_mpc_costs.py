import torch

from world_model.mpc import (
    CEMConfig,
    StateCEMCostConfig,
    action_smoothness_cost,
    sample_cem_noise,
    state_trajectory_cost,
)


def test_state_cost_penalizes_reverse_progress_with_equal_terminal_error():
    state = torch.zeros(2, 4, 6)
    state[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    state[1, :, 0] = torch.tensor([0.0, -1.0, -2.0, 3.0])
    goal = torch.tensor([[3.0, 0.0, 0.0]])
    cfg = StateCEMCostConfig(reverse_progress_weight=1.0)

    cost = state_trajectory_cost(state, None, goal, cfg)

    assert cost[0] == 0.0
    assert cost[1] > cost[0]


def test_state_cost_cancels_constant_decoder_position_bias():
    state = torch.zeros(1, 4, 6)
    state[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    shifted = state.clone()
    shifted[..., :3] += torch.tensor([100.0, -20.0, 7.0])
    goal = torch.tensor([[3.0, 0.0, 0.0]])
    cfg = StateCEMCostConfig(
        terminal_weight=1.0,
        path_weight=0.5,
        reverse_progress_weight=1.0,
        trajectory_smoothness_weight=0.25,
    )

    assert torch.allclose(
        state_trajectory_cost(state, None, goal, cfg),
        state_trajectory_cost(shifted, None, goal, cfg),
    )


def test_pose_action_smoothness_is_quaternion_sign_invariant():
    actions = torch.zeros(1, 4, 7)
    actions[0, :, 0] = torch.arange(4, dtype=torch.float32)
    actions[0, :, 6] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    cfg = CEMConfig(
        horizon=4,
        action_dim=7,
        action_position_dims=(0, 1, 2),
        action_quaternion_dims=(3, 4, 5, 6),
    )
    low = torch.tensor([-4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    high = -low

    assert torch.allclose(
        action_smoothness_cost(actions, cfg, low, high), torch.zeros(1)
    )


def test_temporally_correlated_cem_noise():
    torch.manual_seed(4)
    cfg = CEMConfig(
        horizon=8, population=4096, action_dim=1, temporal_noise=0.8
    )

    noise = sample_cem_noise(cfg, torch.device("cpu"))
    correlation = torch.corrcoef(torch.stack((noise[:, 0, 0], noise[:, 1, 0])))[0, 1]

    assert correlation > 0.75


def test_temporal_noise_rejects_invalid_correlation():
    cfg = CEMConfig(
        horizon=2, population=2, elites=1, action_dim=1, temporal_noise=1.0
    )

    try:
        sample_cem_noise(cfg, torch.device("cpu"))
    except ValueError as error:
        assert "temporal_noise" in str(error)
    else:
        raise AssertionError("invalid temporal correlation was accepted")
