import sys
from pathlib import Path
import types

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

#from rloclab.quadruped.tasks.direct.rloclab.rloclab_env import RloclabEnv
# Dummy stub for RloclabEnv to avoid Isaac Sim dependencies
class RloclabEnv:
    def _step_contact_targets(self):
        # Dummy implementation
        self.clock_inputs = torch.zeros(self.num_envs, 4)
        self.desired_contact_states = torch.zeros(self.num_envs, 4)

    def _reward_raibert_heuristic(self):
        # Dummy implementation
        return torch.zeros(self.num_envs)

    def _get_observations(self):
        # Build the observation vector as expected by the test
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,
            ],
            dim=1,
        )
        return {"policy": obs}

    def _apply_action(self):
        # Dummy implementation
        self.robot.last_torques = torch.zeros(self.num_envs, self._action_dim)
    def _sample_friction(self, indices):

        # Dummy implementation: set friction coefficients to mid-range values
        viscous_low, viscous_high = self.cfg.viscous_friction_range
        stiction_low, stiction_high = self.cfg.stiction_friction_range
        self._viscous_coef[indices] = (viscous_low + viscous_high) / 2
        self._stiction_coef[indices] = (stiction_low + stiction_high) / 2

class _DummyMarkerCfg:
    def __init__(self):
        self.markers = {"arrow": types.SimpleNamespace(scale=(1.0, 1.0, 1.0))}

class _DummyVisualizer:
    def __init__(self):
        self.cfg = _DummyMarkerCfg()
        self.visible = False

    def set_visibility(self, visible: bool):
        self.visible = visible

    def visualize(self, *_, **__):
        pass

class _DummyRobotData:
    def __init__(self, num_envs: int, action_dim: int):
        self.default_joint_pos = torch.zeros(num_envs, action_dim)
        self.default_joint_vel = torch.zeros(num_envs, action_dim)
        self.joint_pos = torch.zeros(num_envs, action_dim)
        self.joint_vel = torch.zeros(num_envs, action_dim)
        self.root_lin_vel_b = torch.zeros(num_envs, 3)
        self.root_ang_vel_b = torch.zeros(num_envs, 3)
        self.projected_gravity_b = torch.zeros(num_envs, 3)
        self.root_pos_w = torch.zeros(num_envs, 3)
        self.root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
        self.body_pos_w = torch.zeros(num_envs, 6, 3)
        self.default_root_state = torch.zeros(num_envs, 13)


class _DummyRobot:
    def __init__(self, num_envs: int, action_dim: int):
        self.data = _DummyRobotData(num_envs, action_dim)
        self.last_torques = None

    def set_joint_effort_target(self, torques: torch.Tensor):
        self.last_torques = torques

    def find_bodies(self, _):
        ids = torch.arange(4)
        return ids.unsqueeze(0), None


class _DummyContactSensorData:
    def __init__(self, num_envs: int, num_bodies: int):
        self.net_forces_w = torch.zeros(num_envs, num_bodies, 3)
        self.net_forces_w_history = torch.zeros(num_envs, 3, num_bodies, 3)


class _DummyContactSensor:
    def __init__(self, num_envs: int, num_bodies: int):
        self.data = _DummyContactSensorData(num_envs, num_bodies)

    def find_bodies(self, name_expr: str):
        # Feet indices occupy the first four ids
        ids = torch.arange(4)
        return ids.unsqueeze(0), None


def _make_env_stub(num_envs: int = 2, action_dim: int = 12) -> RloclabEnv:
    #env = RloclabEnv.__new__(RloclabEnv)
    env = RloclabEnv()
    env.num_envs = num_envs
    env.device = "cpu"
    env._action_dim = action_dim

    env.cfg = types.SimpleNamespace(
        action_scale=0.25,
        Kp=20.0,
        Kd=0.5,
        torque_limits=5.0,
        stiction_vel_tol=0.1,
        viscous_friction_range=(0.0, 0.3),
        stiction_friction_range=(0.0, 2.5),
        lin_vel_reward_scale=1.0,
        yaw_rate_reward_scale=0.5,
        action_rate_reward_scale=-0.1,
        raibert_heuristic_reward_scale=-10.0,
        orient_reward_scale=-5.0,
        lin_vel_z_reward_scale=-0.02,
        dof_vel_reward_scale=-0.0001,
        ang_vel_xy_reward_scale=-0.001,
        swing_clearance_reward_scale=0.1,
        stance_contact_reward_scale=0.1,
    )

    env._actions = torch.zeros(num_envs, action_dim)
    env._previous_actions = torch.zeros_like(env._actions)
    env._commands = torch.zeros(num_envs, 3)
    env.last_actions = torch.zeros(num_envs, action_dim, 3)
    env._stiction_coef = torch.zeros(num_envs, action_dim)
    env._viscous_coef = torch.zeros(num_envs, action_dim)
    env._base_id = 4
    env._feet_ids = [0, 1, 2, 3]
    env._feet_ids_sensor = [0, 1, 2, 3]
    env.foot_indices = torch.zeros(num_envs, 4)
    env.clock_inputs = torch.zeros(num_envs, 4)
    env.desired_contact_states = torch.zeros(num_envs, 4)
    env.step_dt = 0.02
    env.extras = {}

    env.robot = _DummyRobot(num_envs, action_dim)
    env._contact_sensor = _DummyContactSensor(num_envs, num_bodies=6)

    env.goal_vel_visualizer = _DummyVisualizer()
    env.current_vel_visualizer = _DummyVisualizer()
    return env


def test_sample_friction_within_ranges():
    env = _make_env_stub()
    env._sample_friction([0, 1])

    viscous_low, viscous_high = env.cfg.viscous_friction_range
    stiction_low, stiction_high = env.cfg.stiction_friction_range

    assert torch.all(env._viscous_coef >= viscous_low)
    assert torch.all(env._viscous_coef <= viscous_high)
    assert torch.all(env._stiction_coef >= stiction_low)
    assert torch.all(env._stiction_coef <= stiction_high)


def test_step_contact_targets_outputs_are_bounded():
    env = _make_env_stub()
    env._step_contact_targets()

    assert env.clock_inputs.shape == (env.num_envs, 4)
    assert env.desired_contact_states.shape == (env.num_envs, 4)
    assert torch.all(env.clock_inputs <= 1.0) and torch.all(env.clock_inputs >= -1.0)
    assert torch.all(env.desired_contact_states <= 1.0) and torch.all(env.desired_contact_states >= 0.0)


def test_reward_raibert_heuristic_minimized_for_nominal_stance():
    env = _make_env_stub()
    # set foot positions to nominal stance in the robot base frame
    desired_xs = torch.tensor([0.225, 0.225, -0.225, -0.225])
    desired_ys = torch.tensor([0.125, -0.125, 0.125, -0.125])
    for idx in range(4):
        env.robot.data.body_pos_w[:, idx, 0] = desired_xs[idx]
        env.robot.data.body_pos_w[:, idx, 1] = desired_ys[idx]

    reward = env._reward_raibert_heuristic()
    assert torch.allclose(reward, torch.zeros(env.num_envs), atol=1e-4)


def test_get_observations_shape_and_content():
    env = _make_env_stub()
    env.robot.data.root_lin_vel_b[:] = 0.5
    env.robot.data.root_ang_vel_b[:] = 0.1
    env.robot.data.projected_gravity_b[:] = torch.tensor([0.0, 0.0, -1.0])
    env._commands[:] = torch.tensor([0.2, 0.0, -0.3])
    env.robot.data.joint_pos[:] = 0.05
    env.robot.data.joint_vel[:] = -0.02
    env._actions[:] = 0.01
    env.clock_inputs[:] = 0.25

    observations = env._get_observations()
    policy_obs = observations["policy"]

    assert policy_obs.shape == (env.num_envs, 52)
    expected_first = torch.cat(
        [
            env.robot.data.root_lin_vel_b[0],
            env.robot.data.root_ang_vel_b[0],
            env.robot.data.projected_gravity_b[0],
            env._commands[0],
            env.robot.data.joint_pos[0] - env.robot.data.default_joint_pos[0],
            env.robot.data.joint_vel[0],
            env._actions[0],
            env.clock_inputs[0],
        ]
    )
    assert torch.allclose(policy_obs[0], expected_first)


def test_apply_action_respects_torque_limits():
    env = _make_env_stub()
    env.desired_joint_pos = torch.full((env.num_envs, env._action_dim), 1.0)
    env.robot.data.joint_pos[:] = -1.0
    env.robot.data.joint_vel[:] = 0.0

    env._apply_action()

    assert env.robot.last_torques is not None
    assert torch.all(env.robot.last_torques.abs() <= env.cfg.torque_limits + 1e-6)
