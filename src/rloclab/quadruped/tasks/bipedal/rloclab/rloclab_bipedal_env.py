"""
Bipedal walking behavior.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils

from rloclab.quadruped.tasks.direct.rloclab.rloclab_env import RloclabEnv

from .rloclab_bipedal_env_cfg import RloclabBipedalEnvCfg


class RloclabBipedalEnv(RloclabEnv):
    """
    Task for training rear-leg bipedal walking.
    """

    cfg: RloclabBipedalEnvCfg

    def __init__(self, cfg: RloclabBipedalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._front_feet_ids = self._feet_ids[:2]
        self._front_feet_sensor_ids = self._feet_ids_sensor[:2]
        self._hind_feet_ids = self._feet_ids[2:]
        self._hind_feet_sensor_ids = self._feet_ids_sensor[2:]

        self._episode_sums.update(
            {
                "biped_velocity": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_upright": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_height": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_hind_contact": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_front_clearance": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_front_height": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "biped_action_smoothness": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            }
        )

    def _get_observations(self,
    ) -> dict:
        observations = super()._get_observations()
        phase = self.gait_indices
        phase_features = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        observations["policy"] = torch.cat([observations["policy"], phase_features], dim=-1)
        return observations

    def _get_rewards(self,
    ) -> torch.Tensor:
        self._step_contact_targets()

        # forward velocity tracking using an exponential error map
        forward_speed = self.robot.data.root_lin_vel_b[:, 0]
        speed_error = torch.square(forward_speed - self.cfg.target_forward_velocity)
        velocity_reward = torch.exp(-speed_error / (self.cfg.speed_sigma**2))

        # encourage an upright torso with a controlled height target
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        roll_pitch = torch.stack([roll, pitch], dim=1)
        upright_error = torch.sum(torch.square(roll_pitch), dim=1)
        upright_reward = torch.exp(-upright_error / (self.cfg.upright_sigma**2))

        height_error = torch.square(self.robot.data.root_pos_w[:, 2] - self.cfg.target_base_height)
        height_reward = torch.exp(-height_error / (self.cfg.height_sigma**2))

        # contact shaping: hind feet should alternate contact, front feet should stay light
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor]
        contact_force_mag = torch.norm(contact_forces, dim=-1)

        hind_contact_reward = self._tracking_contacts_reward(
            contact_force_mag[:, 2:],
            self.desired_contact_states[:, 2:],
            self.cfg.contact_force_target,
        )

        front_contact_force = torch.sum(contact_force_mag[:, :2], dim=1)
        front_clearance_reward = torch.exp(-front_contact_force / (self.cfg.front_contact_force_limit + 1e-6))

        front_heights = self.foot_positions_w[:, :2, 2]
        front_height_error = torch.mean(torch.square(front_heights - self.cfg.front_foot_height_target), dim=1)
        front_height_reward = torch.exp(-front_height_error / (self.cfg.front_height_sigma**2))

        # penalize sharp action jumps that break balance
        action_delta = self._actions - self._previous_actions
        action_smoothness = torch.mean(torch.square(action_delta), dim=1)

        rewards = {
            "biped_velocity": velocity_reward * self.cfg.velocity_reward_scale,
            "biped_upright": upright_reward * self.cfg.upright_reward_scale,
            "biped_height": height_reward * self.cfg.height_reward_scale,
            "biped_hind_contact": hind_contact_reward * self.cfg.hind_contact_reward_scale,
            "biped_front_clearance": front_clearance_reward * self.cfg.front_clearance_reward_scale,
            "biped_front_height": front_height_reward * self.cfg.front_height_reward_scale,
            "biped_action_smoothness": -action_smoothness * self.cfg.action_smoothness_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # focus commands on forward velocity and neutral yaw for stability
        target = self.cfg.target_forward_velocity
        self._commands[env_ids, 0] = torch.empty_like(self._commands[env_ids, 0]).uniform_(target - 0.3, target + 0.3)
        self._commands[env_ids, 1] = 0.0
        self._commands[env_ids, 2] = 0.0

    def _step_contact_targets(self):
        """
        Update gait phase and desired contact schedule for the hind legs.
        """

        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * self.cfg.gait_frequency, 1.0)
        hind_left_phase = self.gait_indices
        hind_right_phase = torch.remainder(self.gait_indices + self.cfg.hind_phase_offset, 1.0)

        hind_left_stance = hind_left_phase < self.cfg.hind_duty_cycle
        hind_right_stance = hind_right_phase < self.cfg.hind_duty_cycle

        # update clock inputs (front channels stay zero because they are tucked)
        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * hind_left_phase)
        self.clock_inputs[:, 1] = torch.sin(2 * math.pi * hind_right_phase)
        self.clock_inputs[:, 2] = 0.0
        self.clock_inputs[:, 3] = 0.0

        self.desired_contact_states[:, 0] = 0.0
        self.desired_contact_states[:, 1] = 0.0
        self.desired_contact_states[:, 2] = hind_left_stance.float()
        self.desired_contact_states[:, 3] = hind_right_stance.float()

        self.foot_indices = torch.stack(
            [
                torch.zeros_like(hind_left_phase),
                torch.zeros_like(hind_right_phase),
                hind_left_phase,
                hind_right_phase,
            ],
            dim=1,
        )

    def _tracking_contacts_reward(
        self,
        contact_forces: torch.Tensor,
        desired_contacts: torch.Tensor,
        target_force: float,
    ) -> torch.Tensor:
        """
        Reward matching desired contact schedule by comparing force magnitudes.
        """

        normalized_force = torch.clamp(contact_forces / target_force, max=2.0)
        contact_error = torch.square(normalized_force - desired_contacts)
        return -torch.sum(contact_error, dim=1)
