from __future__ import annotations
import math
from collections.abc import Sequence
import torch
import isaaclab.utils.math as math_utils
from rloclab.quadruped.tasks.direct.rloclab.rloclab_env import RloclabEnv
from .rloclab_backflip_env_cfg import RloclabBackflipEnvCfg

class RloclabBackflipEnv(RloclabEnv):
    cfg: RloclabBackflipEnvCfg

    def __init__(self, cfg: RloclabBackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self._episode_sums.update({
            "rear_up_hold": torch.zeros(self.num_envs, device=self.device),
            "takeoff_power": torch.zeros(self.num_envs, device=self.device),
            "air_spin": torch.zeros(self.num_envs, device=self.device),
            "land_orient": torch.zeros(self.num_envs, device=self.device),
            "land_still": torch.zeros(self.num_envs, device=self.device),
            "premature_penalty": torch.zeros(self.num_envs, device=self.device),
            "smoothness": torch.zeros(self.num_envs, device=self.device),
        })

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        phase = self._compute_phase()
        # phase clock
        phase_features = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        obs["policy"] = torch.cat([obs["policy"], phase_features], dim=-1)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        self.progress_buf += 1 
        current_time = self.progress_buf * self.step_dt
        
        # phases
        is_rear_phase = current_time < 1.5
        is_launch_phase = (current_time >= 1.5) & (current_time < 2.5)
        is_air_phase = (current_time >= 2.5) & (current_time < 4.0)
        is_land_phase = current_time >= 4.0

        # sensors
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        root_pitch = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)[1]
        
        # feet height
        foot_pos_w = self.robot.data.body_pos_w[:, self._feet_ids]
        front_h = torch.mean(foot_pos_w[:, :2, 2], dim=1)
        rear_h = torch.mean(foot_pos_w[:, 2:, 2], dim=1)

        # 1. rear up
        front_err = torch.clamp(0.4 - front_h, min=0.0)
        rear_err = torch.clamp(rear_h - 0.05, min=0.0)
        
        pose_quality = torch.exp(-front_err * 5.0) * torch.exp(-rear_err * 10.0)
        rear_reward = is_rear_phase * pose_quality * 5.0

        # 2. launch gate
        ready = (front_h > 0.3).float()

        # 3. takeoff
        vel_z = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_launch_phase * vel_z * ready * 30.0
        
        # 4. spin
        spin_vel = root_ang_vel[:, 1]
        spin_reward = is_air_phase * torch.clamp(-spin_vel, min=0.0) * 15.0
        
        # 5. land
        pitch_err = torch.abs(self._wrap_to_pi(root_pitch))
        land_orient_reward = is_land_phase * torch.exp(-pitch_err * 5.0) * 10.0
        
        vel_xy = torch.norm(root_lin_vel[:, :2], dim=-1)
        land_still_reward = is_land_phase * torch.exp(-vel_xy) * 5.0

        # penalties
        premature_jump = is_rear_phase * vel_z * -10.0
        smoothness = -torch.mean(torch.square(self._actions - self._previous_actions), dim=1) * 0.05

        rewards = {
            "rear_up_hold": rear_reward,
            "takeoff_power": takeoff_reward,
            "air_spin": spin_reward,
            "land_orient": land_orient_reward,
            "land_still": land_still_reward,
            "premature_penalty": premature_jump,
            "smoothness": smoothness
        }

        for k, v in rewards.items():
            if k in self._episode_sums: self._episode_sums[k] += v

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None: env_ids = self.robot._ALL_INDICES
        
        self.progress_buf[env_ids] = 0
        if "episode" not in self.extras: self.extras["episode"] = {}
        for k in self._episode_sums.keys():
            self.extras["episode"][k] = self._episode_sums[k][env_ids].mean()
            self._episode_sums[k][env_ids] = 0.0
        self._commands[env_ids] = 0.0

    def _compute_phase(self) -> torch.Tensor:
        return ((self.progress_buf * self.step_dt) / self.cfg.flip_period_s) % 1.0

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))