"""
Robust locomotion.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rloclab.quadruped.tasks.direct.rloclab.rloclab_env import RloclabEnv

from .rloclab_uneven_env_cfg import RloclabUnevenEnvCfg


class RloclabUnevenEnv(RloclabEnv):
    """
    Task wrapper that injects terrain-aware observations and rewards.
    """

    cfg: RloclabUnevenEnvCfg

    def __init__(self, cfg: RloclabUnevenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._terrain_descriptors = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums.update(
            {
                "terrain_stability": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "contact_consistency": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            }
        )

    def _get_observations(self,
    ) -> dict:
        observations = super()._get_observations()
        observations["policy"] = torch.cat([observations["policy"], self._terrain_descriptors], dim=-1)
        return observations

    def _get_rewards(self,
    ) -> torch.Tensor:
        base_reward = super()._get_rewards()

        # reward stable attitude on tilted terrain using projected gravity as a proxy for roll/pitch error
        attitude_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        terrain_stability_reward = torch.exp(-attitude_error / (self.cfg.orientation_stability_sigma**2))

        # encourage balanced foot loading to avoid slips on uneven surfaces
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor]
        contact_force_mag = torch.norm(contact_forces, dim=-1)
        contact_variance = torch.var(contact_force_mag, dim=1)
        contact_consistency_reward = torch.exp(-contact_variance / (self.cfg.contact_variance_sigma + 1e-6))

        rewards = {
            "terrain_stability": terrain_stability_reward * self.cfg.stability_reward_scale,
            "contact_consistency": contact_consistency_reward * self.cfg.contact_consistency_reward_scale,
        }

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return base_reward + torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._sample_terrain_descriptors(env_ids)

    def _sample_terrain_descriptors(self, env_ids: Sequence[int],
    ) -> None:
        """
        Sample slope, step height, and friction hints for the policy.
        """

        if env_ids is None or len(env_ids) == 0:
            return
        self._terrain_descriptors[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.slope_range)
        self._terrain_descriptors[env_ids, 1] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.step_height_range)
        self._terrain_descriptors[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.friction_range)

