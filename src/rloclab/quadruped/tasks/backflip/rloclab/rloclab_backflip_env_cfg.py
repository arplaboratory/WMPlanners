# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from rloclab.quadruped.tasks.direct.rloclab.rloclab_env_cfg import RloclabEnvCfg

@configclass
class RloclabBackflipEnvCfg(RloclabEnvCfg):
    # timing
    episode_length_s = 6.0
    flip_period_s = 5.0

    # power
    action_scale = 1.0
    
    # physics
    Kp = 20.0
    Kd = 0.5
    torque_limits = 45.0 

    # spaces
    observation_space = 48 + 4 + 2 

    # zero walking rewards
    raibert_heuristic_reward_scale = 0.0
    feet_clarety_reward_scale = 0.0
    swing_clearance_reward_scale = 0.0
    stance_contact_reward_scale = 0.0
    lin_vel_reward_scale = 0.0
    yaw_rate_reward_scale = 0.0
    
    # curriculum
    curriculum_duration_steps = 5_000_000.0
    takeoff_phase_portion = 0.2
    airborne_phase_start = 0.5
    airborne_phase_end = 0.8