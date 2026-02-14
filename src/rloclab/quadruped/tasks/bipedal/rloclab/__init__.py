# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rloclab_bipedal_env import RloclabBipedalEnv
from .rloclab_bipedal_env_cfg import RloclabBipedalEnvCfg

__all__ = [
    "RloclabBipedalEnv",
    "RloclabBipedalEnvCfg",
]

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rloclab-Bipedal-v0",
    entry_point=f"{__name__}.rloclab_bipedal_env:RloclabBipedalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rloclab_bipedal_env_cfg:RloclabBipedalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
