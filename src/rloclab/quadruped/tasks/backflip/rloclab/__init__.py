# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rloclab_backflip_env import RloclabBackflipEnv
from .rloclab_backflip_env_cfg import RloclabBackflipEnvCfg

__all__ = [
    "RloclabBackflipEnv",
    "RloclabBackflipEnvCfg",
]

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rloclab-Backflip-v0",
    entry_point=f"{__name__}.rloclab_backflip_env:RloclabBackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rloclab_backflip_env_cfg:RloclabBackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
