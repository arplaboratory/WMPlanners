import gymnasium as gym

from .drifting import F1TenthDriftRLEnvCfg, MushrDriftPlayEnvCfg, MushrDriftRLEnvCfg

from .drifting.config.agents import f1tenth as f1tenth_drift_agents
from .drifting.config.agents import mushr as mushr_drift_agents

gym.register(
    id="Template-Rloclab-Mushr-DriftRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MushrDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_drift_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrDriftPlayEnvCfg,
    },
)

gym.register(
    id="Template-Rloclab-F1Tenth-DriftRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": F1TenthDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{f1tenth_drift_agents.__name__}.rsl_rl_ppo_cfg:F1TenthPPORunnerCfg",
    },
)
