"""
Configuration for a bipedal walking that emphasizes rear-leg locomotion
while keeping the front legs tucked.
"""

from isaaclab.utils import configclass

from rloclab.quadruped.tasks.direct.rloclab.rloclab_env_cfg import RloclabEnvCfg

@configclass
class RloclabBipedalEnvCfg(RloclabEnvCfg):
    """Bipedal walking configuration tuned for stable two-leg hops."""

    # shorter horizon keeps training focused on the bipedal gait cycle
    episode_length_s = 15.0

    # append 2D gait phase to observations
    observation_space = RloclabEnvCfg().observation_space + 2

    # allow slightly larger actions to pop the base up and balance
    action_scale = 0.35
    debug_vis = False

    # gait timing
    gait_frequency: float = 2.2
    hind_phase_offset: float = 0.5
    hind_duty_cycle: float = 0.6

    # locomotion targets
    target_forward_velocity: float = 1.2
    speed_sigma: float = 0.45
    target_base_height: float = 0.55
    height_sigma: float = 0.04
    upright_sigma: float = 0.18

    # contact shaping
    contact_force_target: float = 120.0
    front_contact_force_limit: float = 80.0
    front_foot_height_target: float = 0.32
    front_height_sigma: float = 0.04

    # reward scaling
    velocity_reward_scale: float = 1.0
    upright_reward_scale: float = 0.7
    height_reward_scale: float = 0.6
    hind_contact_reward_scale: float = 0.4
    front_clearance_reward_scale: float = 0.6
    front_height_reward_scale: float = 0.3
    action_smoothness_scale: float = 0.05

    # environment scaling
    scene = RloclabEnvCfg().scene.replace(num_envs=1536, env_spacing=5.5)
    contact_sensor = RloclabEnvCfg().contact_sensor.replace(history_length=4)
