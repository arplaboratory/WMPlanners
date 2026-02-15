"""
Robust locomotion for forest terrain loaded from USD.
"""

from isaaclab.utils import configclass
from isaaclab import sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

from rloclab.quadruped.tasks.direct.rloclab.rloclab_env_cfg import RloclabEnvCfg

# hack
import os
usd_path = os.environ.get("USD_TERRAIN_PATH", "/default/path/to/terrain.usd")

@configclass
class RloclabUnevenEnvCfg(RloclabEnvCfg):
    """
    Configurations
    """

    # widen the observation vector with terrain descriptors
    observation_space = RloclabEnvCfg().observation_space + 3

    # denser contact visualization for debugging during terrain adaptation
    current_vel_visualizer_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.35, 0.35, 0.35)

    # load terrain from a USD file
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=usd_path,  # set using env USD_TERRAIN_PATH
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.1,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # fewer environments but more spacing for tall obstacles
    scene = RloclabEnvCfg().scene.replace(num_envs=1024, env_spacing=6.0)

    # reward shaping
    stability_reward_scale: float = 0.5
    contact_consistency_reward_scale: float = 0.3

    # observation hints sampled at reset
    slope_range = (-0.25, 0.25)
    step_height_range = (0.03, 0.18)
    friction_range = (0.6, 1.2)

    # stability shaping
    orientation_stability_sigma: float = 0.6
    contact_variance_sigma: float = 45.0
