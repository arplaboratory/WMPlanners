"""
Robust locomotion for uneven terrain.
"""

from isaaclab.utils import configclass
from isaaclab import sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

from rloclab.quadruped.tasks.direct.rloclab.rloclab_env_cfg import RloclabEnvCfg

class FlatSubTerrainCfg(SubTerrainBaseCfg):
    function = flat
    proportion = 1.0
    size = (10.0, 10.0)
    flat_patch_sampling = None

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

    flat_subterrain = FlatSubTerrainCfg(
        function=flat,
        proportion=1.0,
        size=(10.0, 10.0),
        flat_patch_sampling=None,
    )

    # use a procedural terrain generator for randomized fields
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            num_rows=10,
            num_cols=10,
            curriculum=True,
            size=6.0,
            border_width=0.5,
            border_height=0.2,
            color_scheme="default",
            horizontal_scale=0.1,
            vertical_scale=0.05,
            slope_threshold=0.3,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            cache_dir="/tmp/terrain_cache",
            sub_terrains={
                "flat": flat_subterrain,
            },
        ),
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
