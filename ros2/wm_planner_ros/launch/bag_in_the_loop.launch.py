"""Replay an MCAP bag through a robot-configured world-model adapter."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def _robot_config(name_or_path):
    """Resolve a packaged robot name or an explicit YAML file."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        path = candidate
    else:
        if candidate.name != name_or_path or name_or_path in {".", ".."}:
            raise ValueError("robot must be a config name or an existing YAML path")
        path = Path(get_package_share_directory("wm_planner_ros")) / "configs" / "robots" / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Robot config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("topics"), dict):
        raise ValueError(f"Robot config must contain a topics mapping: {path}")
    robot_name = config.get("name")
    if not isinstance(robot_name, str) or not robot_name or "/" in robot_name:
        raise ValueError(f"Robot config must have a valid name: {path}")
    return config


def _setup(context):
    model = LaunchConfiguration("model").perform(context).lower()
    if model not in {"cnn", "dino", "jepa-lite", "dinowm", "relpose"}:
        raise ValueError("model must be 'cnn', 'dino', 'jepa-lite', 'dinowm', or 'relpose'")
    config = _robot_config(LaunchConfiguration("robot").perform(context))
    robot_name = config["name"]
    configured_topics = config["topics"]
    if model == "relpose":
        required_topics = ("color", "odom")  # trained with depth and IMU zeroed
    else:
        required_topics = ("color", "odom", "imu") + (("depth",) if model != "dinowm" else ())
    for key in required_topics:
        topic = configured_topics.get(key)
        if not isinstance(topic, str) or not topic.startswith("/") or topic == "/":
            raise ValueError(f"Robot {robot_name}: topics.{key} must be an absolute ROS topic")
    color = configured_topics["color"]
    sensors = config.get("sensors", {})
    if not isinstance(sensors, dict):
        raise ValueError(f"Robot {robot_name}: sensors must be a mapping")
    encoding = sensors.get("color_encoding", "raw")
    if encoding == "compressed":
        if not color.endswith("/compressed"):
            raise ValueError(f"Robot {robot_name}: compressed color topic must end in /compressed")
        rgb_raw = color.removesuffix("/compressed")
    elif encoding == "raw":
        rgb_raw = color
    else:
        raise ValueError(f"Robot {robot_name}: unsupported color_encoding {encoding!r}")
    bag = Path(LaunchConfiguration("bag").perform(context)).expanduser()
    if not bag.is_file():
        raise FileNotFoundError(f"MCAP bag not found: {bag}")
    rate = float(LaunchConfiguration("rate").perform(context))
    if rate <= 0:
        raise ValueError("rate must be positive")
    checkpoint = LaunchConfiguration("checkpoint_path").perform(context)
    if model in {"jepa-lite", "dinowm", "relpose"} and not checkpoint:
        raise ValueError(f"{model} requires checkpoint_path")
    if checkpoint and not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    norm_stats = LaunchConfiguration("norm_stats_path").perform(context)
    if norm_stats and not Path(norm_stats).expanduser().is_file():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats}")

    if model == "relpose":
        bundle = LaunchConfiguration("planner_bundle_path").perform(context)
        if not bundle:
            bundle = str(Path(checkpoint).expanduser().parent / "planner_bundle.yaml")
        if not Path(bundle).expanduser().is_file():
            raise FileNotFoundError(
                f"Planner bundle not found: {bundle}. Create it with scripts/world_model/export_planner_bundle.py"
            )
        goal = [float(v) for v in LaunchConfiguration("goal_xyz").perform(context).split(",")]
        if len(goal) != 3:
            raise ValueError("goal_xyz must be three comma-separated numbers")
        planner = Node(
            package="wm_planner_ros",
            executable="wm_planner_bridge_node",
            name="wm_planner_bridge",
            namespace=robot_name,
            output="screen",
            parameters=[{
                "backend": "cnn",
                "dynamics_type": "mlp",
                "wm_mode": "standard",
                "checkpoint_path": checkpoint,
                "planner_bundle_path": bundle,
                "goal_frame": LaunchConfiguration("goal_frame"),
                "goal_xyz": goal,
                "device": LaunchConfiguration("device"),
                "rgb_topic": rgb_raw,
                "odom_topic": configured_topics["odom"],
                "use_sim_time": True,
                "plan_on_observation": True,
                "enable_mrp_action": False,
                "sync_slop": 0.05,
            }],
        )
    elif model == "dinowm":
        num_hist = int(LaunchConfiguration("num_hist").perform(context))
        if num_hist < 1:
            raise ValueError("num_hist must be at least one")
        planner = Node(
            package="wm_planner_ros",
            executable="dino_wm_planner_bridge_node",
            name="dino_wm_planner_bridge",
            namespace=robot_name,
            output="screen",
            parameters=[{
                "checkpoint_path": checkpoint,
                "device": LaunchConfiguration("device"),
                "dino_model_name": LaunchConfiguration("dino_model_name"),
                "num_hist": num_hist,
                "rgb_topic": rgb_raw,
                "odom_topic": configured_topics["odom"],
                "imu_topic": configured_topics["imu"],
                "use_sim_time": True,
            }],
        )
    else:
        is_jepa = model == "jepa-lite"
        backend = "dinov2" if model == "dino" else "cnn"
        experiment = "jepa_lite" if is_jepa else f"{model}_rssm_c1"
        planner = Node(
            package="wm_planner_ros",
            executable="wm_planner_bridge_node",
            name="wm_planner_bridge",
            namespace=robot_name,
            output="screen",
            parameters=[{
                "backend": backend,
                "dynamics_type": "mlp" if is_jepa else "rssm",
                "wm_mode": "jepa-lite" if is_jepa else "standard",
                "risk_weight": 0.0 if is_jepa else 0.5,
                "checkpoint_path": checkpoint,
                "checkpoint_experiment": experiment,
                "norm_stats_path": norm_stats,
                "device": LaunchConfiguration("device"),
                "rgb_topic": rgb_raw,
                "depth_topic": configured_topics["depth"],
                "odom_topic": configured_topics["odom"],
                "imu_topic": configured_topics["imu"],
                "use_sim_time": True,
                "plan_on_observation": True,
                "enable_mrp_action": False,
                "sync_slop": 0.05,
            }],
        )
    actions = [planner]
    if encoding == "compressed":
        actions.append(Node(
            package="image_transport",
            executable="republish",
            name=f"{robot_name}_rgb_decompressor",
            output="screen",
            arguments=[
                "--ros-args",
                "-r", f"in:={rgb_raw}",
                "-r", f"out:={rgb_raw}",
            ],
            parameters=[{
                "in_transport": "compressed",
                "out_transport": "raw",
                "use_sim_time": True,
            }],
        ))
    if model == "relpose":
        topics = [color, configured_topics["odom"]]
    else:
        topics = [color, configured_topics["odom"], configured_topics["imu"]]
        if model == "dinowm":
            topics += ["/tf", "/tf_static"]
        else:
            topics.append(configured_topics["depth"])
    playback = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", str(bag), "--storage", "mcap",
            "--clock", "--loop", "--rate", str(rate), "--topics", *topics,
        ],
        output="screen",
    )
    # Give ROS subscriptions time to connect; --loop also covers slow model loading.
    return [*actions, TimerAction(period=5.0, actions=[playback])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag", description="Path to an MCAP file with the configured topics"),
        DeclareLaunchArgument("robot", description="Name in configs/robots or path to a robot YAML config"),
        DeclareLaunchArgument("model", default_value="cnn",
                              description="cnn, dino, jepa-lite, dinowm, or relpose "
                                          "(checkpoints trained on configs/datasets/wm_relative_5hz.yaml)"),
        DeclareLaunchArgument("planner_bundle_path", default_value="",
                              description="relpose: planner_bundle.yaml; defaults beside the checkpoint"),
        DeclareLaunchArgument("goal_frame", default_value="body",
                              description="relpose: frame of goal_xyz and the goal topic, body or odom"),
        DeclareLaunchArgument("goal_xyz", default_value="8.0,0.0,0.0",
                              description="relpose: default goal until one arrives on the goal topic"),
        DeclareLaunchArgument("checkpoint_path", default_value="",
                              description="Local checkpoint; required for JEPA-lite and DinoWM"),
        DeclareLaunchArgument("norm_stats_path", default_value="",
                              description="Optional norm_stats.pt; defaults beside the checkpoint"),
        DeclareLaunchArgument("rate", default_value="0.1",
                              description="Bag playback speed multiplier"),
        DeclareLaunchArgument("dino_model_name", default_value="dinov2_vits14",
                              description="DinoWM encoder model name"),
        DeclareLaunchArgument("num_hist", default_value="3",
                              description="DinoWM observation history length"),
        DeclareLaunchArgument("device", default_value="cpu",
                              description="PyTorch device, e.g. cpu or cuda"),
        OpaqueFunction(function=_setup),
    ])
