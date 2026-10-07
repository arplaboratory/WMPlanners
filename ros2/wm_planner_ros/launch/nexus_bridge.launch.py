"""Start the inference peer used by Navigation Nexus Only (MRP plugin)."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _planner_node(pyenv, **kwargs):
    if pyenv:
        from ianvs.launch import CustomNode

        return CustomNode(pyenv=pyenv, **kwargs)
    return Node(**kwargs)


def _setup(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    model = value("model")
    if model not in {"standard", "dinowm"}:
        raise ValueError("model must be standard or dinowm")
    robot = value("name").strip("/")
    if not robot or "/" in robot:
        raise ValueError("name must be one robot namespace segment")
    checkpoint = Path(value("checkpoint_path")).expanduser()
    if not checkpoint.is_file():
        raise ValueError("Nexus WM planning requires an existing checkpoint_path")

    def sensor_topic(topic):
        if not isinstance(topic, str) or not topic or topic.startswith("~"):
            raise ValueError("WM sensor topics must be robot-relative or absolute")
        return topic if topic.startswith("/") else f"/{robot}/{topic}"

    parameters = [{
        "robot_name": robot,
        "checkpoint_path": str(checkpoint),
        "device": value("device"),
        "rgb_topic": sensor_topic(value("rgb_topic")),
        "depth_topic": sensor_topic(value("depth_topic")),
        "odom_topic": sensor_topic(value("odom_topic")),
        "imu_topic": sensor_topic(value("imu_topic")),
        "enable_mrp_action": True,
        "plan_on_observation": False,
        "use_goal_topic": False,
        "mrp_action_name": "plan_mrp",
        "use_sim_time": value("use_sim_time").lower() == "true",
    }]
    return [_planner_node(
        value("pyenv"),
        package="wm_planner_ros",
        executable="dino_wm_planner_bridge_node" if model == "dinowm" else "wm_planner_bridge_node",
        name="bridge",
        namespace=f"{robot}/wm_planner",
        output="screen",
        parameters=parameters,
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("name", default_value="warthog1"),
        DeclareLaunchArgument("model",
                              default_value="dinowm",
                              description="dinowm or standard (CNN/DINOv2/DINOv3/JEPA)"),
        DeclareLaunchArgument("checkpoint_path",
                              description="Trained checkpoint matching this backend"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("pyenv", default_value="",
                              description="Optional Python virtual environment for the WM bridge"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("rgb_topic",
                              default_value="sensors/multisense_front/aux/image_rect_color"),
        DeclareLaunchArgument("depth_topic",
                              default_value="sensors/multisense_front/depth/image_rect"),
        DeclareLaunchArgument("odom_topic",
                              default_value="odometry/local"),
        DeclareLaunchArgument("imu_topic",
                              default_value="sensors/ouster/imu"),
        OpaqueFunction(function=_setup),
    ])
