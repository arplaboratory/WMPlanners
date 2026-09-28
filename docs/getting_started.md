# Getting started

Run these commands from the local workspace shown below. The repository is the Python project and contains the ROS 2 package at `ros2/wm_planner_ros`.

## Python workspace setup

Refer to the [README Quick Start](https://github.com/lquan9/wm_planner#quick-start) for workspace setup instructions.

## Build the ROS 2 wrapper

```bash
source /opt/ros/jazzy/setup.bash # use setup.zsh from zsh
source $WM_VENV/bin/activate
colcon build --base-paths src/wm_planner/ros2 --packages-select wm_planner_ros
source install/setup.bash
```

**Read next**
- [Data pipeline](data_pipeline.md) prepares episodes from bag recordings.
- [World model](world_model.md) explains the model families and planning path.
- [Training](training.md) contains the existing training and evaluation examples.
- [ROS2](ros2.md) contains ROS2 integration details (e.g., launch arguments, checkpoint requirements, topics and node parameters).
