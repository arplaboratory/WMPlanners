# Getting started

Run these commands from the local workspace shown below. The repository is the Python project and contains the ROS 2 package at `ros2/wm_planner_ros`.

## Python workspace setup

For workspace setup, follow the Quick Start in the repository README, available through the Source code link on the <a href="../index.html">project homepage</a>.

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
