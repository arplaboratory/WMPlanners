# ROS 2 planner wrapper

`ros2/wm_planner_ros` exposes two ROS 2 executables:

| Executable | Model | Observations |
|---|---|---|
| `wm_planner_bridge_node` | Standard `WorldModel`: CNN, DINOv2, or V-JEPA2 encoder with RSSM or MLP dynamics | Raw RGB, raw depth, odometry, IMU |
| `dino_wm_planner_bridge_node` | Separate DinoWM architecture with a token predictor | Raw RGB, odometry, IMU |

Both executables use the same Python bridge infrastructure in `bridge_adapter.py`: ROS subscriptions, goal handling, publishers, a bounded observation history, and a latest-observation-wins planning worker. `model_adapters.py` supplies the model-specific observation and planning methods. The standard adapter covers CNN+RSSM, DINOv2+RSSM, and JEPA-lite; the DinoWM adapter uses a three-frame history by default and keeps DinoWM's own CEM and TF path. JEPA-lite uses the standard `WorldModel` with a different training objective and normally MLP dynamics. See [World model](world_model.md) for the model architectures.

## Setup and launch

Follow [Getting started](getting_started.md) to build and source the wrapper. 

Install `image_transport`, `compressed_image_transport`, `rosbag2_transport`, and `rosbag2_storage_mcap` in the selected ROS distribution.
```bash
sudo apt-get install ros-$ROS_DISTRO-image-transport \
                     ros-$ROS_DISTRO-compressed-image-transport \
                     ros-$ROS_DISTRO-rosbag2_transport \
                     ros-$ROS_DISTRO-rosbag2-storage-mcap
```

Bag-in-the-loop launch:
```bash
ros2 launch wm_planner_ros bag_in_the_loop.launch.py \
  bag:=/path/to/recording.mcap robot:=<robot> \
  model:=cnn checkpoint_path:=/path/to/cnn_rssm/latest.pt device:=cuda
```

From the sourced workspace, replay an MCAP bag with a matching robot config and checkpoint:

```bash
ros2 launch wm_planner_ros bag_in_the_loop.launch.py \
  bag:=/path/to/recording.mcap robot:=<robot> \
  model:=cnn checkpoint_path:=/path/to/cnn_rssm/latest.pt device:=cuda
```

For DINOv2+RSSM, use `model:=dino` and its corresponding checkpoint. For JEPA-lite, first train the state decoder as shown in [Training](training.md), then launch with its updated checkpoint:

```bash
ros2 launch wm_planner_ros bag_in_the_loop.launch.py \
  bag:=/path/to/recording.mcap robot:=<robot> \
  model:=jepa-lite checkpoint_path:=/path/to/latest_with_state_head.pt device:=cuda
```

DinoWM uses the same bag launch with its own checkpoint format and encoder name:

```bash
ros2 launch wm_planner_ros bag_in_the_loop.launch.py \
  bag:=/path/to/recording.mcap robot:=<robot> \
  model:=dinowm checkpoint_path:=/path/to/model_latest.pth \
  dino_model_name:=dinov2_vits14 num_hist:=3 device:=cuda
```

Pass the MCAP path and choose its robot config with `robot:=NAME`, or use `robot:=/path/to/robot.yaml`. `model:=dino` selects DINOv2+RSSM in the standard adapter; `model:=dinowm` selects the separate DinoWM model adapter. The JEPA-lite checkpoint metadata chooses its CNN or DINOv2 encoder and trained dynamics. `device:=cpu` is the portable default. Stop the launch with Ctrl+C.

| Launch argument | Default | Effect |
|---|---|---|
| `bag` | required | MCAP file to replay; relative paths resolve from the launch command's working directory. |
| `robot` | required | Name of an installed `configs/robots/*.yaml` file, or an explicit YAML path. The config supplies the node namespace and recorded sensor topics. |
| `model` | `cnn` | `cnn` selects CNN+RSSM; `dino` selects DINOv2+RSSM; `jepa-lite` selects JEPA-lite; `dinowm` selects DinoWM. |
| `checkpoint_path` | empty | Local `.pt` or `.pth` checkpoint. Required for JEPA-lite and DinoWM. For `cnn` or `dino`, an empty value requests `cnn_rssm_c1/latest.pt` or `dino_rssm_c1/latest.pt` from the default Hugging Face repository. |
| `norm_stats_path` | empty | Standard-model `norm_stats.pt`; if empty, the bridge looks beside the checkpoint. DinoWM uses its own normalization parameters. |
| `rate` | `0.1` | Positive multiplier for bag playback speed. |
| `dino_model_name` | `dinov2_vits14` | DinoWM encoder name; ignored by the standard adapter. |
| `num_hist` | `3` | DinoWM observation history length; ignored by the standard adapter. |
| `device` | `cpu` | PyTorch device, such as `cpu` or `cuda`. |

The launch validates the bag and any supplied checkpoint or normalization file paths. The three standard-WorldModel choices read `args` and `model_config` from training checkpoints to construct the trained backbone, dynamics, image size, and sensor/action dimensions before loading the saved `model` weights. For `cnn` and `dino`, the checkpoint architecture must match the launch choice. The bridge still accepts older plain state dict checkpoints using its configured defaults. If a standard-model checkpoint download fails, the node continues with random weights; provide a trained checkpoint for meaningful plans. Missing normalization statistics also degrade trajectory quality.

JEPA-lite training leaves the state and risk heads unsupervised. Use `latest_with_state_head.pt` produced by `train_state_head_decoder.py` for goal-based planning. The launch sets `risk_weight:=0.0` for JEPA-lite, so its untrained risk head does not affect CEM scoring. The default `norm_stats.pt` lookup is beside the checkpoint; supply `norm_stats_path` if it is elsewhere.

## What the launch starts

1. The selected executable in the namespace set by the robot config's `name`. `cnn`, `dino`, and `jepa-lite` start `wm_planner_bridge_node`; `dinowm` starts `dino_wm_planner_bridge_node`. Standard RSSM choices use their selected backbone, while JEPA-lite uses checkpoint metadata. Both share the buffered planner worker; the standard adapter needs one observation and DinoWM uses `num_hist` observations.
2. For configs with `sensors.color_encoding: compressed`, `image_transport republish` turns the recorded RGB stream into raw `sensor_msgs/Image` on the topic without the `/compressed` suffix. This requires the `compressed_image_transport` plugin. Raw color configs use the recorded topic directly.
3. `ros2 bag play` after a five-second startup delay, with MCAP storage, `/clock`, the selected rate, and `--loop`. The standard adapter replays RGB, depth, odometry, and IMU; DinoWM replays RGB, odometry, IMU, `/tf`, and `/tf_static`. Looping lets the bridge receive later passes if model loading takes longer than the first pass.

The launch reads `topics.color`, `topics.odom`, and `topics.imu` from the selected robot YAML file. Pass `robot:=NAME` to select `configs/robots/NAME.yaml` from the source repository, or pass an explicit YAML path.

| Stream | Config key | Consumer |
|---|---|---|
| RGB | `topics.color`, `sensors.color_encoding` | Both adapters receive raw `sensor_msgs/Image`. |
| Depth | `topics.depth` | Standard adapter only; raw `sensor_msgs/Image`. |
| Odometry | `topics.odom` | Both adapters; `nav_msgs/Odometry`. |
| IMU | `topics.imu` | Both adapters; `sensor_msgs/Imu`. |
| TF | Global `/tf`, `/tf_static` | DinoWM's TF listener only. |

Both adapters use an approximate-time synchronizer and the ROS sensor-data QoS profile. The shared history buffer holds one synchronized observation for the standard adapter and `num_hist` observations for DinoWM. A dedicated worker plans from the newest complete snapshot; incoming observations replace any pending snapshot while a plan is running. The standard adapter's `plan_on_observation` parameter can disable automatic planning for the MRP action path.

## Goals, outputs, and node parameters

The launch does not replay a goal topic. With the launch defaults, either bridge plans toward `goal_xyz:=[0.0, 0.0, 0.0]` until it receives a `geometry_msgs/PoseStamped` on its private `~/goal` topic. The standard node's name is `wm_planner_bridge`; DinoWM's is `dino_wm_planner_bridge`. The selected node publishes:

| Topic | Type | Purpose |
|---|---|---|
| `/<robot_name>/<node_name>/planned_trajectory` | `nav_msgs/Path` | Chosen trajectory. |
| `/<robot_name>/<node_name>/planned_trajectory/markers` | `visualization_msgs/MarkerArray` | Visualization markers for the chosen trajectory. |
| `/<robot_name>/<node_name>/debug_metrics` | `std_msgs/Float32MultiArray` | Standard model: goal distance, mean risk, goal weight, risk weight. DinoWM: goal distance and goal weight. |

The launch leaves `publish_frame` at each node's default (`body`). The standard node also supports `goal_topic`, `goal_xyz`, `use_goal_topic`, `publish_frame`, `queue_size` (8), image size (224×224), model dimensions (`odom_dim`, `imu_dim`, `act_dim`, `state_dim`, `z_vis_dim`, `z_prop_dim`, `z_dim`), and CEM parameters (`plan_horizon` 15, `plan_iters` 5, `plan_population` 128, `plan_elites` 16, action bounds, `cem_alpha`, `goal_weight`, and `risk_weight`). Training checkpoint metadata supplies the checkpoint-sensitive dimensions and image size; plain state dict checkpoints use the node defaults. These are **node parameters**, not additional launch arguments; change them in the launch file or start the executable directly with a ROS parameter file.

DinoWM loads a different `.pth` checkpoint containing separate encoder and predictor modules, uses `num_hist` frames (which must match training), and transforms its output through TF. Its normalization arrays and CEM settings are node parameters; the shared adapter leaves that model-specific math inside the DinoWM node.

For offline extraction and training from MCAP recordings, see [Data pipeline](data_pipeline.md).
