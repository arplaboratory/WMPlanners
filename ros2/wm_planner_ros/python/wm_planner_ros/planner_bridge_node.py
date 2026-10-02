#!/usr/bin/env python3
"""ROS 2 bridge for world-model latent-state planning."""

from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import rclpy
import rclpy.parameter
from huggingface_hub import hf_hub_download
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
import torch

from world_model.architecture import WorldModel
from world_model.data.normalization import NormStats
from world_model.mpc import (CEMConfig, LatentCEMCostConfig, cem_plan,
                             cem_plan_latent_target, cem_plan_with_cost)

from wm_planner_ros.bridge_adapter import BufferedPlannerBridgeNode, Goal
from wm_planner_ros.model_adapters import StandardWorldModelAdapter
from wm_planner_ros.latent_planning import GoalObservation, positions_from_actions

try:
    from phx_nav_msgs.msg import Trajectory, TrajectoryArray
except ImportError:
    Trajectory = None
    TrajectoryArray = None


@dataclass(frozen=True)
class PlanningAnchor:
    """Fixed model frame, coincident with body at the start of a route."""

    source_frame: str
    position: np.ndarray
    rotation: np.ndarray  # source-frame <- planning-frame


@dataclass
class PlanComputation:
    """Computed trajectory plus diagnostics in the selected output frame."""

    frame_id: str
    stamp: Any
    trajectory_states: np.ndarray
    debug_states: np.ndarray
    risk_roll: np.ndarray
    goal: Goal
    cost: float
    confidence: float
    actions: np.ndarray | None = None
    debug: dict[str, float] | None = None


class PlannerBridgeNode(BufferedPlannerBridgeNode):
    """Encodes ROS observations and publishes a planned trajectory."""

    def __init__(self) -> None:
        super().__init__("wm_planner_bridge")
        self._odom_origin: tuple[float, float, float] | None = None
        self._planning_anchor: PlanningAnchor | None = None
        self._planning_anchor_lock = threading.Lock()
        self._norm_stats: NormStats | None = None
        self._cnn_pool_features = False
        self._warned_invalid_depth = False
        self._goal_obs: GoalObservation | None = None
        self._goal_obs_lock = threading.Lock()
        self._head_trained = False
        self._risk_head_trained = True

        self._declare_params()
        self._load_params()

        self._device = torch.device(self._device_str)
        self._model: WorldModel | None = None  # loaded asynchronously after spin starts
        self._model_ready = False
        self._last_plan: PlanComputation | None = None
        self._last_trajectory_array = None
        self._last_plan_lock = threading.Lock()
        self.configure_adapter(StandardWorldModelAdapter(self), publish_markers=True)
        self._goal_image_sub = self.create_subscription(
            Image, self._goal_image_topic, self._on_goal_image, 1)
        self._capture_goal_srv = self.create_service(
            Trigger, self._capture_goal_service, self._on_capture_goal)
        self._clear_goal_srv = self.create_service(
            Trigger, self._clear_goal_service, self._on_clear_goal)
        self._actions_pub = self.create_publisher(Float32MultiArray, self._actions_topic, 10)
        self._init_timer = self.create_timer(0.0, self._deferred_init)

        self.get_logger().info(
            f"wm_planner_bridge ready. "
            f"backend={self._backend}, dynamics={self._dynamics_type}, "
            f"trajectory_topic={self._trajectory_topic}, "
            f"mrp_action_enabled={self._enable_mrp_action}"
        )

    def _deferred_init(self) -> None:
        """Load model and checkpoint after the executor has started spinning."""
        self._init_timer.cancel()
        self.get_logger().info("Loading world model…")
        try:
            self._adapter.load_model()
            self.get_logger().info(
                f"World model ready. backend={self._backend} dynamics={self._dynamics_type} device={self._device}"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load world model: {exc}")
            raise

    def _declare_params(self) -> None:
        self.declare_parameter("backend", "auto")
        self.declare_parameter("dynamics_type", "auto")
        self.declare_parameter("wm_mode", "standard")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("checkpoint_repo_id", "dmody1/nyu-wm-v3")
        self.declare_parameter("checkpoint_experiment", "dino_rssm_c1")
        self.declare_parameter("checkpoint_filename", "latest.pt")
        self.declare_parameter("device", "gpu")

        self.declare_parameter("rgb_topic", "~/rgb")
        self.declare_parameter("depth_topic", "~/depth")
        self.declare_parameter("odom_topic", "~/odom")
        self.declare_parameter("imu_topic", "~/imu")
        self.declare_parameter("publish_frame", "body")
        self.declare_parameter("trajectory_dt", 0.2)
        self.declare_parameter("goal_frame_mode", "map")
        self.declare_parameter("planning_reset_on_goal", True)
        self.declare_parameter("planning_frame_id", "wm_planner_local")

        self.declare_parameter("odom_dim", 13)
        self.declare_parameter("imu_dim", 10)
        self.declare_parameter("act_dim", 2)
        self.declare_parameter("state_dim", 6)
        self.declare_parameter("z_vis_dim", 128)
        self.declare_parameter("z_prop_dim", 128)
        self.declare_parameter("z_dim", 256)

        self.declare_parameter("image_height", 224)
        self.declare_parameter("image_width", 224)
        self.declare_parameter("invalid_depth_fill", 0.0)

        self.declare_parameter("norm_stats_path", "")
        self.declare_parameter("norm_stats_mode", "auto")
        self.declare_parameter("depth_mode", "sensor")
        self.declare_parameter("imu_layout", "legacy")
        self.declare_parameter("planner_mode", "state")
        self.declare_parameter("trajectory_source", "auto")
        self.declare_parameter("action_position_dims", [0, 1, 2])
        self.declare_parameter("state_head_path", "")
        self.declare_parameter("action_bounds_path", "")
        self.declare_parameter("latent_terminal_weight", 1.0)
        self.declare_parameter("latent_path_weight", 0.1)
        self.declare_parameter("latent_success_threshold", 5.0)
        self.declare_parameter("goal_image_topic", "~/goal_image")
        self.declare_parameter("capture_goal_service", "~/capture_goal")
        self.declare_parameter("clear_goal_service", "~/clear_goal")
        self.declare_parameter("actions_topic", "~/planned_actions")

        self.declare_parameter("plan_horizon", 15)
        self.declare_parameter("plan_iters", 5)
        self.declare_parameter("plan_population", 128)
        self.declare_parameter("plan_elites", 16)
        self.declare_parameter("action_min", -3.0)
        self.declare_parameter("action_max", 3.0)
        self.declare_parameter("action_init_std", 1.0)
        self.declare_parameter("cem_alpha", 0.3)

        self.declare_parameter("goal_weight", 1.0)
        self.declare_parameter("risk_weight", 0.5)
        self.declare_parameter("marker_scale", 0.08)

    def _load_params(self) -> None:
        self._backend = str(self.get_parameter("backend").value).lower()
        self._dynamics_type = str(self.get_parameter("dynamics_type").value).lower()
        self._wm_mode = str(self.get_parameter("wm_mode").value).lower()
        if self._wm_mode not in {"standard", "jepa-lite"}:
            raise ValueError(f"wm_mode must be standard or jepa-lite; got {self._wm_mode}")
        self._backbone_name = "dinov2_vitb14"
        self._checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self._checkpoint_repo_id = str(self.get_parameter("checkpoint_repo_id").value)
        self._checkpoint_experiment = str(self.get_parameter("checkpoint_experiment").value)
        self._checkpoint_filename = str(self.get_parameter("checkpoint_filename").value)
        self._device_str = str(self.get_parameter("device").value)
        self._publish_frame = str(self.get_parameter("publish_frame").value).lower()
        self._goal_frame_mode = str(self.get_parameter("goal_frame_mode").value).lower()
        self._planning_reset_on_goal = bool(self.get_parameter("planning_reset_on_goal").value)
        self._planning_frame_id = str(self.get_parameter("planning_frame_id").value)
        if self._goal_frame_mode not in {"map", "initial_body", "current_body"}:
            raise ValueError("goal_frame_mode must be map, initial_body, or current_body")
        if not self._planning_frame_id:
            raise ValueError("planning_frame_id must not be empty")

        # convention: {backbone}_{dynamics}_{run}  e.g. dino_rssm_c1, cnn_mlp_c1
        if self._backend == "auto" or self._dynamics_type == "auto":
            exp = self._checkpoint_experiment.lower()
            detected_backend = "dinov2" if exp.startswith("dino") else "cnn"
            detected_dynamics = "rssm" if "rssm" in exp else "mlp"
            if self._backend == "auto":
                self._backend = detected_backend
            if self._dynamics_type == "auto":
                self._dynamics_type = detected_dynamics
            self.get_logger().info(
                f"Derived from experiment '{self._checkpoint_experiment}': "
                f"backend={self._backend} dynamics_type={self._dynamics_type}"
            )

            self.set_parameters([
                rclpy.parameter.Parameter("backend",
                                          rclpy.parameter.Parameter.Type.STRING,
                                          self._backend),
                rclpy.parameter.Parameter("dynamics_type",
                                          rclpy.parameter.Parameter.Type.STRING,
                                          self._dynamics_type),
            ])

        if self._backend not in {"cnn", "dinov2", "vjepa2"}:
            raise ValueError(f"backend must be one of ['cnn','dinov2','vjepa2']; got {self._backend}")
        if self._dynamics_type not in {"rssm", "mlp"}:
            raise ValueError(f"dynamics_type must be one of ['rssm','mlp']; got {self._dynamics_type}")

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._trajectory_dt = float(self.get_parameter("trajectory_dt").value)

        self._norm_stats_path = str(self.get_parameter("norm_stats_path").value)
        self._norm_stats_mode = str(self.get_parameter("norm_stats_mode").value).lower()
        self._depth_mode = str(self.get_parameter("depth_mode").value).lower()
        self._imu_layout = str(self.get_parameter("imu_layout").value).lower()
        self._planner_mode_param = str(self.get_parameter("planner_mode").value).lower()
        self._planner_mode = self._planner_mode_param
        self._trajectory_source_param = str(self.get_parameter("trajectory_source").value).lower()
        self._trajectory_source = self._trajectory_source_param
        self._action_position_dims = tuple(int(v) for v in self.get_parameter("action_position_dims").value)
        self._state_head_path = str(self.get_parameter("state_head_path").value)
        self._action_bounds_path = str(self.get_parameter("action_bounds_path").value)
        self._latent_terminal_weight = float(self.get_parameter("latent_terminal_weight").value)
        self._latent_path_weight = float(self.get_parameter("latent_path_weight").value)
        self._latent_success_threshold = float(self.get_parameter("latent_success_threshold").value)
        self._goal_image_topic = str(self.get_parameter("goal_image_topic").value)
        self._capture_goal_service = str(self.get_parameter("capture_goal_service").value)
        self._clear_goal_service = str(self.get_parameter("clear_goal_service").value)
        self._actions_topic = str(self.get_parameter("actions_topic").value)
        for name, value, allowed in (
            ("norm_stats_mode", self._norm_stats_mode, {"auto", "on", "off"}),
            ("depth_mode", self._depth_mode, {"sensor", "zeros"}),
            ("imu_layout", self._imu_layout, {"legacy", "dataset"}),
            ("planner_mode", self._planner_mode_param, {"auto", "state", "latent"}),
            ("trajectory_source", self._trajectory_source_param, {"auto", "state_head", "actions"}),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}")
        if len(self._action_position_dims) != 3 or len(set(self._action_position_dims)) != 3:
            raise ValueError("action_position_dims must contain three distinct dimensions")
        self._latent_cost_cfg = LatentCEMCostConfig(
            terminal_weight=self._latent_terminal_weight,
            path_weight=self._latent_path_weight,
        )

        self._odom_dim = int(self.get_parameter("odom_dim").value)
        self._imu_dim = int(self.get_parameter("imu_dim").value)
        self._act_dim = int(self.get_parameter("act_dim").value)
        self._state_dim = int(self.get_parameter("state_dim").value)
        self._z_vis_dim = int(self.get_parameter("z_vis_dim").value)
        self._z_prop_dim = int(self.get_parameter("z_prop_dim").value)
        self._z_dim = int(self.get_parameter("z_dim").value)

        self._img_h = int(self.get_parameter("image_height").value)
        self._img_w = int(self.get_parameter("image_width").value)
        self._invalid_depth_fill = float(self.get_parameter("invalid_depth_fill").value)
        if not np.isfinite(self._invalid_depth_fill):
            raise ValueError("invalid_depth_fill must be finite")

        self._goal_weight = float(self.get_parameter("goal_weight").value)
        self._risk_weight = float(self.get_parameter("risk_weight").value)
        self._marker_scale = float(self.get_parameter("marker_scale").value)

        self._cem_cfg = CEMConfig(
            horizon=int(self.get_parameter("plan_horizon").value),
            iters=int(self.get_parameter("plan_iters").value),
            population=int(self.get_parameter("plan_population").value),
            elites=int(self.get_parameter("plan_elites").value),
            action_dim=self._act_dim,
            action_min=float(self.get_parameter("action_min").value),
            action_max=float(self.get_parameter("action_max").value),
            init_std=float(self.get_parameter("action_init_std").value),
            alpha=float(self.get_parameter("cem_alpha").value),
        )

    def _build_model(self) -> WorldModel:
        bkw: dict = {"device": str(self._device)}
        if self._backend in {"dinov2", "vjepa2"}:
            # Weights are loaded from the checkpoint; skip the pretrained internet download.
            bkw["pretrained"] = False
        if self._backend == "dinov2":
            bkw["model_name"] = self._backbone_name
        if self._backend == "cnn":
            bkw["pool_features"] = self._cnn_pool_features
        return WorldModel(
            odom_dim=self._odom_dim,
            imu_dim=self._imu_dim,
            act_dim=self._act_dim,
            z_vis_dim=self._z_vis_dim,
            z_prop_dim=self._z_prop_dim,
            z_dim=self._z_dim,
            state_dim=self._state_dim,
            backbone=self._backend if self._backend != "cnn" else None,
            backbone_kwargs=bkw,
            dynamics_type=self._dynamics_type,
            image_size=self._img_h,
        ).to(self._device)

    def _load_checkpoint(self) -> None:
        checkpoint_path = self._checkpoint_path
        if self._wm_mode == "jepa-lite" and not checkpoint_path:
            raise ValueError("JEPA-lite requires checkpoint_path")
        if not checkpoint_path:
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self._checkpoint_repo_id,
                    filename=f"{self._checkpoint_experiment}/{self._checkpoint_filename}",
                )
                self.get_logger().info(
                    f"Downloaded checkpoint from Hugging Face repo={self._checkpoint_repo_id} "
                    f"experiment={self._checkpoint_experiment} file={self._checkpoint_filename}."
                )
            except Exception as exc:
                self.get_logger().warning(
                    "checkpoint_path is empty and Hugging Face download failed; "
                    f"planner bridge will run with random weights. error={exc}"
                )
                self._model = self._build_model()
                return

        if os.path.isdir(checkpoint_path):
            checkpoint_path = next((os.path.join(checkpoint_path, name) for name in
                                    ("latest_with_rollout_heads.pt", "latest_with_observed_heads.pt",
                                     "latest_with_state_head.pt", "latest_with_heads.pt", "latest.pt")
                                    if os.path.isfile(os.path.join(checkpoint_path, name))), None)
            if checkpoint_path is None:
                raise FileNotFoundError("No supported checkpoint in run directory")
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        training_metadata = False
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            args_dict = checkpoint.get("args", {})
            model_cfg = checkpoint.get("model_config", {})
            if not args_dict or not model_cfg:
                raise ValueError("Training checkpoint lacks args or model_config metadata")
            training_metadata = True
            trained_mode = args_dict.get("wm_mode", "standard")
            if trained_mode != self._wm_mode:
                raise ValueError(
                    f"Checkpoint wm_mode={trained_mode} does not match launch wm_mode={self._wm_mode}"
                )
            trained_backend = args_dict.get("backbone", "cnn").lower()
            trained_dynamics = args_dict.get("dynamics_type", "mlp").lower()
            if trained_dynamics == "td-jepa":
                trained_dynamics = "mlp"
            if self._wm_mode == "standard" and (
                trained_backend != self._backend or trained_dynamics != self._dynamics_type
            ):
                raise ValueError(
                    f"Checkpoint uses {trained_backend}+{trained_dynamics}, but launch selected "
                    f"{self._backend}+{self._dynamics_type}"
                )
            self._backend = trained_backend
            self._dynamics_type = trained_dynamics
            self._backbone_name = args_dict.get("backbone_name", "dinov2_vitb14")
            self._img_h = self._img_w = int(args_dict.get("img_size", 224))
            self._odom_dim = int(model_cfg["odom_dim"])
            self._imu_dim = int(model_cfg["imu_dim"])
            self._act_dim = int(model_cfg["act_dim"])
            self._state_dim = int(model_cfg["state_dim"])
            if self._backend == "cnn":
                state = checkpoint["model"]
                projection = state.get("vision.proj.weight")
                last_conv = state.get("vision.net.4.weight")
                self._cnn_pool_features = (
                    projection is not None and last_conv is not None
                    and projection.shape[1] == last_conv.shape[0]
                )
            self._cem_cfg.action_dim = self._act_dim
            if self._wm_mode == "jepa-lite" and self._cem_cfg.horizon == 0:
                self._cem_cfg.horizon = int(args_dict.get("horizon", 8))
            self.set_parameters([
                rclpy.parameter.Parameter("backend", value=self._backend),
                rclpy.parameter.Parameter("dynamics_type", value=self._dynamics_type),
                rclpy.parameter.Parameter("image_height", value=self._img_h),
                rclpy.parameter.Parameter("image_width", value=self._img_w),
                rclpy.parameter.Parameter("odom_dim", value=self._odom_dim),
                rclpy.parameter.Parameter("imu_dim", value=self._imu_dim),
                rclpy.parameter.Parameter("act_dim", value=self._act_dim),
                rclpy.parameter.Parameter("state_dim", value=self._state_dim),
            ])
            self._model = self._build_model()
            self._model.load_state_dict(checkpoint["model"], strict=True)
            self.get_logger().info(
                f"Loaded training checkpoint {checkpoint_path}: mode={trained_mode}, "
                f"backbone={self._backend}, dynamics={self._dynamics_type}, "
                f"image_size={self._img_h}, action_dim={self._act_dim}"
            )
        else:
            if self._wm_mode == "jepa-lite":
                raise ValueError("JEPA-lite requires a training checkpoint with args and model_config")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self._model = self._build_model()
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            self.get_logger().info(
                f"Loaded checkpoint {checkpoint_path}. missing={len(missing)} unexpected={len(unexpected)}"
            )

        head_training = checkpoint.get("head_training") or {} if isinstance(checkpoint, dict) else {}
        head_path = self._state_head_path
        if not head_path and self._wm_mode == "jepa-lite" and not head_training:
            sibling = os.path.join(os.path.dirname(checkpoint_path), "state_head_decoder.pt")
            if os.path.isfile(sibling):
                head_path = sibling
        if head_path:
            head_state = torch.load(head_path, map_location=self._device, weights_only=False)
            self._model.state_head.load_state_dict(head_state)
        self._head_trained = (self._wm_mode == "standard" or
                              bool(head_training.get("state_head")) or
                              bool(head_path) or
                              "with_state_head" in os.path.basename(checkpoint_path))
        self._risk_head_trained = self._wm_mode == "standard" or bool(head_training.get("risk_head"))
        if not self._risk_head_trained:
            self._risk_weight = 0.0
        self._planner_mode = ("latent" if self._wm_mode == "jepa-lite" and not self._head_trained
                              else "state") if self._planner_mode_param == "auto" else self._planner_mode_param
        if self._planner_mode == "state" and not self._head_trained:
            raise ValueError("State planning requires a fitted state head; use planner_mode=latent")
        self._trajectory_source = ("state_head" if self._head_trained else "actions") \
            if self._trajectory_source_param == "auto" else self._trajectory_source_param
        if self._trajectory_source == "state_head" and not self._head_trained:
            raise ValueError("trajectory_source=state_head requires a fitted state head")
        if self._trajectory_source == "actions" and (
            self._action_position_dims[0] < 0 or
            max(self._action_position_dims) >= self._act_dim
        ):
            raise ValueError("action_position_dims are invalid for this checkpoint's action dimension")
        if self._planner_mode == "latent":
            bounds_path = self._action_bounds_path or os.path.join(
                os.path.dirname(checkpoint_path), "action_bounds.pt")
            if os.path.isfile(bounds_path):
                bounds = torch.load(bounds_path, map_location="cpu", weights_only=False)
                for name, key in (("action_low", "low"), ("action_high", "high"),
                                  ("action_init_mean", "mean"), ("action_init_std", "std")):
                    values = torch.as_tensor(bounds[key], dtype=torch.float32)
                    if values.numel() != self._act_dim:
                        raise ValueError(f"{bounds_path}: {key} has wrong action dimension")
                    setattr(self._cem_cfg, name, values.clamp_min(1e-3) if key == "std" else values)
            else:
                self.get_logger().warning(
                    f"No per-dimension action bounds at {bounds_path}; using scalar CEM bounds")
        self.get_logger().info(
            f"planner_mode={self._planner_mode} trajectory_source={self._trajectory_source} "
            f"state_head_trained={self._head_trained}")

        # Load NormStats for correct input normalization and output scaling.
        ns_path = self._norm_stats_path
        if not ns_path:
            ns_path = os.path.join(os.path.dirname(checkpoint_path), "norm_stats.pt")
        use_norm_stats = (self._norm_stats_mode == "on" or
                          (self._norm_stats_mode == "auto" and
                           (self._wm_mode == "standard" or not training_metadata or
                            (self._planner_mode == "state" and bool(head_training)))))
        if use_norm_stats and os.path.exists(ns_path):
            self._norm_stats = NormStats.load(ns_path)
            self.get_logger().info(f"Loaded NormStats from {ns_path}")
        elif use_norm_stats:
            self.get_logger().warning(
                f"norm_stats.pt not found at {ns_path}. "
                "Inputs will NOT be normalized — trajectory quality will be poor. "
                "Set the 'norm_stats_path' parameter or place norm_stats.pt beside the checkpoint."
            )

    def _on_goal(self, msg: PoseStamped) -> None:
        """Record a global goal and optionally start a new fixed planning frame."""
        super()._on_goal(msg)
        if self._planning_reset_on_goal:
            with self._planning_anchor_lock:
                self._planning_anchor = None
            self._odom_origin = None
            self.get_logger().info("Cleared WM planning-frame anchor for new goal")

    def _on_capture_goal(self, request: Any, response: Any) -> Any:
        """Store the latest synchronized observation as the latent target."""
        with self._latest_observation_lock:
            latest = self._latest_observation
        if latest is None:
            response.success = False
            response.message = "No synchronized observation is available"
            return response
        rgb, depth, odom, imu = self._adapter.prepare(*latest)
        with self._goal_obs_lock:
            self._goal_obs = GoalObservation(rgb, depth, odom, imu)
        response.success = True
        response.message = "Captured goal observation"
        return response

    def _on_goal_image(self, image: Image) -> None:
        """Pair a goal image with the most recent synchronized proprioception."""
        with self._latest_observation_lock:
            latest = self._latest_observation
        if latest is None:
            self.get_logger().warning("Ignoring goal image until sensors are synchronized")
            return
        _, _, odom, imu = self._adapter.prepare(*latest)
        goal_odom = copy.deepcopy(odom)
        goal_odom.twist.twist.linear.x = 0.0
        goal_odom.twist.twist.linear.y = 0.0
        goal_odom.twist.twist.linear.z = 0.0
        goal_odom.twist.twist.angular.x = 0.0
        goal_odom.twist.twist.angular.y = 0.0
        goal_odom.twist.twist.angular.z = 0.0
        with self._goal_obs_lock:
            self._goal_obs = GoalObservation(image, None, goal_odom, imu)

    def _on_clear_goal(self, request: Any, response: Any) -> Any:
        with self._goal_obs_lock:
            self._goal_obs = None
        response.success = True
        response.message = "Cleared goal observation"
        return response

    def _model_anchor(self, odom: Odometry) -> PlanningAnchor | None:
        """Return the frame in which this checkpoint's window-relative state lives."""
        if self._goal_frame_mode == "map":
            return None
        position = np.array((odom.pose.pose.position.x, odom.pose.pose.position.y,
                             odom.pose.pose.position.z), dtype=np.float64)
        q = odom.pose.pose.orientation
        rotation = self._quat_to_rot(q.x, q.y, q.z, q.w).astype(np.float64)
        if self._goal_frame_mode == "current_body":
            return PlanningAnchor(odom.header.frame_id, position, rotation)
        with self._planning_anchor_lock:
            if (self._planning_anchor is None or
                    self._planning_anchor.source_frame != odom.header.frame_id):
                self._planning_anchor = PlanningAnchor(odom.header.frame_id, position, rotation)
                self._odom_origin = tuple(position)
                self.get_logger().info(
                    f"Anchored {self._planning_frame_id} to {odom.header.frame_id} at route start")
            return self._planning_anchor

    def _point_to_model_frame(self, point: np.ndarray, odom: Odometry,
                              anchor: PlanningAnchor | None) -> np.ndarray:
        if anchor is None:
            return point
        return anchor.rotation.T @ (point - anchor.position)

    def _point_from_model_frame(self, point: np.ndarray, anchor: PlanningAnchor | None) -> np.ndarray:
        if anchor is None:
            return point
        return anchor.rotation @ point + anchor.position

    @staticmethod
    def _rot_to_quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
        """Convert a proper 3x3 rotation matrix to ROS xyzw quaternion order."""
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            return ((rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale)
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            return (0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale)
        if index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            return ((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale)
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        return ((rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale)

    def _odom_in_model_frame(self, odom: Odometry, anchor: PlanningAnchor | None) -> Odometry:
        """Express pose, attitude, and velocity in the fixed model frame."""
        if anchor is None:
            return odom
        result = copy.deepcopy(odom)
        pose = result.pose.pose
        source_rotation = self._quat_to_rot(pose.orientation.x, pose.orientation.y,
                                            pose.orientation.z, pose.orientation.w)
        position = self._point_to_model_frame(
            np.array((pose.position.x, pose.position.y, pose.position.z), dtype=np.float64),
            odom, anchor)
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = self._rot_to_quat(
            anchor.rotation.T @ source_rotation)
        linear = anchor.rotation.T @ np.array((result.twist.twist.linear.x,
                                                result.twist.twist.linear.y,
                                                result.twist.twist.linear.z))
        angular = anchor.rotation.T @ np.array((result.twist.twist.angular.x,
                                                 result.twist.twist.angular.y,
                                                 result.twist.twist.angular.z))
        result.twist.twist.linear.x, result.twist.twist.linear.y, result.twist.twist.linear.z = linear
        result.twist.twist.angular.x, result.twist.twist.angular.y, result.twist.twist.angular.z = angular
        result.header.frame_id = self._planning_frame_id
        return result

    def _imu_in_model_frame(self, imu: Imu, anchor: PlanningAnchor | None) -> Imu:
        if anchor is None:
            return imu
        result = copy.deepcopy(imu)
        q = result.orientation
        source_rotation = self._quat_to_rot(q.x, q.y, q.z, q.w)
        q.x, q.y, q.z, q.w = self._rot_to_quat(anchor.rotation.T @ source_rotation)
        for vector in (result.angular_velocity, result.linear_acceleration):
            values = anchor.rotation.T @ np.array((vector.x, vector.y, vector.z))
            vector.x, vector.y, vector.z = values
        return result

    def _goal_in_odom_frame(self, goal: Goal, odom: Odometry) -> np.ndarray:
        """Transform a waypoint from its stamped frame into the odometry frame."""
        source_frame = goal.frame_id or odom.header.frame_id
        target_frame = odom.header.frame_id
        if not target_frame:
            raise ValueError("Observation odometry has no frame")
        point = np.array((goal.x, goal.y, goal.z), dtype=np.float64)
        if source_frame == target_frame:
            return point
        try:
            stamp = Time.from_msg(goal.stamp) if goal.stamp is not None else Time()
            transform = self._tf_buffer.lookup_transform(target_frame, source_frame, stamp)
        except Exception as exc:
            raise ValueError(
                f"Unable to transform waypoint from {source_frame!r} to {target_frame!r}: {exc}"
            ) from exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = self._quat_to_rot(rotation.x, rotation.y, rotation.z, rotation.w)
        return matrix @ point + np.array((translation.x, translation.y, translation.z))

    def _plan_latent_from_observation(
            self, rgb: Image, depth: Image | None, odom: Odometry, imu: Imu,
            goal: Goal | None, *, output_in_odom: bool = False,
            should_cancel: Callable[[], bool] | None = None,
        ) -> PlanComputation:
        with self._goal_obs_lock:
            goal_obs = self._goal_obs
        if goal_obs is None:
            raise ValueError("Latent planning requires ~/capture_goal or ~/goal_image")
        if not odom.header.frame_id or goal_obs.odom.header.frame_id != odom.header.frame_id:
            raise ValueError("Goal observation and current odometry must share a frame")
        anchor = self._model_anchor(odom)
        current_global = np.array((odom.pose.pose.position.x, odom.pose.pose.position.y,
                                   odom.pose.pose.position.z), dtype=np.float64)
        current = self._point_to_model_frame(current_global, odom, anchor)
        captured = goal_obs.odom.pose.pose.position
        goal_global = (self._goal_in_odom_frame(goal, odom) if goal is not None else
                       np.array((captured.x, captured.y, captured.z), dtype=np.float64))
        goal_xyz = self._point_to_model_frame(goal_global, odom, anchor)

        odom_model = self._odom_in_model_frame(odom, anchor)
        imu_model = self._imu_in_model_frame(imu, anchor)
        goal_odom_model = self._odom_in_model_frame(goal_obs.odom, anchor)
        goal_imu_model = self._imu_in_model_frame(goal_obs.imu, anchor)
        with torch.no_grad():
            rgb_t = self._image_to_tensor(rgb, 3)
            depth_t = (self._image_to_tensor(depth, 1) if depth is not None and
                       self._depth_mode == "sensor" else
                       torch.zeros(1, 1, self._img_h, self._img_w, device=self._device))
            odom_t = self._odom_to_tensor(odom_model, origin=np.zeros(3, dtype=np.float64))
            imu_t = self._imu_to_tensor(imu_model)
            goal_rgb = self._image_to_tensor(goal_obs.rgb, 3)
            goal_depth = (self._image_to_tensor(goal_obs.depth, 1) if goal_obs.depth is not None and
                          self._depth_mode == "sensor" else
                          torch.zeros_like(depth_t))
            goal_orientation = goal.orientation if goal is not None else None
            goal_odom = self._odom_to_tensor(
                goal_odom_model, origin=np.zeros(3, dtype=np.float64),
                position_override=goal_xyz, orientation_override=goal_orientation)
            goal_imu = self._imu_to_tensor(
                goal_imu_model, orientation_override=goal_orientation)
            z0 = self._model.encode(rgb_t, depth_t, odom_t, imu_t).z
            z_goal = self._model.encode(goal_rgb, goal_depth, goal_odom, goal_imu).z
            _, best_seq, cost = cem_plan_latent_target(
                self._model, z0, z_goal, self._cem_cfg, self._device,
                cost_cfg=self._latent_cost_cfg, should_cancel=should_cancel)
            z_roll = self._model.rollout(z0, best_seq.unsqueeze(0))
            actions = best_seq.cpu().numpy()
            if self._trajectory_source == "actions":
                positions = positions_from_actions(actions, self._action_position_dims, current)
            else:
                state_t0 = self._model.state_head(z0)[0]
                decoded = self._model.state_head(z_roll).squeeze(0)
                positions = (decoded[:, :3] - state_t0[:3]).cpu().numpy()
                if self._norm_stats is not None:
                    positions *= self._norm_stats.odom_std[:3].numpy()
                positions = positions + current[np.newaxis, :]
            positions_global = np.array([self._point_from_model_frame(p, anchor) for p in positions])
            risk_roll = (self._model.risk_head(z_roll).squeeze(0).squeeze(-1).cpu().numpy()
                         if self._risk_head_trained else np.zeros(z_roll.shape[1]))
            terminal = float(torch.linalg.norm(z_roll[0, -1] - z_goal[0]))
            path = float(torch.linalg.norm(z_roll[0, 1:] - z_goal, dim=-1).mean())
            now = float(torch.linalg.norm(z0[0] - z_goal[0]))
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite latent planner trajectory")
        output = positions.copy()
        frame = odom.header.frame_id
        if self._publish_frame == "body" and not output_in_odom:
            q = odom.pose.pose.orientation
            rotation = self._quat_to_rot(q.x, q.y, q.z, q.w)
            output = (rotation.T @ (positions_global - current_global[np.newaxis, :]).T).T
            frame = "body"
        if self._publish_frame != "body" or output_in_odom:
            output = positions_global
        debug = {"latent_terminal_dist": terminal, "latent_path_dist": path,
                 "latent_dist_now": now, "best_cost": float(cost),
                 "goal_reached": float(now <= self._latent_success_threshold),
                 "traj_end_goal_dist": float(np.linalg.norm(positions[-1] - goal_xyz))}
        return PlanComputation(frame, odom.header.stamp, output.astype(np.float32),
                               positions_global.astype(np.float32), risk_roll,
                               Goal(*goal_global), float(cost),
                               1.0 / (1.0 + max(terminal, 0.0)), actions, debug)

    def _plan_from_observation(
            self,
            rgb: Image,
            depth: Image,
            odom: Odometry,
            imu: Imu,
            goal: Goal | None,
            *,
            output_in_odom: bool = False,
            should_cancel: Callable[[], bool] | None = None,
        ) -> PlanComputation:
        if self._planner_mode == "latent":
            return self._plan_latent_from_observation(
                rgb, depth, odom, imu, goal, output_in_odom=output_in_odom,
                should_cancel=should_cancel)
        if goal is None:
            goal = self._default_goal
        anchor = self._model_anchor(odom)
        odom_model = self._odom_in_model_frame(odom, anchor)
        imu_model = self._imu_in_model_frame(imu, anchor)
        current_model = np.array((odom_model.pose.pose.position.x, odom_model.pose.pose.position.y,
                                  odom_model.pose.pose.position.z), dtype=np.float64)
        goal_model = self._point_to_model_frame(self._goal_in_odom_frame(goal, odom), odom, anchor)
        rgb_t = self._image_to_tensor(rgb, channels=3)
        depth_t = (self._image_to_tensor(depth, channels=1) if depth is not None else
                   torch.zeros(1, 1, self._img_h, self._img_w, device=self._device))
        odom_t = self._odom_to_tensor(
            odom_model,
            origin=np.zeros(3, dtype=np.float64) if anchor is not None else None,
        )
        imu_t = self._imu_to_tensor(imu_model)
        for name, tensor in (("RGB", rgb_t), ("depth", depth_t),
                             ("odometry", odom_t), ("IMU", imu_t)):
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Nonfinite {name} observation on the WM bridge")

        with torch.no_grad():
            wm_out = self._model.encode(rgb_t, depth_t, odom_t, imu_t)
            z0 = wm_out.z
            if not torch.isfinite(z0).all().item():
                raise ValueError("Nonfinite initial world-model latent from finite observations")

            relative_goal = goal_model - current_model
            if self._norm_stats is not None:
                ns = self._norm_stats
                gx = relative_goal[0] / float(ns.odom_std[0])
                gy = relative_goal[1] / float(ns.odom_std[1])
                gz = relative_goal[2] / float(ns.odom_std[2])
            else:
                gx, gy, gz = relative_goal
            goal_t = torch.tensor([[gx, gy, gz]], dtype=torch.float32, device=self._device)

            true_t0_norm = odom_t[0, :self._state_dim]
            decoded_t0 = self._model.state_head(z0)[0]
            anchor_norm = (decoded_t0 - true_t0_norm).cpu()

            if self._risk_head_trained:
                _, best_seq = cem_plan(
                    model=self._model, z0=z0, goal=goal_t,
                    cost_fn=self._cost_fn, cfg=self._cem_cfg,
                    device=self._device, should_cancel=should_cancel)
            else:
                _, best_seq, _ = cem_plan_with_cost(
                    model=self._model, z0=z0, goal=goal_t,
                    cost_fn=self._cost_fn, cfg=self._cem_cfg,
                    device=self._device, use_risk_head=False,
                    should_cancel=should_cancel)
            z_roll = self._model.rollout(z0, best_seq.unsqueeze(0))
            if not torch.isfinite(z_roll).all().item():
                raise ValueError("Nonfinite world-model latent during planned rollout")
            state_roll = self._model.state_head(z_roll).squeeze(0).cpu().numpy()
            state_roll = state_roll - anchor_norm.numpy()[np.newaxis, :]
            if not np.isfinite(state_roll).all():
                raise ValueError("Nonfinite decoded world-model state during planned rollout")
            risk_roll = (self._model.risk_head(z_roll).squeeze(0).squeeze(-1).cpu().numpy()
                         if self._risk_head_trained else np.zeros(z_roll.shape[1]))

        in_frame = odom.header.frame_id
        if not in_frame:
            raise ValueError("Observation odometry has no frame")
        in_frame_norm = str(in_frame)

        if self._norm_stats is not None:
            ns = self._norm_stats
            pos_std = ns.odom_std[:3].numpy()
            pos_mean = ns.odom_mean[:3].numpy()
            state_roll = state_roll.copy()
            state_roll[:, :3] = state_roll[:, :3] * pos_std + pos_mean

        # State-head positions are in the fixed model frame. Convert them back
        # to the odometry/map frame only at the ROS boundary. Map mode retains
        # the legacy translation-only window normalization.
        state_roll = state_roll.copy()
        if anchor is None:
            state_roll[:, :3] += np.asarray(self._odom_origin or (0.0, 0.0, 0.0))
        else:
            state_roll[:, :3] = np.array([
                self._point_from_model_frame(point, anchor) for point in state_roll[:, :3]
            ])
        state_to_pub = state_roll.copy()
        out_frame = in_frame_norm

        if self._publish_frame == "body" and not output_in_odom:
            out_frame = self._publish_frame
            qx = odom.pose.pose.orientation.x
            qy = odom.pose.pose.orientation.y
            qz = odom.pose.pose.orientation.z
            qw = odom.pose.pose.orientation.w
            rotation = self._quat_to_rot(qx, qy, qz, qw)

            cur_pos = np.array([
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
            ], dtype=np.float32)
            state_to_pub = state_roll.copy()
            state_to_pub[:, :3] = (rotation.T @ (state_roll[:, :3] - cur_pos[np.newaxis, :]).T).T

        final_xyz = state_roll[-1, :3]
        goal_xyz = self._goal_in_odom_frame(goal, odom).astype(np.float32)
        goal_distance = float(np.linalg.norm(final_xyz - goal_xyz))
        risk_mean = float(np.mean(risk_roll))
        cost = self._goal_weight * goal_distance + self._risk_weight * risk_mean
        confidence = 1.0 / (1.0 + max(cost, 0.0))

        return PlanComputation(
            frame_id=out_frame,
            stamp=odom.header.stamp,
            trajectory_states=state_to_pub,
            debug_states=state_roll,
            risk_roll=risk_roll,
            goal=goal,
            cost=float(cost),
            confidence=float(confidence),
        )

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        x, y, z, w = qx, qy, qz, qw
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ], dtype=np.float32)

    def _publish_plan(self, plan: PlanComputation) -> None:
        self._publish_path(plan.frame_id, plan.stamp, plan.trajectory_states)
        self._publish_markers(plan.frame_id, plan.stamp, plan.trajectory_states, plan.risk_roll)
        if plan.debug is None:
            self._publish_debug(plan.goal, plan.debug_states, plan.risk_roll)
        else:
            metrics = Float32MultiArray()
            metrics.data = [float(plan.debug[key]) for key in (
                "latent_terminal_dist", "latent_path_dist", "latent_dist_now",
                "best_cost", "goal_reached", "traj_end_goal_dist")]
            self._debug_pub.publish(metrics)
        if plan.actions is not None:
            actions_msg = Float32MultiArray()
            actions_msg.data = plan.actions.reshape(-1).astype(float).tolist()
            self._actions_pub.publish(actions_msg)

        if TrajectoryArray is not None:
            trajectory_array = self._plan_to_trajectory_array(plan)
            with self._last_plan_lock:
                self._last_plan = plan
                self._last_trajectory_array = trajectory_array

    def _plan_to_trajectory_array(self, plan: PlanComputation) -> Any:
        trajectory_array = TrajectoryArray()
        trajectory = Trajectory()
        base_time = Time.from_msg(plan.stamp)

        for idx, step in enumerate(plan.trajectory_states):
            odom = Odometry()
            odom.header.frame_id = plan.frame_id
            odom.header.stamp = (base_time + RclpyDuration(seconds=self._trajectory_dt * idx)).to_msg()
            odom.pose.pose.position.x = float(step[0])
            odom.pose.pose.position.y = float(step[1])
            odom.pose.pose.position.z = float(step[2])
            odom.pose.pose.orientation.w = 1.0

            if idx > 0 and self._trajectory_dt > 0.0:
                prev = plan.trajectory_states[idx - 1]
                odom.twist.twist.linear.x = float((step[0] - prev[0]) / self._trajectory_dt)
                odom.twist.twist.linear.y = float((step[1] - prev[1]) / self._trajectory_dt)
                odom.twist.twist.linear.z = float((step[2] - prev[2]) / self._trajectory_dt)

            trajectory.trajectory.append(odom)

        trajectory.cost.data = float(plan.cost)
        trajectory_array.trajectories.append(trajectory)
        return trajectory_array

    def _cost_fn(self,
                 state_seq: torch.Tensor,
                 risk_seq: torch.Tensor,
                 goal: torch.Tensor
        ) -> torch.Tensor:
        # delta state cost: minimize ‖(state_T - state_0)1 desired_change‖.
        # Taking the difference cancels any constant decoder bias so CEM plans
        # to the right relative position regardless of state_head offset.
        # state_seq: (N, H+1, state_dim)
        change    = state_seq[:, -1, :3] - state_seq[:, 0, :3]   # (N, 3)
        goal_dist = torch.linalg.norm(change - goal[:, :3], dim=1)
        risk_cost = risk_seq.mean(dim=1) if risk_seq is not None else 0.0
        return self._goal_weight * goal_dist + self._risk_weight * risk_cost

    def _publish_path(self, frame_id: str, stamp: Any, state_roll: np.ndarray) -> None:
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        for step in state_roll:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(step[0])
            pose.pose.position.y = float(step[1])
            pose.pose.position.z = float(step[2])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._path_pub.publish(msg)

    def _publish_markers(self,
                         frame_id: str,
                         stamp: Any,
                         state_roll: np.ndarray,
                         risk_roll: np.ndarray
        ) -> None:
        """Publish the planned trajectory as risk-colored spheres (green=safe, red=risky)."""
        array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = stamp
        delete_all.ns = "wm_plan"
        delete_all.action = Marker.DELETEALL
        array.markers.append(delete_all)

        risk_min = float(risk_roll.min())
        risk_max = float(risk_roll.max())
        risk_range = max(risk_max - risk_min, 1e-6)

        for i, (step, risk) in enumerate(zip(state_roll, risk_roll)):
            t = float(risk - risk_min) / risk_range  # 0=safe, 1=risky
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "wm_plan"
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(step[0])
            m.pose.position.y = float(step[1])
            m.pose.position.z = float(step[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self._marker_scale
            m.color.a = 1.0
            m.color.r = t
            m.color.g = 1.0 - t
            m.color.b = 0.0
            array.markers.append(m)

        self._marker_pub.publish(array)

    def _publish_debug(self, goal: Goal, state_roll: np.ndarray, risk_roll: np.ndarray) -> None:
        final_xyz = state_roll[-1, :3]
        goal_distance = float(np.linalg.norm(final_xyz - np.array([goal.x, goal.y, goal.z], dtype=np.float32)))
        risk_mean = float(np.mean(risk_roll))
        metrics = Float32MultiArray()
        metrics.data = [goal_distance, risk_mean, float(self._goal_weight), float(self._risk_weight)]
        self._debug_pub.publish(metrics)

    # ImageNet statistics used by pretrained ViT backbones (DINOv2, V-JEPA2).
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _image_to_tensor(self, msg: Image, channels: int) -> torch.Tensor:
        encoding = msg.encoding.lower()
        if channels == 1:
            dtypes = {"16uc1": np.uint16, "mono16": np.uint16, "32fc1": np.float32}
            if encoding not in dtypes:
                raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
            dtype = np.dtype(dtypes[encoding]).newbyteorder(">" if msg.is_bigendian else "<")
        else:
            if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
                raise ValueError(f"Unsupported color encoding: {msg.encoding}")
            dtype = np.dtype(np.uint8)
        row_width = msg.step // dtype.itemsize
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_width)
        arr = arr[:, : msg.width * (1 if channels == 1 else (4 if "a8" in encoding else 3))]
        arr = arr.reshape(msg.height, msg.width, -1)
        if encoding in {"bgr8", "bgra8"}:
            arr = arr[:, :, [2, 1, 0]]
        else:
            arr = arr[:, :, :channels]
        if channels == 1:
            arr = arr.astype(np.float32)
            # ROS integer depth images use millimeters; 32FC1 uses meters.
            if encoding in {"16uc1", "mono16"}:
                arr = arr / 1000.0
            invalid = ~np.isfinite(arr)
            if invalid.any():
                if not self._warned_invalid_depth:
                    self.get_logger().warning(
                        f"Depth image contains {int(invalid.sum())}/{arr.size} "
                        f"({100.0 * invalid.mean():.1f}%) nonfinite pixels; "
                        f"replacing them with invalid_depth_fill={self._invalid_depth_fill}"
                    )
                    self._warned_invalid_depth = True
                arr[invalid] = self._invalid_depth_fill
        else:
            arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        ten = torch.from_numpy(arr).unsqueeze(0)
        resize_kwargs = (
            {"align_corners": False, "antialias": True}
            if channels == 3 and self._planner_mode == "latent" else {}
        )
        ten = torch.nn.functional.interpolate(
            ten,
            size=(self._img_h, self._img_w),
            mode="bilinear" if channels == 3 else "nearest",
            **resize_kwargs,
        ).to(self._device)
        # Pretrained ViT backbones (DINOv2, V-JEPA2) require ImageNet normalization.
        # The CNN backbone learns its own normalization, so skip for it.
        if channels == 3 and self._backend in {"dinov2", "vjepa2"}:
            mean = self._IMAGENET_MEAN.to(self._device)
            std  = self._IMAGENET_STD.to(self._device)
            ten = (ten - mean) / std
        return ten

    def _odom_to_tensor(self, msg: Odometry, *, origin: np.ndarray | None = None,
                        position_override: np.ndarray | None = None,
                        orientation_override: tuple[float, float, float, float] | None = None
                        ) -> torch.Tensor:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        if not np.isfinite((px, py, pz)).all():
            raise ValueError("Nonfinite odometry position on the WM bridge")
        if origin is None:
            if self._odom_origin is None:
                self._odom_origin = (px, py, pz)
            origin = np.asarray(self._odom_origin, dtype=np.float64)
        if position_override is not None:
            px, py, pz = position_override
        ox, oy, oz = origin
        q = orientation_override or (
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        vals = [
            px - ox,
            py - oy,
            pz - oz,
            *q,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]
        vals = vals[: self._odom_dim]
        t = torch.tensor([vals], dtype=torch.float32, device=self._device)
        if self._norm_stats is not None:
            t = self._norm_stats.normalize_odom(t)
        return t

    def _imu_to_tensor(self, msg: Imu, *,
                       orientation_override: tuple[float, float, float, float] | None = None
                       ) -> torch.Tensor:
        q = msg.orientation
        orientation = orientation_override or (q.x, q.y, q.z, q.w)
        angular = msg.angular_velocity
        linear = msg.linear_acceleration
        if self._imu_layout == "dataset":
            vals = [*orientation, angular.x, angular.y, angular.z,
                    linear.x, linear.y, linear.z]
        else:
            vals = [angular.x, angular.y, angular.z, linear.x, linear.y,
                    linear.z, *orientation]
        vals = vals[: self._imu_dim]
        t = torch.tensor([vals], dtype=torch.float32, device=self._device)
        if self._norm_stats is not None:
            t = self._norm_stats.normalize_imu(t)
        return t


def main(args: list[str] | None = None) -> None:
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = PlannerBridgeNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
