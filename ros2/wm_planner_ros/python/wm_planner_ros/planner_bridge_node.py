#!/usr/bin/env python3
"""ROS 2 bridge for world-model latent-state planning."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
import rclpy.parameter
from builtin_interfaces.msg import Duration as DurationMsg
from huggingface_hub import hf_hub_download
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import torch

from world_model.architecture import WorldModel
from world_model.data.normalization import NormStats
from world_model.mpc import CEMConfig, cem_plan

from wm_planner_ros.bridge_adapter import BufferedPlannerBridgeNode, Goal
from wm_planner_ros.model_adapters import StandardWorldModelAdapter

try:
    from phx_nav_msgs.action import MRPPlan
    from phx_nav_msgs.msg import PlannerState, Trajectory, TrajectoryArray
except ImportError:
    MRPPlan = None
    PlannerState = None
    Trajectory = None
    TrajectoryArray = None


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


class PlannerBridgeNode(BufferedPlannerBridgeNode):
    """Encodes ROS observations and publishes a planned trajectory."""

    def __init__(self) -> None:
        super().__init__("wm_planner_bridge")

        self._declare_params()
        self._load_params()

        self._device = torch.device(self._device_str)
        self._model: WorldModel | None = None  # loaded asynchronously after spin starts
        self._model_ready = False
        self._last_plan: PlanComputation | None = None
        self._last_trajectory_array = None
        self._last_plan_lock = threading.Lock()
        self._mrp_action_server = None
        self.configure_adapter(StandardWorldModelAdapter(self), publish_markers=True)
        if self._enable_mrp_action:
            self._setup_mrp_action_server()
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
        self.declare_parameter("use_goal_topic", True)
        self.declare_parameter("goal_topic", "~/goal")
        self.declare_parameter("goal_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("trajectory_topic", "~/planned_trajectory")
        self.declare_parameter("debug_topic", "~/debug_metrics")
        self.declare_parameter("publish_frame", "body")
        self.declare_parameter("plan_on_observation", True)
        self.declare_parameter("enable_mrp_action", False)
        self.declare_parameter("mrp_action_name", "~/plan_mrp")
        self.declare_parameter("trajectory_dt", 0.2)

        self.declare_parameter("queue_size", 8)
        self.declare_parameter("sync_slop", 0.05)

        self.declare_parameter("odom_dim", 13)
        self.declare_parameter("imu_dim", 10)
        self.declare_parameter("act_dim", 2)
        self.declare_parameter("state_dim", 6)
        self.declare_parameter("z_vis_dim", 128)
        self.declare_parameter("z_prop_dim", 128)
        self.declare_parameter("z_dim", 256)

        self.declare_parameter("image_height", 224)
        self.declare_parameter("image_width", 224)

        self.declare_parameter("norm_stats_path", "")

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
                rclpy.parameter.Parameter("backend", rclpy.parameter.Parameter.Type.STRING, self._backend),
                rclpy.parameter.Parameter("dynamics_type", rclpy.parameter.Parameter.Type.STRING, self._dynamics_type),
            ])

        if self._backend not in {"cnn", "dinov2", "vjepa2"}:
            raise ValueError(f"backend must be one of ['cnn','dinov2','vjepa2']; got {self._backend}")
        if self._dynamics_type not in {"rssm", "mlp"}:
            raise ValueError(f"dynamics_type must be one of ['rssm','mlp']; got {self._dynamics_type}")

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._use_goal_topic = bool(self.get_parameter("use_goal_topic").value)
        self._goal_topic = str(self.get_parameter("goal_topic").value)
        self._trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self._debug_topic = str(self.get_parameter("debug_topic").value)
        self._plan_on_observation = bool(self.get_parameter("plan_on_observation").value)
        self._enable_mrp_action = bool(self.get_parameter("enable_mrp_action").value)
        self._mrp_action_name = str(self.get_parameter("mrp_action_name").value)
        self._trajectory_dt = float(self.get_parameter("trajectory_dt").value)
        self._queue_size = int(self.get_parameter("queue_size").value)
        self._sync_slop = float(self.get_parameter("sync_slop").value)

        self._norm_stats_path = str(self.get_parameter("norm_stats_path").value)

        default_goal = self.get_parameter("goal_xyz").value
        self._default_goal = Goal(float(default_goal[0]), float(default_goal[1]), float(default_goal[2]))
        self._odom_origin: tuple[float, float, float] | None = None  # delta baseline
        self._norm_stats = None  # NormStats loaded in _load_checkpoint

        self._odom_dim = int(self.get_parameter("odom_dim").value)
        self._imu_dim = int(self.get_parameter("imu_dim").value)
        self._act_dim = int(self.get_parameter("act_dim").value)
        self._state_dim = int(self.get_parameter("state_dim").value)
        self._z_vis_dim = int(self.get_parameter("z_vis_dim").value)
        self._z_prop_dim = int(self.get_parameter("z_prop_dim").value)
        self._z_dim = int(self.get_parameter("z_dim").value)

        self._img_h = int(self.get_parameter("image_height").value)
        self._img_w = int(self.get_parameter("image_width").value)

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
            raise ValueError("JEPA-lite requires checkpoint_path with a trained state head")
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

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            args_dict = checkpoint.get("args", {})
            model_cfg = checkpoint.get("model_config", {})
            if not args_dict or not model_cfg:
                raise ValueError("Training checkpoint lacks args or model_config metadata")
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
            self._cem_cfg.action_dim = self._act_dim
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

        # Load NormStats for correct input normalization and output scaling.
        ns_path = self._norm_stats_path
        if not ns_path:
            ns_path = os.path.join(os.path.dirname(checkpoint_path), "norm_stats.pt")
        if os.path.exists(ns_path):
            self._norm_stats = NormStats.load(ns_path)
            self.get_logger().info(f"Loaded NormStats from {ns_path}")
        else:
            self.get_logger().warning(
                f"norm_stats.pt not found at {ns_path}. "
                "Inputs will NOT be normalized — trajectory quality will be poor. "
                "Set the 'norm_stats_path' parameter or place norm_stats.pt beside the checkpoint."
            )

    def _setup_mrp_action_server(self) -> None:
        if MRPPlan is None:
            self.get_logger().error(
                "enable_mrp_action is true, but phx_nav_msgs.action.MRPPlan "
                "could not be imported. Build/source phx_nav_msgs with MRPPlan.action."
            )
            return

        self._mrp_action_server = ActionServer(
            self,
            MRPPlan,
            self._mrp_action_name,
            execute_callback=self._execute_mrp_plan,
            goal_callback=self._handle_mrp_goal,
            cancel_callback=self._handle_mrp_cancel,
        )
        self.get_logger().info(f"MRPPlan action server ready on {self._mrp_action_name}")

    def _handle_mrp_goal(self, goal_request: Any) -> GoalResponse:
        del goal_request
        return GoalResponse.ACCEPT

    def _handle_mrp_cancel(self, goal_handle: Any) -> CancelResponse:
        del goal_handle
        return CancelResponse.ACCEPT

    def _execute_mrp_plan(self, goal_handle: Any) -> Any:
        start_time = time.monotonic()
        request = goal_handle.request
        result = MRPPlan.Result()
        selected_name = self._resolve_requested_planner(request)

        self._publish_mrp_feedback(goal_handle, PlannerState.PLANNING, selected_name, 0.05, "Planning.")

        if selected_name in {"easl", "hybrid"}:
            message = (
                f"Requested planner '{selected_name}' is not connected yet. "
                "The current wm_planner_ros action path supports world_model only."
            )
            self._fill_mrp_failure(result, request, selected_name, message, start_time)
            goal_handle.abort()
            return result

        if selected_name not in {"world_model", "auto"}:
            message = f"Unsupported planner '{selected_name}'."
            self._fill_mrp_failure(result, request, selected_name, message, start_time)
            goal_handle.abort()
            return result

        if not self._model_ready:
            self._fill_mrp_failure(result, request, selected_name, "World model is not ready.", start_time)
            goal_handle.abort()
            return result

        with self._latest_observation_lock:
            latest = self._latest_observation

        if latest is None:
            self._fill_mrp_failure(result, request, selected_name, "No synchronized RGB-D/odom/IMU observation cached.", start_time)
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            self._fill_mrp_failure(result, request, selected_name, "Planning request canceled.", start_time)
            goal_handle.canceled()
            return result

        goal = self._goal_from_mrp_problem(request.problem)
        try:
            rgb, depth, odom, imu = latest
            plan = self._plan_from_observation(rgb, depth, odom, imu, goal)
            self._publish_plan(plan)

            result.success = True
            result.message = "Plan generated."
            result.selected_planner_mode = int(request.planner_mode)
            result.selected_planner_name = "world_model"
            result.trajectories = self._plan_to_trajectory_array(plan)
            result.confidence = float(plan.confidence)
            result.selected_cost = float(plan.cost)
            result.planning_time = self._seconds_to_duration_msg(time.monotonic() - start_time)
            result.used_cached_result = False
            result.candidate_planner_names = ["world_model"]
            result.candidate_success = [True]
            result.candidate_costs = [float(plan.cost)]
            result.candidate_confidence = [float(plan.confidence)]
            result.candidate_messages = ["accepted"]

            self._publish_mrp_feedback(goal_handle, PlannerState.PLANNING, "world_model", 1.0, "Plan generated.")
            goal_handle.succeed()
            return result
        except Exception as exc:
            self._fill_mrp_failure(result, request, selected_name, f"World-model planning failed: {exc}", start_time)
            goal_handle.abort()
            return result

    def _resolve_requested_planner(self, request: Any) -> str:
        planner_name = str(request.planner_name).strip().lower()
        allowed = {str(name).strip().lower() for name in request.allowed_planners if str(name).strip()}

        if not planner_name:
            if request.planner_mode == MRPPlan.Goal.PLANNER_EASL:
                planner_name = "easl"
            elif request.planner_mode == MRPPlan.Goal.PLANNER_WORLD_MODEL:
                planner_name = "world_model"
            elif request.planner_mode == MRPPlan.Goal.PLANNER_HYBRID:
                planner_name = "hybrid"
            else:
                planner_name = "auto"

        if planner_name == "auto":
            planner_name = "world_model"

        if allowed and planner_name not in allowed:
            if "world_model" in allowed:
                return "world_model"
            return sorted(allowed)[0]

        return planner_name

    def _publish_mrp_feedback(
            self,
            goal_handle: Any,
            state: int,
            planner_name: str,
            progress: float,
            message: str
        ) -> None:
        feedback = MRPPlan.Feedback()
        feedback.state.planner = "wm_planner_ros"
        feedback.state.state = int(state)
        feedback.state.verbose_state = message
        feedback.active_planner_name = planner_name
        feedback.progress = float(progress)
        feedback.message = message
        goal_handle.publish_feedback(feedback)

    def _fill_mrp_failure(
            self,
            result: Any,
            request: Any,
            selected_name: str,
            message: str,
            start_time: float
        ) -> None:
        result.success = False
        result.message = message
        result.selected_planner_mode = int(request.planner_mode)
        result.selected_planner_name = selected_name
        result.trajectories = TrajectoryArray()
        result.confidence = 0.0
        result.selected_cost = float("inf")
        result.planning_time = self._seconds_to_duration_msg(time.monotonic() - start_time)
        result.used_cached_result = False
        result.candidate_planner_names = [selected_name]
        result.candidate_success = [False]
        result.candidate_costs = [float("inf")]
        result.candidate_confidence = [0.0]
        result.candidate_messages = [message]

    def _goal_from_mrp_problem(self, problem: Any) -> Goal:
        pose = problem.goal.region_center.pose
        return Goal(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
        )

    def _plan_from_observation(
            self,
            rgb: Image,
            depth: Image,
            odom: Odometry,
            imu: Imu,
            goal: Goal
        ) -> PlanComputation:
        rgb_t = self._image_to_tensor(rgb, channels=3)
        depth_t = self._image_to_tensor(depth, channels=1)
        odom_t = self._odom_to_tensor(odom)
        imu_t = self._imu_to_tensor(imu)

        with torch.no_grad():
            wm_out = self._model.encode(rgb_t, depth_t, odom_t, imu_t)
            z0 = wm_out.z

            cur_x = float(odom.pose.pose.position.x)
            cur_y = float(odom.pose.pose.position.y)
            cur_z = float(odom.pose.pose.position.z)
            if self._norm_stats is not None:
                ns = self._norm_stats
                gx = (goal.x - cur_x) / float(ns.odom_std[0])
                gy = (goal.y - cur_y) / float(ns.odom_std[1])
                gz = (goal.z - cur_z) / float(ns.odom_std[2])
            else:
                gx, gy, gz = goal.x - cur_x, goal.y - cur_y, goal.z - cur_z
            goal_t = torch.tensor([[gx, gy, gz]], dtype=torch.float32, device=self._device)

            true_t0_norm = odom_t[0, :self._state_dim]
            decoded_t0 = self._model.state_head(z0)[0]
            anchor_norm = (decoded_t0 - true_t0_norm).cpu()

            _, best_seq = cem_plan(
                model=self._model,
                z0=z0,
                goal=goal_t,
                cost_fn=self._cost_fn,
                cfg=self._cem_cfg,
                device=self._device,
            )
            z_roll = self._model.rollout(z0, best_seq.unsqueeze(0))
            state_roll = self._model.state_head(z_roll).squeeze(0).cpu().numpy()
            state_roll = state_roll - anchor_norm.numpy()[np.newaxis, :]
            risk_roll = self._model.risk_head(z_roll).squeeze(0).squeeze(-1).cpu().numpy()

        in_frame = (odom.header.frame_id or "map")
        in_frame_norm = str(in_frame).lower()

        if self._norm_stats is not None:
            ns = self._norm_stats
            pos_std = ns.odom_std[:3].numpy()
            pos_mean = ns.odom_mean[:3].numpy()
            state_roll = state_roll.copy()
            state_roll[:, :3] = state_roll[:, :3] * pos_std + pos_mean

        state_to_pub = state_roll.copy()
        out_frame = in_frame_norm

        if self._publish_frame == "body":
            out_frame = self._publish_frame
            qx = odom.pose.pose.orientation.x
            qy = odom.pose.pose.orientation.y
            qz = odom.pose.pose.orientation.z
            qw = odom.pose.pose.orientation.w
            rotation = self._quat_to_rot(qx, qy, qz, qw)

            cur_pos = np.array([
                odom.pose.pose.position.x - (self._odom_origin[0] if self._odom_origin else 0.0),
                odom.pose.pose.position.y - (self._odom_origin[1] if self._odom_origin else 0.0),
                odom.pose.pose.position.z - (self._odom_origin[2] if self._odom_origin else 0.0),
            ], dtype=np.float32)
            state_to_pub = state_roll.copy()
            state_to_pub[:, :3] = (rotation.T @ (state_roll[:, :3] - cur_pos[np.newaxis, :]).T).T

        final_xyz = state_roll[-1, :3]
        goal_xyz = np.array([goal.x, goal.y, goal.z], dtype=np.float32)
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
        self._publish_debug(plan.goal, plan.debug_states, plan.risk_roll)

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

    @staticmethod
    def _seconds_to_duration_msg(seconds: float) -> DurationMsg:
        seconds = max(float(seconds), 0.0)
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return DurationMsg(sec=sec, nanosec=nanosec)

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
        risk_cost = risk_seq.mean(dim=1)
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
            if arr.max() > 100.0:
                arr = arr / 1000.0
        else:
            arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        ten = torch.from_numpy(arr).unsqueeze(0)
        ten = torch.nn.functional.interpolate(
            ten,
            size=(self._img_h, self._img_w),
            mode="bilinear" if channels == 3 else "nearest",
        ).to(self._device)
        # Pretrained ViT backbones (DINOv2, V-JEPA2) require ImageNet normalization.
        # The CNN backbone learns its own normalization, so skip for it.
        if channels == 3 and self._backend in {"dinov2", "vjepa2"}:
            mean = self._IMAGENET_MEAN.to(self._device)
            std  = self._IMAGENET_STD.to(self._device)
            ten = (ten - mean) / std
        return ten

    def _odom_to_tensor(self, msg: Odometry) -> torch.Tensor:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        if self._odom_origin is None:
            self._odom_origin = (px, py, pz)
        ox, oy, oz = self._odom_origin
        vals = [
            px - ox,
            py - oy,
            pz - oz,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
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

    def _imu_to_tensor(self, msg: Imu) -> torch.Tensor:
        vals = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        vals = vals[: self._imu_dim]
        t = torch.tensor([vals], dtype=torch.float32, device=self._device)
        if self._norm_stats is not None:
            t = self._norm_stats.normalize_imu(t)
        return t


def main(args: list[str] | None = None) -> None:
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = PlannerBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
