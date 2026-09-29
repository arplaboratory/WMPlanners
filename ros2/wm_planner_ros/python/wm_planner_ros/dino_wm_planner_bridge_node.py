#!/usr/bin/env python3
"""ROS 2 bridge for the dino-wm world-model architecture.

- The WM model is assembled from separate sub-modules (encoder, predictor,
  proprio_encoder, action_encoder) that are stored together in a ``*.pth``
  checkpoint dict.
- Observations are passed as ``{"visual": (B,T,3,H,W), "proprio": (B,T,D_imu)}``.
- The model predicts future *embeddings*; positions are recovered by
  calling ``WM.decode_act(z)`` which decodes the action/odom token from the
  latent and yields a tensor matching the training ``action_stats`` (odom-like,
  13-dim.  Dims 0:3 of that tensor are **absolute world-frame positions** —
  matching the raw ``action_0..12`` HF dataset columns used during training.
  Normalization stats (action_mean/std) were computed on these absolute values.
- CEM searches over normalized velocity space, forward-integrating absolute
  positions from the current world position.

Checkpoint dict structure:
    {
        "epoch":            int,
        "encoder":          nn.Module  (DINOv2Encoder),
        "predictor":        nn.Module  (ViTPredictor),
        "proprio_encoder":  nn.Module  (ProprioceptiveEmbedding),
        "action_encoder":   nn.Module  (ProprioceptiveEmbedding),
        "decoder":          nn.Module | None,
        # optimizer states - ignored during inference
        "encoder_optimizer":    ...,
        "predictor_optimizer":  ...,
        "action_encoder_optimizer": ...,
        "decoder_optimizer":    ...,
    }
Default checkpoint path: ``$GAS_WS/checkpoints/wm/model_latest.pth``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import torch
import torch.nn.functional as F

from world_model.backbones.wm import WM
from world_model.backbones.dino import build_dinov2
from world_model.backbones.proprio import ProprioceptiveEmbedding
from world_model.backbones.base import ModelBundle

from wm_planner_ros.bridge_adapter import BufferedPlannerBridgeNode, Goal
from wm_planner_ros.model_adapters import DinoWMAdapter
from wm_planner_ros.nexus_action import PlannedTrajectory
from wm_planner_ros.trajectory_geometry import body_to_world_rotation

@dataclass
class _NormStats:
    """Simple per-channel mean/std tensors for a single modality."""
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class DinoWMPlannerBridgeNode(BufferedPlannerBridgeNode):
    """ROS 2 bridge for the dino-wm world-model planning architecture.

    Subscribes to RGB/IMU/Odometry sensor streams, builds a sliding-window
    observation buffer, runs CEM planning with the WM rollout, and publishes
    the planned trajectory as ``nav_msgs/Path`` and ``visualization_msgs/MarkerArray``.
    """

    # ImageNet normalisation constants expected by DINOv2: shape (1,1,3,1,1) for (B,T,C,H,W)
    _DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    _DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def __init__(self) -> None:
        super().__init__("dino_wm_planner_bridge")

        self._declare_params()
        self._load_params()

        self._device = torch.device(self._device_str)
        self._model = None
        self._adapter = DinoWMAdapter(self)
        self._adapter.load_model()

        self._odom_origin: tuple[float, float, float] | None = None

        self.configure_adapter(self._adapter, publish_markers=True)

        self.get_logger().info(
            f"dino_wm_planner_bridge ready. "
            f"num_hist={self._num_hist}, action_dim={self._action_dim}, "
            f"trajectory_topic={self._trajectory_topic}"
        )

    def _declare_params(self) -> None:
        """Declare ROS parameters with defaults."""
        gas_ws = os.getenv("GAS_WS", "")
        default_ckpt = os.path.join(gas_ws, "checkpoints", "wm", "model_latest.pth") if gas_ws else ""
        self.declare_parameter("checkpoint_path", default_ckpt)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("dino_model_name", "dinov2_vits14")

        # topics
        self.declare_parameter("rgb_topic",         "~/rgb")
        self.declare_parameter("odom_topic",        "~/odom")
        self.declare_parameter("imu_topic",         "~/imu")
        self.declare_parameter("publish_frame",     "body")

        # model dimensions
        self.declare_parameter("num_hist",   3)    # train.yaml: num_hist
        self.declare_parameter("action_dim", 13) 
        self.declare_parameter("imu_dim",    10)   # imu_stats has 10 entries
        self.declare_parameter("odom_dim",   13)   # odom_stats has 13 entries
        self.declare_parameter("image_height", 224)
        self.declare_parameter("image_width",  224)
        self.declare_parameter("concat_dim",         1)
        self.declare_parameter("num_action_repeat",  1)
        self.declare_parameter("num_proprio_repeat", 1)

        # normalization stats
        # odom: 13-dim  [pos(3) + orient(4) + lin_vel(3) + ang_vel(3)]
        self.declare_parameter("odom_mean", [
             8.2521, -8.7959,  4.0899,
            -0.0034,  0.0025,  0.3368,  0.3907,
             0.5030,  0.0686, -0.0195,
             0.0079, -0.0002, -0.0033,
        ])
        self.declare_parameter("odom_std", [
            62.1941, 86.9492, 10.1459,
             0.0790,  0.1024,  0.5851,  0.6120,
             1.8989,  1.2242,  0.7451,
             0.3444,  0.3458,  0.3957,
        ])
        # imu: 10-dim  [ang_vel(3) + lin_acc(3) + orient(4)] - order matches hf_datasets.py
        self.declare_parameter("imu_mean", [
             0.0043,  0.0334, -0.0012,
             0.2248,  0.0035, -0.0002,
            -0.0023,  0.1360, -4.8097,  3.9993,
        ])
        self.declare_parameter("imu_std", [
            0.3253, 0.3077, 0.0199,
            0.4098, 0.3289, 0.3188,
            0.3858, 1.7707, 5.1886, 5.2892,
        ])
        # action_stats
        self.declare_parameter("action_mean", [
             8.2521, -8.7959,  4.0899,
            -0.0034,  0.0025,  0.3368,  0.3907,
             0.5030,  0.0686, -0.0195,
             0.0079, -0.0002, -0.0033,
        ])
        self.declare_parameter("action_std", [
            62.1941, 86.9492, 10.1459,
             0.0790,  0.1024,  0.5851,  0.6120,
             1.8989,  1.2242,  0.7451,
             0.3444,  0.3458,  0.3957,
        ])

        # CEM
        self.declare_parameter("plan_horizon",    15)
        self.declare_parameter("plan_iters",       5)
        self.declare_parameter("plan_population", 128)
        self.declare_parameter("plan_elites",      16)
        self.declare_parameter("action_min",      -3.0)
        self.declare_parameter("action_max",       3.0)
        self.declare_parameter("action_init_std",  1.0)
        self.declare_parameter("cem_alpha",        0.3)
        # dt per model step = 1 / (data_hz / frameskip).  frameskip=5 @ 10 Hz → 0.5 s
        self.declare_parameter("frameskip_dt",     0.5)

        # cost weights
        self.declare_parameter("goal_weight",  1.0)
        self.declare_parameter("marker_scale", 0.08)

    def _load_params(self) -> None:
        self._checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self._device_str      = str(self.get_parameter("device").value)
        self._dino_model_name = str(self.get_parameter("dino_model_name").value)
        self._publish_frame   = str(self.get_parameter("publish_frame").value)

        self._rgb_topic        = str(self.get_parameter("rgb_topic").value)
        self._odom_topic       = str(self.get_parameter("odom_topic").value)
        self._imu_topic        = str(self.get_parameter("imu_topic").value)

        self._num_hist   = int(self.get_parameter("num_hist").value)
        self._action_dim = int(self.get_parameter("action_dim").value)
        self._imu_dim    = int(self.get_parameter("imu_dim").value)
        self._odom_dim   = int(self.get_parameter("odom_dim").value)
        self._img_h      = int(self.get_parameter("image_height").value)
        self._img_w      = int(self.get_parameter("image_width").value)
        self._concat_dim        = int(self.get_parameter("concat_dim").value)
        self._num_action_repeat  = int(self.get_parameter("num_action_repeat").value)
        self._num_proprio_repeat = int(self.get_parameter("num_proprio_repeat").value)

        def _ns(mean_key: str, std_key: str) -> _NormStats:
            m = list(self.get_parameter(mean_key).value)
            s = list(self.get_parameter(std_key).value)
            return _NormStats(
                mean=torch.tensor(m, dtype=torch.float32),
                std=torch.tensor(s, dtype=torch.float32),
            )

        self._odom_ns   = _ns("odom_mean",   "odom_std")
        self._imu_ns    = _ns("imu_mean",    "imu_std")
        self._action_ns = _ns("action_mean", "action_std")

        # CEM config
        self._cem_horizon    = int(self.get_parameter("plan_horizon").value)
        self._cem_iters      = int(self.get_parameter("plan_iters").value)
        self._cem_population = int(self.get_parameter("plan_population").value)
        self._cem_elites     = int(self.get_parameter("plan_elites").value)
        self._cem_action_min = float(self.get_parameter("action_min").value)
        self._cem_action_max = float(self.get_parameter("action_max").value)
        self._cem_init_std   = float(self.get_parameter("action_init_std").value)
        self._cem_alpha      = float(self.get_parameter("cem_alpha").value)

        self._goal_weight  = float(self.get_parameter("goal_weight").value)
        self._marker_scale = float(self.get_parameter("marker_scale").value)
        self._frameskip_dt = float(self.get_parameter("frameskip_dt").value)

    def _load_checkpoint(self):
        """Load the WM model from a .pth checkpoint dict.
        """
        ckpt_path = self._checkpoint_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            self.get_logger().warning(
                f"Checkpoint not found at '{ckpt_path}'. "
                "WM will run with random weights.  "
                "Set the 'checkpoint_path' parameter or $GAS_WS."
            )
            return self._build_dummy_model()

        self.get_logger().info(f"Loading dino-wm checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        # Unwrap DDP wrappers if present
        def _unwrap(m):
            if hasattr(m, "module"):
                return m.module
            return m

        encoder_bundle = build_dinov2(
            model_name=self._dino_model_name,
            device=self._device_str,
            image_size=self._img_h,
        )
        encoder = encoder_bundle.model if isinstance(encoder_bundle, ModelBundle) else encoder_bundle
        predictor      = _unwrap(ckpt.get("predictor"))
        proprio_encoder = _unwrap(ckpt["proprio_encoder"])
        action_encoder  = _unwrap(ckpt["action_encoder"])
        decoder        = _unwrap(ckpt.get("decoder"))

        # Infer dims from the loaded sub-modules
        proprio_emb_dim = proprio_encoder.emb_dim
        action_emb_dim  = action_encoder.emb_dim

        model = WM(
            image_size=self._img_h,
            num_hist=self._num_hist,
            num_pred=1,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            decoder=decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self._concat_dim,
            num_action_repeat=self._num_action_repeat,
            num_proprio_repeat=self._num_proprio_repeat,
            raw_action_dim=self._action_dim,
            train_encoder=False,
            train_predictor=False,
            train_decoder=False,
        ).to(self._device)

        model.eval()
        epoch = ckpt.get("epoch", "?")
        self.get_logger().info(
            f"Loaded dino-wm checkpoint epoch={epoch} from {ckpt_path}"
        )
        return model

    def _build_dummy_model(self):
        """Construct a random-weight WM for testing without a checkpoint."""
        encoder_bundle = build_dinov2(device=self._device_str)
        encoder = encoder_bundle.model if isinstance(encoder_bundle, ModelBundle) else encoder_bundle

        proprio_emb_dim = 10
        action_emb_dim  = 10
        proprio_encoder = ProprioceptiveEmbedding(
            in_chans=self._imu_dim, emb_dim=proprio_emb_dim
        )
        action_encoder = ProprioceptiveEmbedding(
            in_chans=self._action_dim, emb_dim=action_emb_dim
        )

        model = WM(
            image_size=self._img_h,
            num_hist=self._num_hist,
            num_pred=1,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            predictor=None,
            decoder=None,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self._concat_dim,
            num_action_repeat=self._num_action_repeat,
            num_proprio_repeat=self._num_proprio_repeat,
            raw_action_dim=self._action_dim,
            train_encoder=False,
            train_predictor=False,
            train_decoder=False,
        ).to(self._device)
        return model

    def _compute_plan(self, snapshot: dict, should_cancel=None) -> PlannedTrajectory:
        """Run CEM and return absolute positions in the observation frame."""
        odom       = snapshot["odom"]
        stamp      = snapshot["stamp"]
        goal       = snapshot["goal"]
        odom_origin = snapshot["odom_origin"]

        # Build tensors
        rgb_stack = np.stack(snapshot["rgb"], axis=0)         # (T, 3, H, W)
        visual_t  = torch.from_numpy(rgb_stack).float().unsqueeze(0).to(self._device)
        dino_mean = self._DINO_MEAN.to(self._device)
        dino_std  = self._DINO_STD.to(self._device)
        visual_t  = (visual_t - dino_mean) / dino_std         # (1, T, 3, H, W)

        imu_stack = np.stack(snapshot["imu"], axis=0)         # (T, imu_dim)
        imu_t     = torch.from_numpy(imu_stack).float().unsqueeze(0).to(self._device)
        imu_t     = self._imu_ns.normalize(imu_t)

        act_stack = np.stack(snapshot["act"], axis=0)         # (T, action_dim)
        act_t     = torch.from_numpy(act_stack).float().unsqueeze(0).to(self._device)
        act_t     = self._action_ns.normalize(act_t)

        obs_0 = {"visual": visual_t, "proprio": imu_t}

        # Current normalised position and velocity from last history step.
        cur_norm = act_t[0, -1, :3]   # (3,) normalised x,y,z
        vel_std  = self._action_ns.std[7:10].to(self._device)   # lin vel std (3,)
        vel_mean = self._action_ns.mean[7:10].to(self._device)

        self.get_logger().debug(f"cur_norm={cur_norm.cpu().numpy()}")

        rotation = body_to_world_rotation(odom.pose.pose.orientation)
        with torch.no_grad():
            best_vel_seq = self._cem_plan(
                obs_0, act_t, goal, odom_origin, cur_norm,
                body_to_world=rotation, should_cancel=should_cancel)
            # best_vel_seq: (H, 3) normalised linear velocities

        # Training action twist is expressed in odometry's child/body frame.
        # CEM holds orientation fixed, so rotate velocities by that same pose
        # before integrating them into absolute odometry-frame positions.
        vel_phys = (best_vel_seq * vel_std + vel_mean).cpu().numpy()  # (H, 3) m/s
        dt = self._frameskip_dt
        pos_delta = np.cumsum((vel_phys @ rotation.T) * dt, axis=0)  # (H, 3) displacement from now

        self.get_logger().debug(
            f"vel_phys[0]={vel_phys[0]}  pos_delta[-1]={pos_delta[-1]}  "
            f"dist={np.linalg.norm(pos_delta[-1]):.2f}m"
        )

        # Build absolute poses in the odom/source frame, then transform to the
        # requested publish frame via TF2.
        source_frame = snapshot["frame"]   # frame_id from the odom message
        cur_world_pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float32)
        traj_src = pos_delta + cur_world_pos   # (H, 3) absolute in source frame

        cost = float(np.linalg.norm(traj_src[-1] - [goal.x, goal.y, goal.z]))
        return PlannedTrajectory(source_frame,
                                 stamp,
                                 np.vstack([cur_world_pos, traj_src]),
                                 dt,
                                 cost)

    def _run_plan_and_publish(self, snapshot: dict) -> None:
        plan = self._compute_plan(snapshot)
        traj = self._transform_traj(
            plan.positions, plan.frame_id, self._publish_frame, plan.stamp
        )
        self._publish_path(self._publish_frame, plan.stamp, traj)
        self._publish_markers(self._publish_frame, plan.stamp, traj)
        self._publish_debug(snapshot["goal"], plan.positions)

    def _cem_plan(
        self,
        obs_0: dict,
        act_hist: torch.Tensor,
        goal: Goal,
        odom_origin: tuple[float, float, float] | None = None,  # kept for API compat, unused
        cur_norm: torch.Tensor | None = None,
        should_cancel=None,
        body_to_world=None,
    ) -> torch.Tensor:
        """CEM over the WM rollout in normalized action space.

        Parameters
        ----------
        obs_0 : dict
            ``{"visual": (1,T,3,H,W), "proprio": (1,T,D_imu)}`` - normalised.
        act_hist : torch.Tensor
            ``(1, num_hist, action_dim)`` - normalised history actions.
            Position dims 0:3 are normalised absolute world-frame coordinates.
        goal : Goal
            World-frame goal position in metres (absolute, same frame as odom).
        odom_origin : unused — kept for API compatibility.
        cur_norm : torch.Tensor, optional
            Normalised current position ``(3,)`` from last history step.

        Returns
        -------
        torch.Tensor
            Best velocity sequence ``(horizon, 3)`` in normalised lin-vel space.
        """
        H   = self._cem_horizon
        N   = self._cem_population
        E   = self._cem_elites
        DA  = self._action_dim
        dev = self._device
        dt  = self._frameskip_dt
        if body_to_world is None:
            raise ValueError("CEM requires the observation's body-to-world rotation")
        rotation = torch.as_tensor(body_to_world, dtype=act_hist.dtype, device=dev)

        # --- Normalisation stats for relevant dims ---
        pos_std  = self._action_ns.std[0:3].to(dev)   # (3,)
        pos_mean = self._action_ns.mean[0:3].to(dev)  # (3,)
        vel_std  = self._action_ns.std[7:10].to(dev)  # lin_vel std (3,)
        vel_mean = self._action_ns.mean[7:10].to(dev) # lin_vel mean (3,)

        # --- Goal and current position in absolute world-frame metres ---
        # act_hist positions are absolute (matching training action columns), so
        # denormalising gives world-frame coordinates directly.
        goal_phys = torch.tensor(
            [goal.x, goal.y, goal.z], dtype=torch.float32, device=dev
        )  # (3,) absolute world frame

        # Current physical position (absolute world frame)
        _cur = cur_norm.to(dev) if cur_norm is not None else act_hist[0, -1, :3]
        cur_phys = _cur * pos_std + pos_mean   # (3,) absolute world frame metres

        # Goal displacement from current position in physical space (xy only for cost)
        goal_disp_phys = goal_phys - cur_phys  # (3,) metres

        # --- CEM over linear velocity dims only (3D, odom dims 7-9) ---
        # Positions are forward-integrated so the trajectory is physically consistent.
        # mean/std in *normalised* velocity space.
        mean = torch.zeros(H, 3, device=dev)  # normalised vel, init at 0 (no motion)
        std  = torch.ones(H, 3, device=dev) * self._cem_init_std

        best_seq  = None
        best_cost = None

        for _ in range(self._cem_iters):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("World-model planning canceled or timed out")
            # Sample normalised velocity sequences: (N, H, 3)
            eps   = torch.randn(N, H, 3, device=dev)
            v_seq = mean.unsqueeze(0) + std.unsqueeze(0) * eps
            v_seq = v_seq.clamp(self._cem_action_min, self._cem_action_max)  # (N, H, 3)

            # Convert normalised velocities to physical m/s
            v_phys = v_seq * vel_std + vel_mean   # (N, H, 3) m/s

            # Forward-integrate positions from current absolute world position
            # pos_phys[n, t] = cur_phys_world + cumsum(v_phys * dt)  (absolute world frame)
            pos_phys = cur_phys.unsqueeze(0).unsqueeze(0) + torch.cumsum((v_phys @ rotation.T) * dt, dim=1)  # (N, H, 3)

            # Normalise integrated positions (absolute world coords) for model input
            pos_norm_seq = (pos_phys - pos_mean) / pos_std  # (N, H, 3)

            # Build full 13-dim action tensor by combining integrated positions with
            # current history's orientation and sampled velocities.
            # Dims: [pos(3), orient(4), lin_vel(3), ang_vel(3)]
            # Keep orientation fixed at last history value; set ang_vel to 0.
            cur_act_norm = act_hist[0, -1].unsqueeze(0).unsqueeze(0).expand(N, H, DA)  # (N, H, 13)
            a = cur_act_norm.clone()
            a[:, :, 0:3]  = pos_norm_seq          # integrated positions
            a[:, :, 7:10] = v_seq                  # sampled lin velocities (normalised)
            a[:, :, 10:13] = 0.0                   # ang_vel = 0 (simplified)

            # Replicate obs_0 over population
            vis_rep = obs_0["visual"].expand(N, -1, -1, -1, -1)   # (N, T, 3, H, W)
            imu_rep = obs_0["proprio"].expand(N, -1, -1)           # (N, T, imu_dim)
            obs_rep = {"visual": vis_rep, "proprio": imu_rep}

            act_hist_rep = act_hist.expand(N, -1, -1)              # (N, T, DA)
            act_full     = torch.cat([act_hist_rep, a], dim=1)      # (N, T+H, DA)

            with torch.no_grad():
                _, z_full    = self._model.rollout(obs_rep, act_full)  # (N, T+H+1, ...)
                act_decoded  = self._model.decode_act(z_full)          # (N, T+H+1, DA)

            # Decoded positions at each future step (normalised)
            fut_pos_norm = act_decoded[:, self._num_hist:, 0:3]    # (N, H+1, 3)
            # Denormalise to absolute world-frame positions, then get displacement from current
            fut_pos_phys = fut_pos_norm * pos_std + pos_mean        # (N, H+1, 3) absolute world
            final_disp   = fut_pos_phys[:, -1, :] - cur_phys       # (N, 3) metres from current

            # Cost = L2 distance to goal (xy plane only for ground robot)
            costs = torch.linalg.norm(final_disp[:, :2] - goal_disp_phys[:2].unsqueeze(0), dim=1)  # (N,)

            # Elite update
            elite_idx  = torch.topk(costs, k=E, largest=False).indices
            elite_v    = v_seq[elite_idx]                             # (E, H, 3)
            elite_mean = elite_v.mean(dim=0)
            elite_std  = elite_v.std(dim=0).clamp_min(1e-3)

            mean = (1 - self._cem_alpha) * mean + self._cem_alpha * elite_mean
            std  = (1 - self._cem_alpha) * std  + self._cem_alpha * elite_std

            min_cost, min_idx = costs.min(dim=0)
            if best_cost is None or float(min_cost) < best_cost:
                best_cost = float(min_cost)
                best_seq  = v_seq[min_idx].detach()   # (H, 3) normalised velocities

        if should_cancel is not None and should_cancel():
            raise InterruptedError("World-model planning canceled or timed out")
        return best_seq  # (H, 3) normalised linear velocities

    def _image_to_numpy(self, msg: Image) -> np.ndarray:
        """Decode a ROS Image to a float32 (3, H, W) array in [0,1]."""
        is_uint8 = "8" in msg.encoding and "32" not in msg.encoding
        dtype = np.uint8 if is_uint8 else np.float32
        arr   = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
        arr   = arr[:, :, :3]
        if is_uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)  # already in [0,1] or metric units
        arr   = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        # Resize to model input size
        ten   = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)
        ten   = F.interpolate(ten, size=(self._img_h, self._img_w), mode="bilinear",
                               align_corners=False)
        return ten.squeeze(0).numpy()  # (3, H, W)

    def _imu_to_numpy(self, msg: Imu) -> np.ndarray:
        """Extract IMU values as a (imu_dim,) float32 array (raw, unormalized)."""
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
        return np.array(vals[: self._imu_dim], dtype=np.float32)

    def _odom_to_numpy(self, msg: Odometry) -> np.ndarray:
        """Extract odom values as a (odom_dim,) float32 array.

        Positions are returned in **absolute world-frame coordinates** to match
        the training data convention: action_0..12 columns in the HF dataset
        contain raw odom readings (no per-episode delta conversion).  The
        normalization stats (action_mean/std) were computed on these absolute
        values, so we must feed absolute positions at inference.

        odom_origin is still stored for bookkeeping (e.g. goal displacement
        in the CEM cost), but it is NOT subtracted here.
        """
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        pz = float(msg.pose.pose.position.z)

        # Record first odom as origin reference for CEM goal displacement.
        if self._odom_origin is None:
            self._odom_origin = (px, py, pz)

        vals = [
            px,
            py,
            pz,
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
        return np.array(vals[: self._odom_dim], dtype=np.float32)

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        x, y, z, w = qx, qy, qz, qw
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
            [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
        ], dtype=np.float32)

    def _to_body_frame(self, pos_delta: np.ndarray, odom: Odometry) -> np.ndarray:
        """Rotate world-frame position deltas (relative to current robot position) into body frame."""
        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w
        R   = self._quat_to_rot(qx, qy, qz, qw)
        return (R.T @ pos_delta.T).T  # (horizon, 3)

    def _transform_traj(
        self,
        traj_src: np.ndarray,
        source_frame: str,
        target_frame: str,
        stamp: Any,
    ) -> np.ndarray:
        """Transform an (H, 3) absolute-position trajectory from *source_frame*
        to *target_frame* using TF2.

        Failed transforms raise so coordinates are never mislabeled.
        """
        if source_frame == target_frame:
            return traj_src

        try:
            tf_stamped: TransformStamped = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(
                f"TF2 lookup {source_frame}->{target_frame} failed: {e}. "
                "Skipping this publication."
            )
            raise

        # Extract rotation (quaternion) and translation from the transform.
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)

        # Apply: p_target = R @ p_source + trans
        return (R @ traj_src.T).T + trans

    def _publish_path(self, frame_id: str, stamp: Any, traj: np.ndarray) -> None:
        """Publish the planned trajectory as a nav_msgs/Path."""
        msg = RosPath()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        for pt in traj:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._path_pub.publish(msg)

    def _publish_markers(
        self, frame_id: str, stamp: Any, traj: np.ndarray
    ) -> None:
        """Publish the planned trajectory as a visualization_msgs/MarkerArray."""
        array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = stamp
        delete_all.ns = "dino_wm_plan"
        delete_all.action = Marker.DELETEALL
        array.markers.append(delete_all)

        n = len(traj)
        for i, pt in enumerate(traj):
            t = float(i) / max(n - 1, 1)  # 0 = near, 1 = far
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns    = "dino_wm_plan"
            m.id    = i + 1
            m.type  = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pt[0])
            m.pose.position.y = float(pt[1])
            m.pose.position.z = float(pt[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self._marker_scale
            m.color.a = 1.0
            m.color.r = t
            m.color.g = 1.0 - t
            m.color.b = 0.2
            array.markers.append(m)

        self._marker_pub.publish(array)

    def _publish_debug(self, goal: Goal, traj: np.ndarray) -> None:
        """Publish debug metrics (e.g. final distance to goal) as a Float32MultiArray."""
        if len(traj) == 0:
            return
        final = traj[-1]
        dist  = float(np.linalg.norm(
            final[:3] - np.array([goal.x, goal.y, goal.z], dtype=np.float32)
        ))
        msg = Float32MultiArray()
        msg.data = [dist, float(self._goal_weight)]
        self._debug_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = DinoWMPlannerBridgeNode()
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
