"""ROS-free planning core for world models trained on window-relative inputs.

Checkpoints trained with ``configs/datasets/wm_relative_5hz.yaml`` see the
world differently from the bridge's legacy path:

- odometry is relative to the current frame, so the current pose is always the
  identity; only the body-frame twist carries information,
- depth and IMU are zero,
- actions are ``[v_forward, v_lateral, yaw_rate]`` per ~0.2 s step,
- goals are poses, reached by CEM through a :class:`PoseDecoder` that reads the
  rolled-out latents (no goal image needed).

Everything needed beyond the checkpoint lives in a small YAML *planner bundle*
written by ``scripts/world_model/export_planner_bundle.py``. This module has no
ROS dependency so it can be tested without a ROS installation; the bridge node
only converts messages to and from the numpy inputs used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from world_model.architecture import WorldModel
from world_model.mpc import CEMConfig, cem_plan_with_cost
from world_model.state_decoder import PoseDecoder

BUNDLE_FORMAT = "relative_pose_v1"


@dataclass
class PlannerBundle:
    """Deployment settings that travel with a relative-pose checkpoint."""

    checkpoint: Path
    state_decoder: Path
    step_dt: float
    horizon: int
    action_low: list[float]
    action_high: list[float]
    action_init_mean: list[float]
    action_init_std: list[float]
    use_depth: bool = False
    use_imu: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "PlannerBundle":
        path = Path(path).expanduser()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path}: expected format {BUNDLE_FORMAT!r}, got {data.get('format')!r}")

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()

        action = data["action"]
        return cls(
            checkpoint=resolve(data["checkpoint"]),
            state_decoder=resolve(data["state_decoder"]),
            step_dt=float(data["step_dt"]),
            horizon=int(data["horizon"]),
            action_low=[float(v) for v in action["low"]],
            action_high=[float(v) for v in action["high"]],
            action_init_mean=[float(v) for v in action["init_mean"]],
            action_init_std=[float(v) for v in action["init_std"]],
            use_depth=bool(data.get("sensors", {}).get("depth", False)),
            use_imu=bool(data.get("sensors", {}).get("imu", False)),
        )


@dataclass
class PoseGoal:
    """Goal in the robot's current body frame (x forward, y left)."""

    x: float
    y: float
    yaw: float | None = None  # None: position-only goal


@dataclass
class RelativePosePlan:
    actions: np.ndarray  # (H, 3) [v_forward, v_lateral, yaw_rate]
    poses: np.ndarray    # (H + 1, 3) [x, y, yaw] in the start body frame, from the decoder
    cost: float

    @property
    def first_action(self) -> np.ndarray:
        return self.actions[0]


def load_world_model(checkpoint: str | Path, device: torch.device) -> tuple[WorldModel, dict, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args, model_cfg = ckpt["args"], ckpt["model_config"]
    dynamics = args.get("dynamics_type", "mlp")
    model = WorldModel(
        odom_dim=model_cfg["odom_dim"], imu_dim=model_cfg["imu_dim"],
        act_dim=model_cfg["act_dim"], state_dim=model_cfg["state_dim"],
        backbone=args.get("backbone", "cnn"),
        backbone_kwargs={"model_name": args.get("backbone_name", "dinov2_vitb14"), "device": str(device)},
        dynamics_type="mlp" if dynamics == "td-jepa" else dynamics,
        image_size=args.get("img_size", 224),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), args, model_cfg


class RelativePosePlanner:
    def __init__(
        self,
        bundle: PlannerBundle,
        device: str | torch.device = "cpu",
        *,
        population: int = 128,
        iters: int = 5,
        elites: int = 16,
        alpha: float = 0.25,
        heading_weight: float = 1.0,
    ) -> None:
        self.bundle = bundle
        self.device = torch.device(device)
        self.model, self.train_args, model_cfg = load_world_model(bundle.checkpoint, self.device)
        self.decoder = PoseDecoder.load(bundle.state_decoder, self.device)
        self.odom_dim, self.imu_dim, self.act_dim = model_cfg["odom_dim"], model_cfg["imu_dim"], model_cfg["act_dim"]
        self.image_size = int(self.train_args.get("img_size", 224))
        if self.act_dim != 3 or len(bundle.action_low) != 3:
            raise ValueError(f"expected 3-dim body-velocity actions, checkpoint has act_dim={self.act_dim}")
        if self.odom_dim < 13:
            raise ValueError(f"expected 13-dim odometry [pose(7), twist(6)], checkpoint has odom_dim={self.odom_dim}")
        tensor = lambda v: torch.tensor(v, dtype=torch.float32, device=self.device)  # noqa: E731
        self.cem = CEMConfig(
            horizon=bundle.horizon, iters=iters, population=population, elites=elites,
            action_dim=3, alpha=alpha,
            action_low=tensor(bundle.action_low), action_high=tensor(bundle.action_high),
            action_init_mean=tensor(bundle.action_init_mean), action_init_std=tensor(bundle.action_init_std),
        )
        self.heading_weight = float(heading_weight)

    def image_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """``(H, W, 3)`` uint8 RGB → ``(1, 3, S, S)`` in [0, 1].

        Training images were resized to a square with PIL bilinear, which
        antialiases when downscaling; ``antialias=True`` matches that.
        """
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"expected (H, W, 3) image, got {rgb.shape}")
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        return F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear",
                             align_corners=False, antialias=True)

    def odom_vector(self, twist: np.ndarray) -> torch.Tensor:
        """Window-relative odometry for the current frame: identity pose + body-frame twist.

        ``twist`` = ``[vx, vy, vz, wx, wy, wz]`` from ``nav_msgs/Odometry.twist``
        (expressed in the child/body frame), as in the training exports.
        """
        odom = np.zeros(self.odom_dim, dtype=np.float32)
        odom[6] = 1.0  # qw: the current frame is the window origin
        odom[7:13] = np.asarray(twist, dtype=np.float32)[:6]
        return torch.from_numpy(odom)[None].to(self.device)

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, twist: np.ndarray) -> torch.Tensor:
        image = self.image_tensor(rgb)
        depth = torch.zeros(1, 1, self.image_size, self.image_size, device=self.device)
        imu = torch.zeros(1, self.imu_dim, device=self.device)
        return self.model.encode(image, depth, self.odom_vector(twist), imu).z

    @torch.no_grad()
    def plan(self, rgb: np.ndarray, twist: np.ndarray, goal: PoseGoal) -> RelativePosePlan:
        z0 = self.encode(rgb, twist)
        goal_t = torch.tensor([goal.x, goal.y], dtype=torch.float32, device=self.device)
        goal_heading = None
        if goal.yaw is not None and self.heading_weight > 0:
            goal_heading = torch.tensor([np.sin(goal.yaw), np.cos(goal.yaw)], dtype=torch.float32, device=self.device)

        def cost(state_seq: torch.Tensor, _risk: torch.Tensor | None, _goal: torch.Tensor) -> torch.Tensor:
            final = state_seq[:, -1]  # (N, 4) [x, y, sin(yaw), cos(yaw)]
            c = torch.norm(final[:, :2] - goal_t, dim=-1)
            if goal_heading is not None:
                c = c + self.heading_weight * torch.norm(final[:, 2:] - goal_heading, dim=-1)
            return c

        _, actions, best_cost = cem_plan_with_cost(
            self.model, z0, goal_t, cost, self.cem, self.device,
            use_risk_head=False, state_decoder=self.decoder,
        )
        decoded = self.decoder(self.model.rollout(z0, actions[None]))[0]
        poses = torch.stack([decoded[:, 0], decoded[:, 1], torch.atan2(decoded[:, 2], decoded[:, 3])], dim=-1)
        return RelativePosePlan(actions=actions.cpu().numpy(), poses=poses.cpu().numpy(), cost=float(best_cost))


def goal_in_body_frame(
    goal_xyz: tuple[float, float, float], goal_yaw: float | None,
    robot_xyz: tuple[float, float, float], robot_quat_xyzw: tuple[float, float, float, float],
) -> PoseGoal:
    """Express a goal given in the odometry frame in the robot's current (planar) body frame."""
    qx, qy, qz, qw = robot_quat_xyzw
    robot_yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    dx, dy = goal_xyz[0] - robot_xyz[0], goal_xyz[1] - robot_xyz[1]
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    yaw = None if goal_yaw is None else float(np.arctan2(np.sin(goal_yaw - robot_yaw), np.cos(goal_yaw - robot_yaw)))
    return PoseGoal(x=float(c * dx + s * dy), y=float(-s * dx + c * dy), yaw=yaw)


def body_poses_to_odom_frame(
    poses: np.ndarray, robot_xyz: tuple[float, float, float], robot_quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """``(N, 3)`` body-frame ``[x, y, yaw]`` → ``(N, 4)`` odom-frame ``[x, y, z, yaw]`` (z = robot z)."""
    qx, qy, qz, qw = robot_quat_xyzw
    robot_yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    x = robot_xyz[0] + c * poses[:, 0] - s * poses[:, 1]
    y = robot_xyz[1] + s * poses[:, 0] + c * poses[:, 1]
    return np.stack([x, y, np.full_like(x, robot_xyz[2]), poses[:, 2] + robot_yaw], axis=-1)


def image_msg_to_rgb(msg) -> np.ndarray:
    """``sensor_msgs/Image`` (or anything with its fields) → ``(H, W, 3)`` uint8 RGB.

    Grayscale cameras are replicated to three channels and 16-bit images are
    scaled by 1/256, as the dataset converter (``convert_rosbag_to_wm_sim.py``)
    does for training data.
    """
    encoding = msg.encoding.lower()
    layouts = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
               "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1), "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1)}
    if encoding not in layouts:
        raise ValueError(f"Unsupported color encoding: {msg.encoding}")
    dtype, channels = layouts[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    arr = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.step // dtype.itemsize)
    arr = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if dtype.itemsize == 2:
        arr = (arr // 256).astype(np.uint8)
    if channels == 1:
        return np.repeat(arr, 3, axis=-1)
    arr = arr[:, :, :3]
    return np.ascontiguousarray(arr[:, :, ::-1] if encoding.startswith("bgr") else arr)
