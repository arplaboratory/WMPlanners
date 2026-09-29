"""Model-family adapters for the shared ROS planner bridge."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu

from .bridge_adapter import Goal, PlannerAdapter
from .nexus_action import PlannedTrajectory


class StandardWorldModelAdapter(PlannerAdapter):
    """CNN/DINOv2/V-JEPA2 standard and JEPA-lite WorldModel planning."""

    @property
    def history_length(self) -> int:
        return 1

    def streams(self) -> Sequence[tuple[type, str]]:
        node = self.node
        return (
            (Image, node._rgb_topic),
            (Image, node._depth_topic),
            (Odometry, node._odom_topic),
            (Imu, node._imu_topic),
        )

    def load_model(self) -> None:
        self.node._load_checkpoint()
        self.node._model.eval()
        self.node._model_ready = True

    def ready(self) -> bool:
        return self.node._model_ready

    def prepare(self, *messages: Any) -> tuple[Any, ...]:
        return messages

    def snapshot(self, history: tuple[Any, ...], goal: Goal) -> tuple[Any, ...]:
        return (*history[-1], goal)

    def observation_odom(self, history):
        return history[-1][2]

    def plan(self, snapshot, should_cancel: Callable[[], bool]):
        plan = self.node._plan_from_observation(
            *snapshot, output_in_odom=True, should_cancel=should_cancel)
        return PlannedTrajectory(plan.frame_id, plan.stamp, plan.trajectory_states[:, :3],
                                 self.node._trajectory_dt, plan.cost)

    def plan_and_publish(self, snapshot: tuple[Any, ...]) -> None:
        plan = self.node._plan_from_observation(*snapshot)
        self.node._publish_plan(plan)


class DinoWMAdapter(PlannerAdapter):
    """DinoWM history preparation and its existing CEM/TF planning path."""

    @property
    def history_length(self) -> int:
        return self.node._num_hist

    @property
    def history_interval_s(self) -> float:
        interval = self.node._frameskip_dt
        if interval <= 0:
            raise ValueError("frameskip_dt must be positive")
        return interval

    def streams(self) -> Sequence[tuple[type, str]]:
        node = self.node
        return (
            (Image, node._rgb_topic),
            (Odometry, node._odom_topic),
            (Imu, node._imu_topic),
        )

    def load_model(self) -> None:
        self.node._model = self.node._load_checkpoint()
        self.node._model.eval()

    def ready(self) -> bool:
        return self.node._model is not None

    def prepare(self, rgb: Image, odom: Odometry, imu: Imu) -> dict[str, Any]:
        node = self.node
        odom_np = node._odom_to_numpy(odom)
        return {
            "rgb": node._image_to_numpy(rgb),
            "imu": node._imu_to_numpy(imu),
            "act": odom_np[:node._action_dim].copy(),
            "odom": odom,
        }

    def snapshot(self, history: tuple[dict[str, Any], ...], goal: Goal) -> dict[str, Any]:
        node = self.node
        odom = history[-1]["odom"]
        return {
            "rgb": [sample["rgb"] for sample in history],
            "imu": [sample["imu"] for sample in history],
            "act": [sample["act"] for sample in history],
            "odom": odom,
            "stamp": odom.header.stamp,
            "frame": str(odom.header.frame_id),
            "odom_origin": node._odom_origin,
            "goal": goal,
        }

    def observation_odom(self, history):
        return history[-1]["odom"]

    def plan(self, snapshot, should_cancel: Callable[[], bool]):
        return self.node._compute_plan(snapshot, should_cancel=should_cancel)

    def plan_and_publish(self, snapshot: dict[str, Any]) -> None:
        self.node._run_plan_and_publish(snapshot)
