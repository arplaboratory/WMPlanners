"""Shared ROS interface and latest-observation planner scheduling."""

from __future__ import annotations

import collections
import math
import os
import threading
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import message_filters
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray


@dataclass(frozen=True)
class Goal:
    x: float
    y: float
    z: float
    yaw: float | None = None  # set only when the planner uses goal headings


class PlannerAdapter(ABC):
    """Model-specific observation preparation and planning API."""

    def __init__(self, node: "BufferedPlannerBridgeNode") -> None:
        self.node = node

    @property
    @abstractmethod
    def history_length(self) -> int:
        """Number of synchronized observations needed for one plan."""

    @abstractmethod
    def streams(self) -> Sequence[tuple[type, str]]:
        """Message type and topic for each synchronized input."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and any checkpoint before planning."""

    @abstractmethod
    def ready(self) -> bool:
        """Whether model inference can run."""

    @abstractmethod
    def prepare(self, *messages: Any) -> Any:
        """Convert one synchronized observation for the history buffer."""

    @abstractmethod
    def snapshot(self, history: tuple[Any, ...], goal: Goal) -> Any:
        """Build a stable planning request from the current history."""

    @abstractmethod
    def plan_and_publish(self, snapshot: Any) -> None:
        """Run model-specific planning on the worker thread."""


class BufferedPlannerBridgeNode(Node):
    """Own ROS I/O, observation history, and a latest-request-wins worker."""

    def __init__(self, name: str) -> None:
        super().__init__(name, namespace=os.getenv("GAS_ROBOT_NAME", ""))

    def configure_adapter(self, adapter: PlannerAdapter, *, publish_markers: bool) -> None:
        self._adapter = adapter
        if adapter.history_length < 1:
            raise ValueError("Adapter history_length must be at least one")
        self._history = collections.deque(maxlen=adapter.history_length)
        self._latest_observation = None
        self._latest_observation_lock = threading.Lock()
        self._goal: Goal | None = None
        if self._use_goal_topic:
            self.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)

        self._path_pub = self.create_publisher(Path, self._trajectory_topic, 10)
        self._debug_pub = self.create_publisher(Float32MultiArray, self._debug_topic, 10)
        self._marker_pub = (
            self.create_publisher(MarkerArray, self._trajectory_topic + "/markers", 10)
            if publish_markers else None
        )

        streams = adapter.streams()
        self._sensor_subs = [
            message_filters.Subscriber(self, message_type, topic, qos_profile=qos_profile_sensor_data)
            for message_type, topic in streams
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._sensor_subs, queue_size=self._queue_size, slop=self._sync_slop
        )
        self._sync.registerCallback(self._on_observations)

        self._pending_lock = threading.Lock()
        self._pending_snapshot: Any | None = None
        self._pending_event = threading.Event()
        self._stop_worker = False
        self._plan_thread = threading.Thread(
            target=self._planning_loop, daemon=True, name=f"{self.get_name()}_planner"
        )
        self._plan_thread.start()

    def _on_goal(self, msg: PoseStamped) -> None:
        yaw = None
        if getattr(self, "_use_goal_heading", False):
            q = msg.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._goal = Goal(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            yaw,
        )

    def _on_observations(self, *messages: Any) -> None:
        with self._latest_observation_lock:
            self._latest_observation = messages
        try:
            sample = self._adapter.prepare(*messages)
            self._history.append(sample)
            if len(self._history) < self._adapter.history_length:
                return
            if not self._adapter.ready() or not getattr(self, "_plan_on_observation", True):
                return
            goal = self._goal if self._goal is not None else self._default_goal
            snapshot = self._adapter.snapshot(tuple(self._history), goal)
            with self._pending_lock:
                self._pending_snapshot = snapshot
                self._pending_event.set()
        except Exception as exc:
            self.get_logger().error(f"Observation processing error: {exc}\n{traceback.format_exc()}")

    def _planning_loop(self) -> None:
        while True:
            self._pending_event.wait()
            with self._pending_lock:
                if self._stop_worker:
                    return
                snapshot = self._pending_snapshot
                self._pending_snapshot = None
                self._pending_event.clear()
            if snapshot is None:
                continue
            try:
                self._adapter.plan_and_publish(snapshot)
            except Exception as exc:
                self.get_logger().error(f"Planning error: {exc}\n{traceback.format_exc()}")

    def destroy_node(self):
        if hasattr(self, "_plan_thread"):
            with self._pending_lock:
                self._stop_worker = True
                self._pending_snapshot = None
                self._pending_event.set()
            self._plan_thread.join()
        return super().destroy_node()
