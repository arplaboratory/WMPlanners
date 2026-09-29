"""Shared ROS interface and planner scheduling."""

from __future__ import annotations

import collections
import math
import threading
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import message_filters
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener

@dataclass(frozen=True)
class Goal:
    """Position goal"""
    x: float
    y: float
    z: float


class PlannerAdapter(ABC):
    """Model-specific observation preparation and planning API."""

    def __init__(self, node: "BufferedPlannerBridgeNode") -> None:
        self.node = node

    @property
    @abstractmethod
    def history_length(self) -> int:
        """Number of synchronized observations needed for one plan."""

    @property
    def history_interval_s(self) -> float:
        """Required time between history samples, or zero for single-step models."""
        return 0.0

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
    def observation_odom(self, history: tuple[Any, ...]) -> Any:
        """Odometry associated with the last buffered observation."""

    @abstractmethod
    def plan(self, snapshot: Any, should_cancel: Callable[[], bool]) -> Any:
        """Return a PlannedTrajectory in the observation odometry frame."""

    @abstractmethod
    def plan_and_publish(self, snapshot: Any) -> None:
        """Run model-specific planning on the worker thread."""


class BufferedPlannerBridgeNode(Node):
    """Own ROS I/O, observation history, and a latest-request-wins worker."""

    def __init__(self, name: str) -> None:
        super().__init__(name, namespace="/wm_planner")
        self._robot_name = str(self.declare_parameter("robot_name", "").value).strip("/")
        if "/" in self._robot_name:
            raise ValueError("robot_name must be one robot namespace segment")
        self._plan_on_observation = self.declare_parameter("plan_on_observation", True).value
        self._enable_mrp_action = self.declare_parameter("enable_mrp_action", False).value
        self._mrp_action_name = self.declare_parameter("mrp_action_name", "plan_mrp").value
        self._max_observation_age = self.declare_parameter("max_observation_age", 2.0).value
        if not 0 < self._max_observation_age < float("inf"):
            raise ValueError("max_observation_age must be finite and positive")
        self._use_goal_topic = bool(self.declare_parameter("use_goal_topic", True).value)
        self._goal_topic = str(self.declare_parameter("goal_topic", "~/goal").value)
        self._trajectory_topic = str(self.declare_parameter("trajectory_topic",
                                                            "~/planned_trajectory").value)
        self._debug_topic = str(self.declare_parameter("debug_topic", "~/debug_metrics").value)
        self._queue_size = int(self.declare_parameter("queue_size", 8).value)
        self._sync_slop = float(self.declare_parameter("sync_slop", 0.05).value)
        goal_xyz = self.declare_parameter("goal_xyz", [0.0, 0.0, 0.0]).value
        if len(goal_xyz) != 3:
            raise ValueError("goal_xyz must contain exactly three coordinates")
        self._default_goal = Goal(*(float(value) for value in goal_xyz))
        self._adapter: PlannerAdapter | None = None
        self._history: collections.deque[Any] = collections.deque()
        self._goal_sub = None
        self._path_pub = None
        self._debug_pub = None
        self._marker_pub = None
        self._sensor_subs = []
        self._sync = None
        self._plan_thread: threading.Thread | None = None
        self._history_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._latest_observation = None
        self._latest_observation_lock = threading.Lock()
        self._goal: Goal | None = None
        self._pending_lock = threading.Lock()
        self._pending_snapshot: Any | None = None
        self._pending_event = threading.Event()
        self._stop_worker = False
        self._nexus_action = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def configure_adapter(self, adapter: PlannerAdapter, *, publish_markers: bool) -> None:
        """Planner adapter"""
        if adapter.history_length < 1:
            raise ValueError("Adapter history_length must be at least one")
        interval = adapter.history_interval_s
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("History interval must be finite and nonnegative")
        self._adapter = adapter
        self._history = collections.deque(maxlen=adapter.history_length)
        if self._use_goal_topic:
            self._goal_sub = self.create_subscription(PoseStamped,
                                                      self._goal_topic,
                                                      self._on_goal, 10)

        self._path_pub = self.create_publisher(Path, self._trajectory_topic, 10)
        self._debug_pub = self.create_publisher(Float32MultiArray, self._debug_topic, 10)
        self._marker_pub = (
            self.create_publisher(MarkerArray, self._trajectory_topic + "/markers", 10)
            if publish_markers else None
        )

        streams = adapter.streams()
        self._sensor_subs = [
            message_filters.Subscriber(self,
                                       message_type,
                                       topic,
                                       qos_profile=qos_profile_sensor_data)
            for message_type, topic in streams
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._sensor_subs, queue_size=self._queue_size, slop=self._sync_slop
        )
        self._sync.registerCallback(self._on_observations)

        self._plan_thread = threading.Thread(
            target=self._planning_loop, daemon=True, name=f"{self.get_name()}_planner"
        )
        self._plan_thread.start()
        if self._enable_mrp_action:
            from .nexus_action import NexusActionServer

            self._nexus_action = NexusActionServer(self)

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = Goal(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def _on_observations(self, *messages: Any) -> None:
        with self._latest_observation_lock:
            self._latest_observation = messages
        try:
            sample = self._adapter.prepare(*messages)
            with self._history_lock:
                interval = self._adapter.history_interval_s
                if interval > 0 and self._history:
                    current = self._adapter.observation_odom((sample,))
                    previous = self._adapter.observation_odom((self._history[-1],))
                    delta = (Time.from_msg(current.header.stamp) -
                             Time.from_msg(previous.header.stamp)).nanoseconds / 1e9
                    if current.header.frame_id != previous.header.frame_id or delta <= 0:
                        self._history.clear()
                    elif delta < interval - 1e-6:
                        return
                    elif delta > interval * 1.25:
                        # A gap cannot be interpreted as one trained model step.
                        self._history.clear()
                self._history.append(sample)
                history = tuple(self._history)
            if len(history) < self._adapter.history_length:
                return
            if not self._adapter.ready() or not getattr(self, "_plan_on_observation", True):
                return
            goal = self._goal if self._goal is not None else self._default_goal
            snapshot = self._adapter.snapshot(history, goal)
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
                with self._inference_lock:
                    self._adapter.plan_and_publish(snapshot)
            except Exception as exc:
                self.get_logger().error(f"Planning error: {exc}\n{traceback.format_exc()}")

    def destroy_node(self):
        if self._plan_thread is not None:
            with self._pending_lock:
                self._stop_worker = True
                self._pending_snapshot = None
                self._pending_event.set()
            self._plan_thread.join()
        if self._nexus_action is not None:
            self._nexus_action.server.destroy()
        return super().destroy_node()
