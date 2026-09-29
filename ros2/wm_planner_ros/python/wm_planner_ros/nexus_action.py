"""Nexus action transport shared by all world-model backends."""

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped

# @TODO: remove deps
from phx_nav_msgs.msg import Trajectory
from phx_nav_msgs.action import MRPPlan
from .bridge_adapter import Goal


@dataclass
class PlannedTrajectory:
    frame_id: str
    stamp: Any
    positions: np.ndarray
    dt: float
    cost: float
    first_step: int = 0

    def to_message(self):
        """Convert to traj"""
        positions = np.asarray(self.positions)
        if not self.frame_id:
            raise ValueError("Invalid world-model trajectory: empty frame_id")
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError(
                f"Invalid world-model trajectory: positions shape {positions.shape}, "
                "expected (N, 3) with N >= 2"
            )
        invalid = np.argwhere(~np.isfinite(positions))
        if invalid.size:
            row, column = invalid[0]
            raise ValueError(
                f"Invalid world-model trajectory: nonfinite position at step {row}, "
                f"coordinate {column}"
            )
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(
                f"Invalid world-model trajectory: dt={self.dt}, expected finite positive value"
            )
        if not math.isfinite(self.cost):
            raise ValueError(f"Invalid world-model trajectory: nonfinite cost={self.cost}")
        result = Trajectory()
        result.cost.data = float(self.cost)
        base_time = Time.from_msg(self.stamp)
        for i, xyz in enumerate(positions):
            state = Odometry()
            state.header.frame_id = self.frame_id
            state.header.stamp = (base_time + Duration(seconds=(i + self.first_step) * self.dt)).to_msg()
            state.pose.pose.position.x = float(xyz[0])
            state.pose.pose.position.y = float(xyz[1])
            state.pose.pose.position.z = float(xyz[2])
            # Face along the path and express twist in that body frame.
            delta = positions[min(i + 1, len(positions) - 1)] - positions[max(i - 1, 0)]
            steps = min(i + 1, len(positions) - 1) - max(i - 1, 0)
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            state.pose.pose.orientation.z = math.sin(yaw / 2)
            state.pose.pose.orientation.w = math.cos(yaw / 2)
            state.twist.twist.linear.x = float(np.linalg.norm(delta[:2]) / (steps * self.dt))
            state.twist.twist.linear.z = float(delta[2] / (steps * self.dt))
            result.trajectory.append(state)
        return result


class NexusActionServer:
    """One bounded inference request at a time; observations keep updating."""

    def __init__(self, node):
        self.node = node
        self.action_type = MRPPlan
        self._busy = threading.Lock()
        self.server = ActionServer(
            node, MRPPlan, node._mrp_action_name,
            execute_callback=self.execute,
            goal_callback=self.accept,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def accept(self, request):
        """Accept goal"""
        node = self.node
        requested = request.planner_name.strip().lower()
        if (requested not in {"", "auto", "world_model"}
                or request.planner_mode not in {self.action_type.Goal.PLANNER_AUTO,
                                               self.action_type.Goal.PLANNER_WORLD_MODEL}
                or (request.allowed_planners and "world_model" not in request.allowed_planners)):
            return GoalResponse.REJECT
        p = request.problem.goal.region_center.pose.position
        if (not request.problem.goal.region_center.header.frame_id or not all(math.isfinite(v) for v in (p.x, p.y, p.z))
                or not node._adapter.ready()):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT if self._busy.acquire(blocking=False) else GoalResponse.REJECT

    def execute(self, handle):
        """Handle goal"""
        node = self.node
        result = self.action_type.Result()
        result.selected_planner_name = "world_model"
        result.selected_planner_mode = self.action_type.Goal.PLANNER_WORLD_MODEL
        start = time.monotonic()
        budget = Duration.from_msg(handle.request.planning_timeout).nanoseconds / 1e9
        deadline = start + budget if budget > 0 else None

        def should_cancel():
            """Cancel"""
            return handle.is_cancel_requested or (deadline is not None and time.monotonic() >= deadline)
        try:
            with node._history_lock:
                history = tuple(node._history)
            if len(history) < node._adapter.history_length:
                raise ValueError("Waiting for synchronized observation history")
            odom = node._adapter.observation_odom(history)
            age = (node.get_clock().now() - Time.from_msg(odom.header.stamp)).nanoseconds / 1e9
            if age < 0 or age > node._max_observation_age:
                raise ValueError(f"Observation age {age:.3f}s is outside the allowed window")
            if not odom.header.frame_id or Time.from_msg(odom.header.stamp).nanoseconds == 0:
                raise ValueError("Observation odometry requires a frame and nonzero timestamp")
            goal = PoseStamped()
            goal.header.frame_id = handle.request.problem.goal.region_center.header.frame_id
            goal.header.stamp = odom.header.stamp
            goal.pose = handle.request.problem.goal.region_center.pose
            if goal.header.frame_id != odom.header.frame_id:
                transform = node._tf_buffer.lookup_transform(
                    odom.header.frame_id, goal.header.frame_id, Time.from_msg(odom.header.stamp),
                    timeout=Duration(seconds=0.2),
                )
                goal = do_transform_pose_stamped(goal, transform)
            p = goal.pose.position
            snapshot = node._adapter.snapshot(history, Goal(p.x, p.y, p.z))
            with node._inference_lock:
                if handle.is_cancel_requested:
                    handle.canceled()
                    return result
                plan = node._adapter.plan(snapshot, should_cancel)
            if should_cancel():
                raise InterruptedError("Planning canceled or timed out")
            requested_age = Duration.from_msg(handle.request.max_plan_age).nanoseconds / 1e9
            final_age = (node.get_clock().now() - Time.from_msg(odom.header.stamp)).nanoseconds / 1e9
            if final_age < 0 or (requested_age > 0 and final_age > requested_age):
                raise ValueError("Observation became stale during inference")
            # Action adapters return absolute observation-frame coordinates;
            # visualization publish_frame must never leak into this contract.
            if plan.frame_id != odom.header.frame_id:
                raise ValueError("Model trajectory is not in the observation odometry frame")
            if plan.stamp != odom.header.stamp or plan.first_step != 0:
                raise ValueError("Model trajectory must start at the observation timestamp")
            trajectory = plan.to_message()
            # The first state is at the observation timestamp. Its orientation
            # and body-frame velocity must match the observed robot state;
            # the first path segment describes motion after that timestamp.
            trajectory.trajectory[0].pose.pose.orientation = odom.pose.pose.orientation
            trajectory.trajectory[0].twist.twist = odom.twist.twist
            if should_cancel():
                raise InterruptedError("Planning canceled or timed out")
            # Publish the raw candidate for inspection before Nexus filters it.
            # This topic is diagnostic only and is never used for SRP guidance.
            debug_path = Path()
            debug_path.header = trajectory.trajectory[0].header
            for state in trajectory.trajectory:
                pose = PoseStamped()
                pose.header = state.header
                pose.pose = state.pose.pose
                debug_path.poses.append(pose)
            node._raw_path_pub.publish(debug_path)
            result.trajectories.trajectories = [trajectory]
            result.success = True
            result.used_cached_result = False
            result.message = "Plan generated."
            result.selected_cost = float(plan.cost)
            result.confidence = 1.0 / (1.0 + max(float(plan.cost), 0.0))
            result.candidate_planner_names = ["world_model"]
            result.candidate_success = [True]
            result.candidate_costs = [result.selected_cost]
            result.candidate_confidence = [result.confidence]
            result.candidate_messages = ["accepted"]
            if should_cancel():
                raise InterruptedError("Planning canceled or timed out")
            handle.succeed()
        except Exception as exc:
            result.success = False
            result.trajectories.trajectories.clear()
            result.message = str(exc)
            node.get_logger().error(f"WM MRP request failed: {exc}")
            if handle.is_cancel_requested:
                handle.canceled()
            else:
                handle.abort()
        finally:
            result.planning_time = Duration(seconds=time.monotonic() - start).to_msg()
            self._busy.release()
        return result
