"""Robot state aggregation service — thread-safe odometry and operational status collection."""

import math
import threading
from collections import deque

from peace.schemas import RobotState


class RobotStateAggregatorService:
    """
    Thread-safe aggregator for robot state.

    Nodes call update_*() methods from ROS2 callbacks.
    The agent calls get_ego_state() to obtain a complete snapshot for LLM reasoning.

    A single _ego_state_lock covers the RobotState snapshot and the history deque.
    """

    def __init__(self, context_window: int = 10) -> None:
        """Initialise the ego state snapshot, bounded history deque, and thread lock."""
        self._ego_state = RobotState()
        self._history: deque[RobotState] = deque(maxlen=context_window)
        self._ego_state_lock = threading.Lock()

    # ── Update methods (called from ROS2 callbacks) ──────────────────────────

    def update_odom(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        """
        Update position, velocity, and orientation from a single odometry message.

        Orientation is derived from the quaternion using the same ZYX Euler convention
        as the MAVROS IMU callback previously used.
        """
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = (
            math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        )

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        with self._ego_state_lock:
            self._ego_state.position_x = x
            self._ego_state.position_y = y
            self._ego_state.position_z = z
            self._ego_state.velocity_x = vx
            self._ego_state.velocity_y = vy
            self._ego_state.velocity_z = vz
            self._ego_state.orientation_roll = math.degrees(roll)
            self._ego_state.orientation_pitch = math.degrees(pitch)
            self._ego_state.orientation_yaw = math.degrees(yaw)
            self._history.append(self._ego_state.model_copy())

    def update_flight_mode(self, mode: str, armed: bool, connected: bool) -> None:
        """
        Update flight/operational mode, armed, and connected flags.

        Called from the /mavros/state (or equivalent) topic callback.
        """
        with self._ego_state_lock:
            self._ego_state.flight_mode = mode
            self._ego_state.armed = armed
            self._ego_state.connected = connected

    # ── Query methods (called from agent / LLM thread) ───────────────────────

    def get_ego_state(self) -> RobotState:
        """Return a complete snapshot of the current robot state."""
        with self._ego_state_lock:
            return self._ego_state.model_copy()

    def get_ego_state_history(self, n: int) -> list[RobotState]:
        """Return the last n odometry-stamped robot state snapshots, oldest first."""
        with self._ego_state_lock:
            history = list(self._history)
        return history[-n:] if n < len(history) else history
