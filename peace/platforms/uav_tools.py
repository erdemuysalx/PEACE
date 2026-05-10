"""
UAVTools — UAV-specific tool set, MAVROS/PX4 actuation backend for the planner-executor.

Inherits the generic motion primitives (forward/back/left/right/rotate/waypoint/
navigate_to_object/localize_object) from RobotTools and provides:
  - The actuation backend (_navigate_to_waypoint, _hold_position, stop_safely)
    over OFFBOARD setpoint streaming.
  - UAV-only tools registered via @tool: arm/disarm/takeoff/land/up/down.
All MAVROS infrastructure (publisher, service clients, setpoint stream thread)
lives here; the planner injects the rclpy node and shared services.
"""

import math
import threading
import time
from typing import Optional

from geometry_msgs.msg import Point, PoseStamped, Quaternion
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from peace.exceptions import SafetyViolationError, RobotError
from peace.platforms.robot_tools import RobotTools, tool

# Setpoint streaming constants (PX4-specific)
_STREAM_RATE: float = 10.0               # Hz
_OFFBOARD_PRESTREAM_DELAY: float = 0.3   # seconds (PX4 requirement before mode switch)


class UAVTools(RobotTools):
    """
    UAV tool set. Owns MAVROS infrastructure and exposes flight-specific tools
    on top of the generic RobotTools primitives.

    Each tool method returns a descriptive string (success or failure reason).
    When ACTION_ENABLE_CONTROL=False all control methods return immediately with
    a descriptive message and never call any MAVROS service.
    """

    def __init__(self, node, ego_state_svc, world_model_svc, constraint_svc, settings) -> None:
        """Inject dependencies and create the MAVROS publisher and service clients."""
        super().__init__(node, ego_state_svc, world_model_svc, constraint_svc, settings)

        # ── Execution state ────────────────────────────────────────────────────
        self._lock = threading.RLock()
        self._current_state: Optional[State] = None

        # ── Setpoint streaming ─────────────────────────────────────────────────
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()
        self._stream_target: Optional[dict] = None

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reentrant_cb = ReentrantCallbackGroup()

        # ── Publisher ──────────────────────────────────────────────────────────
        self._position_pub = node.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", best_effort_qos
        )

        # ── Service clients ────────────────────────────────────────────────────
        self._set_mode_client = node.create_client(
            SetMode, "/mavros/set_mode", callback_group=reentrant_cb
        )
        self._arming_client = node.create_client(
            CommandBool, "/mavros/cmd/arming", callback_group=reentrant_cb
        )
        self._takeoff_client = node.create_client(
            CommandTOL, "/mavros/cmd/takeoff", callback_group=reentrant_cb
        )
        self._land_client = node.create_client(
            CommandTOL, "/mavros/cmd/land", callback_group=reentrant_cb
        )

    # ── UAV-specific tools ────────────────────────────────────────────────────

    @tool
    def up(self, dist: float = 0.5) -> str:
        """Move up dist metres."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = ego.position_z + dist
        return self._navigate_to_waypoint(ego.position_x, ego.position_y, tz, yaw_rad,
                                          tolerance=0.3, label=f"Moved up {dist}m")

    @tool
    def down(self, dist: float = 0.5) -> str:
        """Move down dist metres."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = max(self._settings.action_min_altitude, ego.position_z - dist)
        return self._navigate_to_waypoint(ego.position_x, ego.position_y, tz, yaw_rad,
                                          tolerance=0.3, label=f"Moved down {dist}m")

    @tool
    def arm(self) -> str:
        """Arm the drone motors."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        if not self._set_arm(True):
            return "ERROR: Arming failed"
        return "Armed successfully"

    @tool
    def disarm(self) -> str:
        """Disarm the drone motors."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        if not self._set_arm(False):
            return "ERROR: Disarming failed"
        return "Disarmed successfully"

    @tool
    def takeoff(self, height: float = 3.0) -> str:
        """Arm the drone and climb to height metres."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()

        target_z = max(self._settings.action_min_altitude,
                       min(self._settings.action_max_altitude, height))

        # PX4 sequence: stream setpoints first, then OFFBOARD, then arm
        self._start_position_stream(ego.position_x, ego.position_y, target_z,
                                    math.radians(ego.orientation_yaw))
        time.sleep(_OFFBOARD_PRESTREAM_DELAY)

        if not self._set_mode("OFFBOARD"):
            return "ERROR: Failed to switch to OFFBOARD mode for takeoff"

        # Re-read armed state: PX4 auto-disarms on-ground drones after ~10s,
        # so the cached ego.armed sampled before the OFFBOARD switch may be stale.
        ego_pre_arm = self._ego_state_svc.get_ego_state()
        if not ego_pre_arm.armed:
            if not self._set_arm(True):
                return "ERROR: Arming failed before takeoff"

        arrived = self._wait_for_arrival(
            ego.position_x, ego.position_y, target_z,
            tolerance=0.5,
            timeout=self._settings.action_command_timeout,
        )
        ego_after = self._ego_state_svc.get_ego_state()
        alt = ego_after.position_z
        if arrived:
            return f"Takeoff complete. Altitude={alt:.1f}m"
        return f"Takeoff timeout — current altitude={alt:.1f}m (target={target_z:.1f}m)"

    @tool
    def land(self, x: Optional[float] = None, y: Optional[float] = None) -> str:
        """Land at the given position (current position if omitted) and wait for touchdown."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        land_x = x if x is not None else ego.position_x
        land_y = y if y is not None else ego.position_y

        # Use a horizontal tolerance to decide whether a re-position is needed —
        # exact float equality would force a redundant navigation step every time.
        horiz_err = math.hypot(land_x - ego.position_x, land_y - ego.position_y)
        if horiz_err > 0.5:
            self._navigate_to_waypoint(
                land_x, land_y, ego.position_z,
                math.radians(ego.orientation_yaw),
                tolerance=0.5,
                label="Navigating to landing position",
            )

        # Stop the OFFBOARD setpoint stream BEFORE switching to AUTO.LAND so PX4
        # does not see a fresh /mavros/setpoint_position/local message arrive
        # mid mode-switch and re-engage OFFBOARD.
        self._stop_stream()

        if not self._set_mode("AUTO.LAND"):
            return "ERROR: Failed to set AUTO.LAND mode"

        # Touchdown timeout scales with altitude — PX4 default LAND descent rate is
        # ~0.7 m/s, so a 28 m hover needs ~40 s before disarm. Add slack on top.
        ego_at_land = self._ego_state_svc.get_ego_state()
        descent_budget = max(
            self._settings.action_command_timeout,
            ego_at_land.position_z / 0.7 + 10.0,
        )
        deadline = time.time() + descent_budget

        # Wait for PX4's auto-disarm after touchdown. Don't fail just because the
        # cached MAVROS mode hasn't caught up yet — the state callback runs at a
        # finite rate and lags the set_mode service response by a few hundred ms.
        while time.time() < deadline:
            ego_now = self._ego_state_svc.get_ego_state()
            if not ego_now.armed:
                return (
                    f"Landed at ({land_x:.1f},{land_y:.1f}), "
                    f"altitude={ego_now.position_z:.2f}m"
                )
            time.sleep(0.2)

        ego_final = self._ego_state_svc.get_ego_state()
        return (
            f"Land timeout — alt={ego_final.position_z:.2f}m, armed={ego_final.armed} "
            f"at ({land_x:.1f},{land_y:.1f})"
        )

    # ── Abstract method implementations ──────────────────────────────────────

    def _navigate_to_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        tolerance: float,
        label: str,
    ) -> str:
        """Stream setpoints, engage OFFBOARD, wait for arrival. Returns result string."""
        # Constraint enforcement — clamp altitude, reject geofence violations
        z = self._constraint_svc.clamp_altitude(z)
        try:
            self._constraint_svc.validate_position(x, y, z)
        except SafetyViolationError as e:
            return f"ERROR: Safety violation — {e}"

        self._start_position_stream(x, y, z, yaw_rad)
        # PX4 requires setpoints to be streaming for ≥0.3s before switching to OFFBOARD
        time.sleep(_OFFBOARD_PRESTREAM_DELAY)
        self._ensure_offboard_and_armed()
        arrived = self._wait_for_arrival(
            x, y, z, tolerance=tolerance, timeout=self._settings.action_command_timeout
        )
        ego = self._ego_state_svc.get_ego_state()
        pos_str = (
            f"Position: ({ego.position_x:.1f},{ego.position_y:.1f},{ego.position_z:.1f})m, "
            f"yaw={ego.orientation_yaw:.0f}°, alt={ego.position_z:.1f}m"
        )
        if arrived:
            return f"{label}. {pos_str}"
        return f"{label} (arrival timeout). {pos_str}"

    def _hold_position(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Command a hold setpoint. Idempotently engages OFFBOARD on first call."""
        self._start_position_stream(x, y, z, yaw_rad)
        with self._lock:
            in_offboard = (
                self._current_state is not None and self._current_state.mode == "OFFBOARD"
            )
        if not in_offboard:
            time.sleep(_OFFBOARD_PRESTREAM_DELAY)
            self._set_mode("OFFBOARD")

    def stop_safely(self) -> None:
        """Switch to AUTO.LOITER if in OFFBOARD, then stop setpoint stream."""
        with self._lock:
            in_offboard = self._current_state and self._current_state.mode == "OFFBOARD"
        if in_offboard:
            self._node.get_logger().info("Switching to AUTO.LOITER before stopping setpoints")
            self._set_mode("AUTO.LOITER", timeout=5.0)
            time.sleep(0.5)
        self._stop_stream()

    # ── State callback — called by AgentNode to keep UAVTools in sync ───────

    def update_state(self, state: State) -> None:
        """Update the cached MAVROS state (called from AgentNode._state_cb)."""
        with self._lock:
            self._current_state = state

    # ── Internal navigation helpers ───────────────────────────────────────────

    def _ensure_offboard_and_armed(self) -> None:
        """Switch to OFFBOARD mode if not already there."""
        with self._lock:
            current_mode = self._current_state.mode if self._current_state else None
        if current_mode != "OFFBOARD":
            if not self._set_mode("OFFBOARD"):
                raise RobotError("Failed to switch to OFFBOARD mode")

    def _wait_for_arrival(
        self, x: float, y: float, z: float, tolerance: float, timeout: float
    ) -> bool:
        """Poll ego_state_svc until within tolerance or timeout. Returns True on arrival."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ego = self._ego_state_svc.get_ego_state()
            dist = math.sqrt(
                (ego.position_x - x) ** 2
                + (ego.position_y - y) ** 2
                + (ego.position_z - z) ** 2
            )
            if dist <= tolerance:
                return True
            time.sleep(0.2)
        return False

    # ── Setpoint streaming ────────────────────────────────────────────────────

    def _start_position_stream(self, x: float, y: float, z: float, yaw: float) -> None:
        """Stream PoseStamped setpoints toward (x, y, z, yaw) at 10 Hz.

        If the worker thread is already running, the new target is swapped in
        under the lock and picked up on the next tick — this avoids tearing
        down and rebuilding the thread on every yaw step inside rotate() /
        localize_object() loops.
        """
        with self._lock:
            self._stream_target = {"x": x, "y": y, "z": z, "yaw": yaw}
            already_running = (
                self._stream_thread is not None and self._stream_thread.is_alive()
            )
        if already_running:
            return
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()
        self._node.get_logger().info(
            f"Setpoint stream started: ({x:.2f},{y:.2f},{z:.2f}), yaw={math.degrees(yaw):.1f}°"
        )

    def _stop_stream(self) -> None:
        """Signal the stream thread to stop and join it."""
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_stop.set()
            self._stream_thread.join(timeout=2.0)
            self._node.get_logger().info("Setpoint stream stopped")

    def _stream_worker(self) -> None:
        """Daemon thread: publish PoseStamped at action_setpoint_stream_rate Hz."""
        period = 1.0 / self._settings.action_setpoint_stream_rate
        while not self._stream_stop.is_set():
            tick = time.time()
            with self._lock:
                target = self._stream_target
            if target:
                self._publish_setpoint(target["x"], target["y"], target["z"], target["yaw"])
            elapsed = time.time() - tick
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        """Build and publish a single PoseStamped message to MAVROS."""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position = Point(x=x, y=y, z=z)
        msg.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )
        self._position_pub.publish(msg)

    # ── MAVROS service helpers ────────────────────────────────────────────────

    def _set_mode(self, mode: str, timeout: float = 5.0) -> bool:
        """Call the MAVROS set_mode service and return True on success."""
        if not self._set_mode_client.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().error("Set mode service unavailable")
            return False
        req = SetMode.Request()
        req.custom_mode = mode
        future = self._set_mode_client.call_async(req)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self._node.get_logger().error("Set mode request timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        if result and result.mode_sent:
            return True
        self._node.get_logger().error(f"Failed to set mode to {mode}")
        return False

    def _set_arm(self, arm: bool, timeout: float = 5.0) -> bool:
        """Call the MAVROS arming service and return True on success."""
        if not self._arming_client.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().error("Arming service unavailable")
            return False
        req = CommandBool.Request()
        req.value = arm
        future = self._arming_client.call_async(req)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self._node.get_logger().error("Arming request timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        if result and result.success:
            self._node.get_logger().info("Armed" if arm else "Disarmed")
            return True
        self._node.get_logger().error("Arming command rejected")
        return False
