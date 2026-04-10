"""
DroneTools — deterministic executor for ToolCall steps produced by the LLM planner.

All MAVROS infrastructure lives here: publisher, service clients, setpoint streaming.
The node object is injected so this class never calls rclpy directly.
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

from robot_agent.exceptions import SafetyViolationError, VehicleError

# Setpoint streaming constants
_STREAM_RATE: float = 10.0          # Hz
_OFFBOARD_PRESTREAM_DELAY: float = 0.3  # seconds (PX4 requirement)
_SCAN_SETTLE_TIME: float = 1.5      # seconds to settle at each scan yaw step
_SCAN_STEPS: int = 8                # 360° / 45° per step


class DroneTools:
    """
    Owns all MAVROS infrastructure and exposes high-level tool methods.

    Each tool method returns a descriptive string (success or failure reason).
    When ACTION_ENABLE_CONTROL=False all control methods return immediately with
    a descriptive message and never call any MAVROS service.
    """

    def __init__(self, node, ego_state_svc, world_model_svc, safety_svc, settings) -> None:
        """Inject dependencies and create the MAVROS publisher and service clients."""
        self._node = node
        self._ego_state_svc = ego_state_svc
        self._world_model_svc = world_model_svc
        self._safety_svc = safety_svc
        self._settings = settings

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

    # ── Public tool methods ────────────────────────────────────────────────────

    def forward(self, dist: float = 0.5) -> str:
        """Move forward dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.cos(yaw_rad) * dist
        dy = math.sin(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3,
                                          label=f"Moved forward {dist}m")

    def backward(self, dist: float = 0.5) -> str:
        """Move backward dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.cos(yaw_rad) * (-dist)
        dy = math.sin(yaw_rad) * (-dist)
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3,
                                          label=f"Moved backward {dist}m")

    def left(self, dist: float = 0.5) -> str:
        """Move left dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        # left is negative y in body frame → subtract sin/cos components
        dx = -math.sin(yaw_rad) * dist
        dy = math.cos(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3,
                                          label=f"Moved left {dist}m")

    def right(self, dist: float = 0.5) -> str:
        """Move right dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.sin(yaw_rad) * dist
        dy = -math.cos(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3,
                                          label=f"Moved right {dist}m")

    def up(self, dist: float = 0.5) -> str:
        """Move up dist metres."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = ego.position_z + dist
        return self._navigate_to_waypoint(ego.position_x, ego.position_y, tz, yaw_rad,
                                          tolerance=0.3, label=f"Moved up {dist}m")

    def down(self, dist: float = 0.5) -> str:
        """Move down dist metres."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = max(self._settings.action_min_altitude, ego.position_z - dist)
        return self._navigate_to_waypoint(ego.position_x, ego.position_y, tz, yaw_rad,
                                          tolerance=0.3, label=f"Moved down {dist}m")

    def rotate(self, degrees: float) -> str:
        """Rotate by degrees relative to current heading (positive = clockwise)."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        start_yaw_deg = ego.orientation_yaw
        pos_x, pos_y, pos_z = ego.position_x, ego.position_y, ego.position_z

        self._start_position_stream(pos_x, pos_y, pos_z, math.radians(start_yaw_deg))
        time.sleep(_OFFBOARD_PRESTREAM_DELAY)
        self._ensure_offboard_and_armed()

        num_steps = max(1, round(abs(degrees) / _SCAN_STEPS))
        step_deg = degrees / num_steps
        for i in range(1, num_steps + 1):
            yaw_deg = start_yaw_deg + i * step_deg
            self._start_position_stream(pos_x, pos_y, pos_z, math.radians(yaw_deg))
            time.sleep(_SCAN_SETTLE_TIME)

        ego_after = self._ego_state_svc.get_ego_state()
        return f"Rotated {degrees:+.0f}°. Yaw={ego_after.orientation_yaw:.0f}°"

    def waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> str:
        """Fly to an absolute ENU position."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(yaw) if yaw is not None else math.radians(ego.orientation_yaw)
        return self._navigate_to_waypoint(x, y, z, yaw_rad, tolerance=tolerance,
                                          label=f"Flew to waypoint ({x:.1f},{y:.1f},{z:.1f})m")

    def approach_to_object(
        self,
        label: str,
        stop_distance: float = 2.0,
        z: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> str:
        """Fly toward a detected object and stop stop_distance metres short."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        detections = self._world_model_svc.get_world_model().detections
        target = next((d for d in detections if d["label"] == label and d["depth_valid"]), None)
        if target is None:
            return f"ERROR: Object '{label}' not found or depth invalid"

        obj_x, obj_y = target["world_x"], target["world_y"]
        alt = z if z is not None else ego.position_z
        dx, dy = obj_x - ego.position_x, obj_y - ego.position_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= stop_distance:
            self._node.get_logger().info(
                f"Already within {stop_distance}m of '{label}', skipping approach"
            )
            return f"Already within {stop_distance}m of '{label}'"

        scale = (dist - stop_distance) / dist
        ap_x = ego.position_x + dx * scale
        ap_y = ego.position_y + dy * scale
        approach_yaw = math.atan2(dy, dx)

        self._node.get_logger().info(
            f"Approaching '{label}': stopping {stop_distance}m short at ({ap_x:.1f},{ap_y:.1f})"
        )
        result = self._navigate_to_waypoint(
            ap_x, ap_y, alt, approach_yaw, tolerance=tolerance,
            label=f"Approached '{label}' to {stop_distance}m"
        )

        detections_after = self._world_model_svc.get_world_model().detections
        still_visible = any(d["label"] == label for d in detections_after)
        if still_visible:
            self._node.get_logger().info(f"'{label}' confirmed visible at approach position")
        else:
            self._node.get_logger().warning(f"'{label}' no longer visible after approach")
        return result

    def scan_for_object(
        self,
        label: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> str:
        """Rotate in 45° yaw steps until target_label appears in detections."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        hold_x = x if x is not None else ego.position_x
        hold_y = y if y is not None else ego.position_y
        hold_z = z if z is not None else ego.position_z
        start_yaw_deg = ego.orientation_yaw
        step_deg = 360.0 / _SCAN_STEPS

        self._node.get_logger().info(f"Scanning for '{label}' ({_SCAN_STEPS} steps, {step_deg}° each)")

        with self._lock:
            current_mode = self._current_state.mode if self._current_state else None
        if current_mode != "OFFBOARD":
            self._start_position_stream(
                hold_x, hold_y, hold_z, math.radians(start_yaw_deg)
            )
            time.sleep(_OFFBOARD_PRESTREAM_DELAY)
            if not self._set_mode("OFFBOARD"):
                return "ERROR: Failed to switch to OFFBOARD mode for scan"

        for i in range(_SCAN_STEPS):
            yaw_deg = start_yaw_deg + i * step_deg
            yaw_rad = math.radians(yaw_deg)
            self._start_position_stream(hold_x, hold_y, hold_z, yaw_rad)
            time.sleep(_SCAN_SETTLE_TIME)

            detections = self._world_model_svc.get_world_model().detections
            self._node.get_logger().info(
                f"Scan step {i + 1}/{_SCAN_STEPS}: yaw={yaw_deg:.0f}°, "
                f"detections: {[d['label'] for d in detections]}"
            )
            found = next((d for d in detections if d["label"] == label), None)
            if found:
                bearing_rad = math.atan2(
                    found["world_y"] - hold_y, found["world_x"] - hold_x
                )
                bearing_deg = math.degrees(bearing_rad)
                self._node.get_logger().info(
                    f"Found '{label}' at scan yaw={yaw_deg:.0f}°, "
                    f"world=({found['world_x']:.1f},{found['world_y']:.1f}), "
                    f"rotating to bearing={bearing_deg:.0f}°"
                )
                self._start_position_stream(hold_x, hold_y, hold_z, bearing_rad)
                time.sleep(_SCAN_SETTLE_TIME)
                return f"Found '{label}' at yaw={bearing_deg:.0f}°"

        self._node.get_logger().warning(f"'{label}' not found after full 360° scan")
        return f"'{label}' not found after full 360° scan"

    def arm(self) -> str:
        """Arm the drone motors."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        if not self._set_arm(True):
            return "ERROR: Arming failed"
        return "Armed successfully"

    def disarm(self) -> str:
        """Disarm the drone motors."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        if not self._set_arm(False):
            return "ERROR: Disarming failed"
        return "Disarmed successfully"

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

        if not ego.armed:
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

    def land(self, x: Optional[float] = None, y: Optional[float] = None) -> str:
        """Land at the given position (current position if omitted)."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        land_x = x if x is not None else ego.position_x
        land_y = y if y is not None else ego.position_y

        if land_x != ego.position_x or land_y != ego.position_y:
            # Navigate to landing position first
            self._navigate_to_waypoint(
                land_x, land_y, ego.position_z,
                math.radians(ego.orientation_yaw),
                tolerance=0.5,
                label="Navigating to landing position",
            )

        if not self._set_mode("AUTO.LAND"):
            return "ERROR: Failed to set AUTO.LAND mode"
        self._stop_stream()
        return f"Landing initiated at ({land_x:.1f},{land_y:.1f})"

    def stop_safely(self) -> None:
        """Switch to AUTO.LOITER if in OFFBOARD, then stop setpoint stream."""
        with self._lock:
            in_offboard = self._current_state and self._current_state.mode == "OFFBOARD"
        if in_offboard:
            self._node.get_logger().info("Switching to AUTO.LOITER before stopping setpoints")
            self._set_mode("AUTO.LOITER", timeout=5.0)
            time.sleep(0.5)
        self._stop_stream()

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call by name with the given arguments dict."""
        dispatch = {
            "forward": self.forward,
            "backward": self.backward,
            "left": self.left,
            "right": self.right,
            "up": self.up,
            "down": self.down,
            "rotate": self.rotate,
            "waypoint": self.waypoint,
            "approach_to_object": self.approach_to_object,
            "scan_for_object": self.scan_for_object,
            "arm": self.arm,
            "disarm": self.disarm,
            "takeoff": self.takeoff,
            "land": self.land,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"ERROR: Unknown tool '{tool_name}'"
        try:
            return fn(**arguments)
        except TypeError as e:
            return f"ERROR: Bad arguments for '{tool_name}': {e}"
        except Exception as e:
            return f"ERROR: '{tool_name}' raised {type(e).__name__}: {e}"

    # ── State callback — called by AgentNode to keep DroneTools in sync ───────

    def update_state(self, state: State) -> None:
        """Update the cached MAVROS state (called from AgentNode._state_cb)."""
        with self._lock:
            self._current_state = state

    # ── Internal navigation helpers ───────────────────────────────────────────

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
        # Safety validation — clamp altitude, reject geofence violations
        z = self._safety_svc.clamp_altitude(z)
        try:
            self._safety_svc.validate_position(x, y, z)
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

    def _ensure_offboard_and_armed(self) -> None:
        """Switch to OFFBOARD mode if not already there."""
        with self._lock:
            current_mode = self._current_state.mode if self._current_state else None
        if current_mode != "OFFBOARD":
            if not self._set_mode("OFFBOARD"):
                raise VehicleError("Failed to switch to OFFBOARD mode")

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
        """Start (or restart) the daemon thread streaming PoseStamped at 10 Hz."""
        self._stop_stream()
        with self._lock:
            self._stream_target = {"x": x, "y": y, "z": z, "yaw": yaw}
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
