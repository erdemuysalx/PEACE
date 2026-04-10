"""
Keyboard teleoperation node for manual drone control.

Streams position setpoints to /mavros/setpoint_position/local at 10 Hz and
responds to keyboard input to update the target pose. Follows the same
OFFBOARD setpoint-streaming pattern as AgentNode.

Key bindings:
  w / s   — forward / backward  (body-relative +x / -x)
  a / d   — left / right        (body-relative +y / -y)
  q / e   — yaw CCW / CW
  r / f   — altitude up / down
  o       — engage OFFBOARD mode
  Space   — arm / disarm toggle
  t       — takeoff (arm → OFFBOARD → climb to TAKEOFF_ALTITUDE)
  l       — land (AUTO.LAND mode)
  h       — print key map
  Esc / Ctrl+C — quit (switches to AUTO.LOITER before exit)
"""

import math
import sys
import termios
import threading
import time
import tty
from typing import Optional

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Header

from robot_agent.core.config import get_settings

# ── Tunable constants ─────────────────────────────────────────────────────────

STEP_XY: float = 0.5          # metres per keypress / joy callback
STEP_Z: float = 0.3           # metres per keypress / joy callback
STEP_YAW: float = math.radians(10.0)  # radians per keypress / joy callback
TAKEOFF_ALTITUDE: float = 2.0  # metres (relative)
STREAM_RATE: float = 10.0      # Hz (must be >= 2 for OFFBOARD)
OFFBOARD_PRESTREAM_DELAY: float = 0.3  # seconds (PX4 requirement)

# PS5 DualSense axis/button indices (ROS2 joy, macOS GameController framework)
JOY_AXIS_STRAFE: int = 0      # left stick X  — left/right
JOY_AXIS_FWD: int = 1         # left stick Y  — forward/backward (inverted)
JOY_AXIS_YAW: int = 2         # right stick X — yaw
JOY_AXIS_ALT: int = 3         # right stick Y — altitude (inverted)
JOY_BTN_ARM: int = 0          # Cross (X)     — arm toggle
JOY_BTN_LAND: int = 1         # Circle (O)    — land
JOY_BTN_OFFBOARD: int = 2     # Square        — engage OFFBOARD
JOY_BTN_TAKEOFF: int = 3      # Triangle      — takeoff
JOY_DEADBAND: float = 0.1     # ignore axis values below this threshold

_HELP_TEXT = """
--------------------------------------------------
  PX4 Teleop — Keyboard Controls
--------------------------------------------------
  w / s      forward / backward  ({xy} m)
  a / d      left / right        ({xy} m)
  r / f      altitude up / down  ({z} m)
  q / e      yaw CCW / CW        ({yaw} deg)

  o          engage OFFBOARD mode
  Space      arm / disarm toggle
  t          takeoff to {talt} m
  l          AUTO.LAND
  h          this help text
  Esc        quit (safe)
--------------------------------------------------
""".format(
    xy=STEP_XY,
    z=STEP_Z,
    yaw=round(math.degrees(STEP_YAW), 1),
    talt=TAKEOFF_ALTITUDE,
)


class TeleopNode(Node):
    """
    Manual keyboard teleoperation node.

    Threading model:
      - Main thread:   raw keyboard loop (tty.setraw)
      - Spin thread:   rclpy.spin(self), daemon
      - Stream thread: publishes PoseStamped at STREAM_RATE Hz, daemon
      - _lock (RLock): guards _setpoint + _current_state
    """

    def __init__(self) -> None:
        """Set up subscribers, publisher, service clients, and the setpoint stream thread."""
        super().__init__("teleop_node")
        self.settings = get_settings()

        if not self.settings.action_enable_control:
            self.get_logger().warning(
                "Control disabled — set ACTION_ENABLE_CONTROL=True to enable flight commands"
            )

        # ── State (guarded by _lock) ──────────────────────────────────────────
        self._lock = threading.RLock()
        self._setpoint: dict = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self._current_state: Optional[State] = None
        self._pos_seeded = False  # True once odometry has been used to seed setpoint
        self._prev_buttons: list[int] = []  # previous Joy button state for edge detection

        # ── Setpoint streaming ─────────────────────────────────────────────────
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()

        # ── QoS profiles (mirrors AgentNode) ─────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        reentrant_cb = ReentrantCallbackGroup()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            State,
            self.settings.topic_namespace + "/state",
            self._state_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            Odometry,
            self.settings.topic_odom,
            self._odom_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )

        self.create_subscription(
            Joy, "/joy", self._joy_cb, best_effort_qos, callback_group=reentrant_cb
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._position_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", best_effort_qos
        )

        # ── Service clients ───────────────────────────────────────────────────
        self._set_mode_client = self.create_client(
            SetMode, "/mavros/set_mode", callback_group=reentrant_cb
        )
        self._arming_client = self.create_client(
            CommandBool, "/mavros/cmd/arming", callback_group=reentrant_cb
        )

        self.get_logger().info(
            "Teleop Node ready | press 'h' for key map | "
            f"control={'enabled' if self.settings.action_enable_control else 'DISABLED'}"
        )

    # ── Telemetry callbacks ────────────────────────────────────────────────────

    def _state_cb(self, msg: State) -> None:
        """Cache the latest MAVROS State message."""
        with self._lock:
            self._current_state = msg

    def _odom_cb(self, msg: Odometry) -> None:
        """Seed the initial setpoint from odometry on first message, then ignore."""
        with self._lock:
            if self._pos_seeded:
                return
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            self._setpoint = {"x": p.x, "y": p.y, "z": p.z, "yaw": yaw}
            self._pos_seeded = True
            self.get_logger().info(
                f"Initial setpoint seeded from odometry: "
                f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f}), yaw={math.degrees(yaw):.1f} deg"
            )

    def _joy_cb(self, msg: Joy) -> None:
        """Map PS5 DualSense axes and buttons to drone commands."""
        axes = list(msg.axes)
        buttons = list(msg.buttons)

        def _axis(idx: int) -> float:
            v = axes[idx] if idx < len(axes) else 0.0
            return v if abs(v) >= JOY_DEADBAND else 0.0

        # Analog axes — applied every callback while stick is held
        strafe = _axis(JOY_AXIS_STRAFE)
        fwd = -_axis(JOY_AXIS_FWD)   # inverted: push up = positive Y on stick = forward
        yaw = _axis(JOY_AXIS_YAW)
        alt = -_axis(JOY_AXIS_ALT)   # inverted: push up = positive altitude

        if strafe or fwd or alt:
            self._move(dfwd=fwd * STEP_XY, dleft=-strafe * STEP_XY, dz=alt * STEP_Z)
        if yaw:
            self._rotate(-yaw * STEP_YAW)

        # Buttons — fire once on press (rising edge)
        prev = self._prev_buttons
        for idx, (cur, prv) in enumerate(
            zip(buttons, prev + [0] * (len(buttons) - len(prev)))
        ):
            if cur == 1 and prv == 0:
                if idx == JOY_BTN_ARM:
                    self._do_arm_toggle()
                elif idx == JOY_BTN_LAND:
                    self._do_land()
                elif idx == JOY_BTN_OFFBOARD:
                    self._engage_offboard()
                elif idx == JOY_BTN_TAKEOFF:
                    self._do_takeoff()
        self._prev_buttons = buttons

    # ── Setpoint streaming (mirrors AgentNode pattern) ─────────────────────────

    def _start_stream(self) -> None:
        """Start the daemon setpoint streaming thread, stopping any existing one first."""
        self._stop_stream()
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()
        self.get_logger().info("Setpoint stream started")

    def _stop_stream(self) -> None:
        """Signal the stream thread to stop and wait for it to finish."""
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_stop.set()
            self._stream_thread.join(timeout=2.0)
            self.get_logger().info("Setpoint stream stopped")

    def _stream_worker(self) -> None:
        """Daemon thread: publish the current setpoint at STREAM_RATE Hz."""
        period = 1.0 / STREAM_RATE
        while not self._stream_stop.is_set():
            tick = time.time()
            with self._lock:
                sp = dict(self._setpoint)
            self._publish_setpoint(sp["x"], sp["y"], sp["z"], sp["yaw"])
            elapsed = time.time() - tick
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        """Publish a single PoseStamped setpoint to MAVROS."""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position = Point(x=x, y=y, z=z)
        msg.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )
        self._position_pub.publish(msg)

    # ── MAVROS service helpers ─────────────────────────────────────────────────

    def _set_mode(self, mode: str, timeout: float = 5.0) -> bool:
        """Request a MAVROS mode change and return True on success."""
        if not self._set_mode_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("Set mode service unavailable")
            return False
        req = SetMode.Request()
        req.custom_mode = mode
        future = self._set_mode_client.call_async(req)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error("Set mode request timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        if result and result.mode_sent:
            return True
        self.get_logger().error(f"Failed to set mode to {mode}")
        return False

    def _set_arm(self, arm: bool, timeout: float = 5.0) -> bool:
        """Send an arming or disarming command and return True on success."""
        if not self._arming_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("Arming service unavailable")
            return False
        req = CommandBool.Request()
        req.value = arm
        future = self._arming_client.call_async(req)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error("Arming request timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        if result and result.success:
            self.get_logger().info("Armed" if arm else "Disarmed")
            return True
        self.get_logger().error("Arming command rejected")
        return False

    # ── Key actions ───────────────────────────────────────────────────────────

    def _current_mode(self) -> str:
        """Return the current flight mode string, or empty string if state is unknown."""
        with self._lock:
            return self._current_state.mode if self._current_state else ""

    def _is_armed(self) -> bool:
        """Return True if the vehicle is currently armed."""
        with self._lock:
            return bool(self._current_state and self._current_state.armed)

    def _engage_offboard(self) -> bool:
        """Start stream, wait 0.3 s, then switch to OFFBOARD (PX4 requirement)."""
        if not self.settings.action_enable_control:
            print("\n[WARN] Control disabled — set ACTION_ENABLE_CONTROL=True")
            return False
        self._start_stream()
        time.sleep(OFFBOARD_PRESTREAM_DELAY)
        ok = self._set_mode("OFFBOARD")
        if ok:
            self.get_logger().info("OFFBOARD mode engaged")
        return ok

    def _do_takeoff(self) -> None:
        """Arm the drone, set target altitude, and engage OFFBOARD mode."""
        if not self.settings.action_enable_control:
            print("\n[WARN] Control disabled — set ACTION_ENABLE_CONTROL=True")
            return
        if not self._is_armed():
            print("\nArming...")
            if not self._set_arm(True):
                return
        # Set target altitude before streaming
        with self._lock:
            self._setpoint["z"] = TAKEOFF_ALTITUDE
        print(f"\nTaking off to {TAKEOFF_ALTITUDE} m...")
        self._engage_offboard()

    def _do_land(self) -> None:
        """Switch to AUTO.LAND mode."""
        if not self.settings.action_enable_control:
            print("\n[WARN] Control disabled — set ACTION_ENABLE_CONTROL=True")
            return
        print("\nLanding (AUTO.LAND)...")
        self._set_mode("AUTO.LAND")

    def _do_arm_toggle(self) -> None:
        """Arm the drone if disarmed, disarm if armed."""
        if not self.settings.action_enable_control:
            print("\n[WARN] Control disabled — set ACTION_ENABLE_CONTROL=True")
            return
        if self._is_armed():
            self._set_arm(False)
        else:
            self._set_arm(True)

    def _move(self, dfwd: float = 0.0, dleft: float = 0.0, dz: float = 0.0) -> None:
        """Apply a body-relative position delta to the current setpoint."""
        with self._lock:
            yaw = self._setpoint["yaw"]
            # Rotate body-relative delta to ENU frame
            dx_enu = math.cos(yaw) * dfwd - math.sin(yaw) * dleft
            dy_enu = math.sin(yaw) * dfwd + math.cos(yaw) * dleft
            new_x = self._setpoint["x"] + dx_enu
            new_y = self._setpoint["y"] + dy_enu
            new_z = self._setpoint["z"] + dz
            # Clamp to safety limits
            limit = self.settings.action_horizontal_limit
            new_x = max(-limit, min(limit, new_x))
            new_y = max(-limit, min(limit, new_y))
            new_z = max(
                self.settings.action_min_altitude,
                min(self.settings.action_max_altitude, new_z),
            )
            self._setpoint["x"] = new_x
            self._setpoint["y"] = new_y
            self._setpoint["z"] = new_z

    def _rotate(self, dyaw: float) -> None:
        """Add dyaw (radians) to the current setpoint yaw, wrapping to [0, 2π)."""
        with self._lock:
            self._setpoint["yaw"] = (self._setpoint["yaw"] + dyaw) % (2 * math.pi)

    def _print_status(self) -> None:
        """Print the current mode, arm state, and setpoint to stdout on a single line."""
        with self._lock:
            sp = dict(self._setpoint)
            mode = self._current_state.mode if self._current_state else "?"
            armed = self._current_state.armed if self._current_state else False
        print(
            f"\r[{mode}|{'ARMED' if armed else 'DISARMED'}] "
            f"target=({sp['x']:.1f}, {sp['y']:.1f}, {sp['z']:.1f}) m  "
            f"yaw={math.degrees(sp['yaw']):.0f} deg   ",
            end="",
            flush=True,
        )

    # ── Main keyboard loop ─────────────────────────────────────────────────────

    def run_keyboard_loop(self) -> None:
        """
        Blocking raw-terminal keyboard loop. Restores the terminal on exit.
        Must be called from the main thread.
        """
        print(_HELP_TEXT)
        print("Waiting for MAVROS connection and odometry...")

        # Wait until we have state and an initial position
        deadline = time.time() + 10.0
        while not self._pos_seeded:
            if time.time() > deadline:
                print("\n[WARN] No odometry received — setpoint starts at (0,0,0)")
                break
            time.sleep(0.1)

        print("Ready. Use keys to fly. Press 'h' for help, Esc to quit.\n")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                key = sys.stdin.read(1)
                if not key:
                    continue

                if key in ("\x03", "\x1b"):  # Ctrl+C or Esc
                    break
                elif key == "w":
                    self._move(dfwd=STEP_XY)
                elif key == "s":
                    self._move(dfwd=-STEP_XY)
                elif key == "a":
                    self._move(dleft=STEP_XY)
                elif key == "d":
                    self._move(dleft=-STEP_XY)
                elif key == "r":
                    self._move(dz=STEP_Z)
                elif key == "f":
                    self._move(dz=-STEP_Z)
                elif key == "q":
                    self._rotate(STEP_YAW)
                elif key == "e":
                    self._rotate(-STEP_YAW)
                elif key == "o":
                    self._engage_offboard()
                elif key == " ":
                    self._do_arm_toggle()
                elif key == "t":
                    self._do_takeoff()
                elif key == "l":
                    self._do_land()
                elif key == "h":
                    # Temporarily restore terminal to print multi-line help cleanly
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    print(_HELP_TEXT)
                    tty.setraw(fd)
                    continue

                self._print_status()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print()  # newline after raw-mode output

    def shutdown(self) -> None:
        """Safe shutdown: switch out of OFFBOARD before stopping the stream."""
        if self._current_mode() == "OFFBOARD":
            self.get_logger().info("Switching to AUTO.LOITER before stopping setpoints")
            self._set_mode("AUTO.LOITER")
            time.sleep(0.5)
        self._stop_stream()


def main(args=None) -> None:
    """Initialise ROS2, start the TeleopNode, and run the keyboard or joystick loop."""
    import argparse
    parser = argparse.ArgumentParser(description="PX4 Teleop Node")
    parser.add_argument(
        "--joystick",
        action="store_true",
        help="Joystick-only mode: skip keyboard loop, rely solely on /joy topic",
    )
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args or None)
    node = TeleopNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if parsed.joystick:
            print("Joystick mode — connect PS5 controller and run: ros2 run joy joy_node")
            print("Press Ctrl+C to quit.\n")
            spin_thread.join()
        else:
            node.run_keyboard_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down teleop node...")
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
