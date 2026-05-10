"""Generic RobotTools abstraction — the base contract every platform-specific tool set extends.

Defines the platform-agnostic motion primitives the LLM planner can call
(forward/back/left/right/rotate/waypoint/navigate_to_object/localize_object) plus
the @tool registry used by execute() to dispatch ToolCall steps. Subclasses
(UAVTools, future MobileRobotTools, ...) implement the actuation backend by
providing _navigate_to_waypoint, _hold_position, and stop_safely.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

# Default timing/step constants for the generic primitives.
# Subclasses may override these as class attributes if their actuation
# stack requires different pacing.
_SCAN_SETTLE_TIME: float = 1.5      # seconds to settle at each yaw step
_LOCALIZE_STEPS: int = 36           # 360° / 10° per step — fine scan for object localisation
_ROTATE_STEP_DEG: float = 30.0      # max yaw delta per intermediate setpoint during rotate()


def tool(fn: Callable) -> Callable:
    """Mark a method as LLM-callable.

    Tagged methods are merged into the owning class's ``_tools`` registry by
    ``RobotTools.__init_subclass__`` (and by the module-level collection pass for
    ``RobotTools`` itself), so subclasses inherit parent tools and add their own
    without manual bookkeeping.
    """
    fn._is_tool = True  # type: ignore[attr-defined]
    return fn


def _collect_tools(cls: type) -> dict[str, Callable]:
    """Walk the MRO and collect every method tagged with ``@tool``.

    Subclass entries override parent entries with the same name (standard MRO
    semantics), so a platform can specialise a generic primitive if needed.
    """
    merged: dict[str, Callable] = {}
    for base in reversed(cls.__mro__):
        for name, attr in vars(base).items():
            if callable(attr) and getattr(attr, "_is_tool", False):
                merged[name] = attr
    return merged


class RobotTools(ABC):
    """Abstract base for platform-specific tool sets exposed to the LLM planner.

    Concrete subclasses (e.g. ``UAVTools``) own the actuation backend and
    register additional ``@tool`` methods for capabilities unique to their
    platform (arming, takeoff, vertical motion, gripper, ...).
    """

    _tools: ClassVar[dict[str, Callable]] = {}

    # Defaults — subclasses can override as class attributes.
    SCAN_SETTLE_TIME: ClassVar[float] = _SCAN_SETTLE_TIME
    LOCALIZE_STEPS: ClassVar[int] = _LOCALIZE_STEPS
    ROTATE_STEP_DEG: ClassVar[float] = _ROTATE_STEP_DEG

    def __init__(
        self,
        node,
        ego_state_svc,
        world_model_svc,
        constraint_svc,
        settings,
    ) -> None:
        """Inject ROS node and shared services. Subclasses extend with backend setup."""
        self._node = node
        self._ego_state_svc = ego_state_svc
        self._world_model_svc = world_model_svc
        self._constraint_svc = constraint_svc
        self._settings = settings

    def __init_subclass__(cls, **kwargs) -> None:
        """Merge inherited and freshly-decorated @tool methods into cls._tools."""
        super().__init_subclass__(**kwargs)
        cls._tools = _collect_tools(cls)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch an LLM ToolCall by name with the given keyword arguments."""
        fn = self._tools.get(tool_name)
        if fn is None:
            return f"ERROR: Unknown tool '{tool_name}'"
        try:
            return fn(self, **arguments)
        except TypeError as e:
            return f"ERROR: Bad arguments for '{tool_name}': {e}"
        except Exception as e:
            return f"ERROR: '{tool_name}' raised {type(e).__name__}: {e}"

    @classmethod
    def tool_names(cls) -> list[str]:
        """Names of every LLM-callable tool on this robot (including inherited).

        TODO: drive ``peace/system_prompts/language.txt`` generation from this
        registry so the prompt and the dispatch table stay in sync.
        """
        return list(cls._tools.keys())

    # ── Generic motion primitives (concrete; expressed via abstract hooks) ────

    @tool
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

    @tool
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

    @tool
    def left(self, dist: float = 0.5) -> str:
        """Move left dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = -math.sin(yaw_rad) * dist
        dy = math.cos(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3,
                                          label=f"Moved left {dist}m")

    @tool
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

    @tool
    def rotate(self, degrees: float) -> str:
        """Rotate by degrees relative to current heading (positive = counter-clockwise in ENU)."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        start_yaw_deg = ego.orientation_yaw
        pos_x, pos_y, pos_z = ego.position_x, ego.position_y, ego.position_z

        # Slice the rotation into intermediate setpoints no larger than ROTATE_STEP_DEG
        # so the low-level controller tracks smoothly instead of slewing in one jump.
        self._hold_position(pos_x, pos_y, pos_z, math.radians(start_yaw_deg))
        num_steps = max(1, math.ceil(abs(degrees) / self.ROTATE_STEP_DEG))
        step_deg = degrees / num_steps
        for i in range(1, num_steps + 1):
            yaw_deg = start_yaw_deg + i * step_deg
            self._hold_position(pos_x, pos_y, pos_z, math.radians(yaw_deg))
            time.sleep(self.SCAN_SETTLE_TIME)

        ego_after = self._ego_state_svc.get_ego_state()
        return f"Rotated {degrees:+.0f}°. Yaw={ego_after.orientation_yaw:.0f}°"

    @tool
    def waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> str:
        """Fly/drive to an absolute ENU position."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(yaw) if yaw is not None else math.radians(ego.orientation_yaw)
        return self._navigate_to_waypoint(x, y, z, yaw_rad, tolerance=tolerance,
                                          label=f"Moved to waypoint ({x:.1f},{y:.1f},{z:.1f})m")

    @tool
    def navigate_to_object(
        self,
        label: str,
        stop_distance: float = 2.0,
        z: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> str:
        """Move toward a detected object and stop stop_distance metres short."""
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

    @tool
    def localize_object(
        self,
        label: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> str:
        """Rotate in fine yaw steps until target_label appears in detections."""
        if not self._settings.action_enable_control:
            return "Control disabled (ACTION_ENABLE_CONTROL=False)"
        ego = self._ego_state_svc.get_ego_state()
        hold_x = x if x is not None else ego.position_x
        hold_y = y if y is not None else ego.position_y
        hold_z = z if z is not None else ego.position_z
        start_yaw_deg = ego.orientation_yaw
        step_deg = 360.0 / self.LOCALIZE_STEPS

        self._node.get_logger().info(
            f"Localising '{label}' ({self.LOCALIZE_STEPS} steps, {step_deg:.1f}° each)"
        )

        # Prime the stream at start_yaw so OFFBOARD engages before the first
        # rotation step; the loop then yaws away from start_yaw on every tick.
        self._hold_position(hold_x, hold_y, hold_z, math.radians(start_yaw_deg))

        for i in range(1, self.LOCALIZE_STEPS + 1):
            yaw_deg = start_yaw_deg + i * step_deg
            yaw_rad = math.radians(yaw_deg)
            self._hold_position(hold_x, hold_y, hold_z, yaw_rad)
            time.sleep(self.SCAN_SETTLE_TIME)

            detections = self._world_model_svc.get_world_model().detections
            self._node.get_logger().info(
                f"Localise step {i}/{self.LOCALIZE_STEPS}: yaw={yaw_deg:.1f}°, "
                f"detections: {[d['label'] for d in detections]}"
            )
            found = next((d for d in detections if d["label"] == label), None)
            if found:
                bearing_rad = math.atan2(
                    found["world_y"] - hold_y, found["world_x"] - hold_x
                )
                bearing_deg = math.degrees(bearing_rad)
                self._node.get_logger().info(
                    f"Found '{label}' at scan yaw={yaw_deg:.1f}°, "
                    f"world=({found['world_x']:.1f},{found['world_y']:.1f}), "
                    f"rotating to bearing={bearing_deg:.1f}°"
                )
                self._hold_position(hold_x, hold_y, hold_z, bearing_rad)
                time.sleep(self.SCAN_SETTLE_TIME)
                return f"Found '{label}' at yaw={bearing_deg:.1f}°"

        self._node.get_logger().warning(f"'{label}' not found after full 360° scan")
        return f"ERROR: '{label}' not found after full 360° scan"

    # ── Abstract — actuation backend; subclass implements ────────────────────

    @abstractmethod
    def _navigate_to_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        tolerance: float,
        label: str,
    ) -> str:
        """Drive the platform to (x, y, z, yaw) and block until arrival or timeout."""

    @abstractmethod
    def _hold_position(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Command the platform to maintain (x, y, z, yaw). Idempotent."""

    @abstractmethod
    def stop_safely(self) -> None:
        """Bring the platform to a safe halt (loiter, brake, etc.) and stop control output."""


# Ensure RobotTools itself has its own decorated methods registered, even though
# __init_subclass__ only fires for subclasses.
RobotTools._tools = _collect_tools(RobotTools)
