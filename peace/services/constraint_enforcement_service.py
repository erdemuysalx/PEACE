"""Constraint enforcement service — validates positions and altitude limits.

Currently enforces safety constraints (geofence, altitude bounds). Designed as
the central entry point for additional constraint classes (mission, regulatory,
operational) as PEACE evolves.
"""

from peace.exceptions import SafetyViolationError


class ConstraintEnforcementService:
    """
    Validates and clamps positions against configurable constraint limits.

    Raises SafetyViolationError for hard violations (position out of geofence).
    Provides clamp_altitude() for soft violations (altitude adjustments).
    """

    def __init__(
        self,
        min_altitude: float,
        max_altitude: float,
        horizontal_limit: float,
    ) -> None:
        """Store altitude and horizontal geofence constraint bounds from settings."""
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.horizontal_limit = horizontal_limit

    def validate_position(self, x: float, y: float, z: float) -> None:
        """
        Raise SafetyViolationError if (x, y) violates the horizontal geofence.

        Altitude is treated as a soft constraint: callers must run the value
        through clamp_altitude() before validate_position(), so the altitude
        bounds are not re-checked here.
        """
        if abs(x) > self.horizontal_limit or abs(y) > self.horizontal_limit:
            raise SafetyViolationError(
                f"Position ({x:.1f}, {y:.1f}) exceeds "
                f"±{self.horizontal_limit}m geofence limit"
            )

    def clamp_altitude(self, z: float) -> float:
        """Return altitude clamped to the safe range."""
        return max(self.min_altitude, min(self.max_altitude, z))
