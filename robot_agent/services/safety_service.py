"""Safety validation service for positions and altitude limits."""

from robot_agent.exceptions import SafetyViolationError


class SafetyValidatorService:
    """
    Validates and clamps positions against configurable safety limits.

    Raises SafetyViolationError for hard violations (position out of geofence).
    Provides clamp_altitude() for soft violations (altitude adjustments).
    """

    def __init__(
        self,
        min_altitude: float,
        max_altitude: float,
        horizontal_limit: float,
    ) -> None:
        """Store altitude and horizontal geofence limits from settings."""
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.horizontal_limit = horizontal_limit

    def validate_position(self, x: float, y: float, z: float) -> None:
        """
        Raise SafetyViolationError if the position violates safety constraints.

        Altitude violations are surfaced as warnings (caller should call
        clamp_altitude instead). Geofence violations are hard errors.
        """
        if abs(x) > self.horizontal_limit or abs(y) > self.horizontal_limit:
            raise SafetyViolationError(
                f"Position ({x:.1f}, {y:.1f}) exceeds "
                f"±{self.horizontal_limit}m geofence limit"
            )

        if z < self.min_altitude:
            raise SafetyViolationError(
                f"Altitude {z:.1f}m is below minimum {self.min_altitude}m"
            )

        if z > self.max_altitude:
            raise SafetyViolationError(
                f"Altitude {z:.1f}m exceeds maximum {self.max_altitude}m"
            )

    def clamp_altitude(self, z: float) -> float:
        """Return altitude clamped to the safe range."""
        return max(self.min_altitude, min(self.max_altitude, z))
