"""Custom exceptions for PEACE."""


class PeaceError(Exception):
    """Base exception for all PEACE errors."""


class SafetyViolationError(PeaceError):
    """Raised when a command would violate safety constraints."""


class InferenceError(PeaceError):
    """Raised when LLM or vision model inference fails."""


class RobotError(PeaceError):
    """Raised when a MAVROS vehicle command fails."""


class ConfigurationError(PeaceError):
    """Raised when configuration is invalid or missing required values."""

