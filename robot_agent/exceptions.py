"""Custom exceptions for Robot Agent."""


class RobotAgentError(Exception):
    """Base exception for all Robot Agent errors."""


class SafetyViolationError(RobotAgentError):
    """Raised when a command would violate safety constraints."""


class InferenceError(RobotAgentError):
    """Raised when LLM or vision model inference fails."""


class VehicleError(RobotAgentError):
    """Raised when a MAVROS vehicle command fails."""


class ConfigurationError(RobotAgentError):
    """Raised when configuration is invalid or missing required values."""

