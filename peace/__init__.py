"""
PEACE — agentic control architecture for robotic platforms, built on ROS2 and leveraging large models for decision-making.
"""

__version__ = "0.2.0"

__all__ = [
    # Nodes
    "VisionNode",
    "AgentNode",
    # Schemas
    "Object",
    "ObjectWithDepth",
    "SceneDescription",
    "RobotState",
    "WorldModel",
    "ToolCall",
    "MissionPlan",
    "ExecutionStatus",
    # Services
    "RobotStateAggregatorService",
    "WorldModelAggregatorService",
    # Configuration
    "get_settings",
    "PeaceSettings",
    # Exceptions
    "PeaceError",
    "SafetyViolationError",
    "InferenceError",
    "RobotError",
    "ConfigurationError",
]


def __getattr__(name: str):
    """Lazily import and return public symbols to avoid heavy top-level imports."""
    if name == "VisionNode":
        from peace.nodes.vision import VisionNode
        return VisionNode
    if name == "AgentNode":
        from peace.nodes.planner_executor import AgentNode
        return AgentNode

    if name in {"Object", "ObjectWithDepth", "SceneDescription", "RobotState", "WorldModel",
                "ToolCall", "MissionPlan", "ExecutionStatus"}:
        import peace.schemas as _s
        return getattr(_s, name)

    if name == "RobotStateAggregatorService":
        from peace.services.robot_state_service import RobotStateAggregatorService
        return RobotStateAggregatorService
    if name == "WorldModelAggregatorService":
        from peace.services.world_model_service import WorldModelAggregatorService
        return WorldModelAggregatorService

    if name == "get_settings":
        from peace.core.config import get_settings
        return get_settings
    if name == "PeaceSettings":
        from peace.core.config import PeaceSettings
        return PeaceSettings

    if name in {"PeaceError", "SafetyViolationError", "InferenceError",
                "RobotError", "ConfigurationError"}:
        import peace.exceptions as _e
        return getattr(_e, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
