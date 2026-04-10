"""
Robot Agent — agentic control architecture for robotic platforms, built on ROS2 and leveraging large models for decision-making.
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
    "RobotAgentSettings",
    # Exceptions
    "RobotAgentError",
    "SafetyViolationError",
    "InferenceError",
    "VehicleError",
    "ConfigurationError",
]


def __getattr__(name: str):
    """Lazily import and return public symbols to avoid heavy top-level imports."""
    if name == "VisionNode":
        from robot_agent.nodes.vision import VisionNode
        return VisionNode
    if name == "AgentNode":
        from robot_agent.nodes.planner_executor import AgentNode
        return AgentNode

    if name in {"Object", "ObjectWithDepth", "SceneDescription", "RobotState", "WorldModel",
                "ToolCall", "MissionPlan", "ExecutionStatus"}:
        import robot_agent.schemas as _s
        return getattr(_s, name)

    if name == "RobotStateAggregatorService":
        from robot_agent.services.robot_state_service import RobotStateAggregatorService
        return RobotStateAggregatorService
    if name == "WorldModelAggregatorService":
        from robot_agent.services.world_model_service import WorldModelAggregatorService
        return WorldModelAggregatorService

    if name == "get_settings":
        from robot_agent.core.config import get_settings
        return get_settings
    if name == "RobotAgentSettings":
        from robot_agent.core.config import RobotAgentSettings
        return RobotAgentSettings

    if name in {"RobotAgentError", "SafetyViolationError", "InferenceError",
                "VehicleError", "ConfigurationError"}:
        import robot_agent.exceptions as _e
        return getattr(_e, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
