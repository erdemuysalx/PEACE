"""Platform-specific tool sets for the planner-executor.

Each platform exposes a concrete subclass of :class:`RobotTools` whose ``@tool``
decorated methods are dispatchable by the LLM via :meth:`RobotTools.execute`.
"""

from peace.platforms.uav_tools import UAVTools
from peace.platforms.robot_tools import RobotTools, tool

__all__ = ["UAVTools", "RobotTools", "tool"]
