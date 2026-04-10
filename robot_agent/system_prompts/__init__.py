"""
Prompt templates for Robot Agent.

System prompts are stored as .txt files under system_prompts/ and loaded at
import time. Helper functions format live ego state and world model data into
context strings.
"""

from pathlib import Path

from robot_agent.schemas import RobotState, WorldModel

_PROMPTS_DIR = Path(__file__).parent


def _load(name: str) -> str:
    """Read and return a prompt file from the system_prompts directory."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


VISION_SYSTEM_PROMPT = _load("vision.txt")

LANGUAGE_SYSTEM_PROMPT = _load("language.txt")


# HELPER FUNCTIONS


def get_perception_prompt() -> str:
    """Get the vision/perception system prompt for VLM inference.

    Returns:
        Vision system prompt string
    """
    return VISION_SYSTEM_PROMPT


def _format_detections(detections: list) -> str:
    """Format a list of detection dicts into a human-readable summary."""
    if not detections:
        return "No objects detected"
    lines = []
    for i, det in enumerate(detections[:10], 1):
        depth_status = (
            "valid position"
            if det.get("depth_valid", False)
            else "position UNCERTAIN (depth sensor failure)"
        )
        lines.append(
            f"{i}. {det.get('label', 'unknown')} at "
            f"({det.get('world_x', 0):.1f}m, {det.get('world_y', 0):.1f}m) - "
            f"{depth_status}, confidence={det.get('confidence', 0):.0%}"
        )
    return "\n".join(lines)


def _format_lidar(lidar_ranges: dict) -> str:
    """Format LiDAR sector distances into a human-readable summary."""
    if not lidar_ranges:
        return "No LiDAR data available"
    min_range = lidar_ranges.get("min_range", float("inf"))
    lines = [
        (
            f"Closest obstacle: {min_range:.1f}m"
            if min_range != float("inf")
            else "No obstacles detected"
        ),
        f"- Front: {lidar_ranges.get('front', float('inf')):.1f}m",
        f"- Front-Right: {lidar_ranges.get('front_right', float('inf')):.1f}m",
        f"- Right: {lidar_ranges.get('right', float('inf')):.1f}m",
        f"- Back-Right: {lidar_ranges.get('back_right', float('inf')):.1f}m",
        f"- Back: {lidar_ranges.get('back', float('inf')):.1f}m",
        f"- Back-Left: {lidar_ranges.get('back_left', float('inf')):.1f}m",
        f"- Left: {lidar_ranges.get('left', float('inf')):.1f}m",
        f"- Front-Left: {lidar_ranges.get('front_left', float('inf')):.1f}m",
    ]
    return "\n".join(lines)


def _format_memory_block(history: list[RobotState]) -> str:
    """
    Format past ego state snapshots into a compact temporal context block.

    Each row shows position, yaw, flight mode, and armed status so the LLM
    can reason about motion and state changes over time.
    """
    if not history:
        return ""
    lines = ["--- Ego State History (oldest → newest) ---"]
    for i, snap in enumerate(history):
        lines.append(
            f"t-{len(history) - i}  "
            f"pos=({snap.position_x:.1f},{snap.position_y:.1f},{snap.position_z:.1f})m  "
            f"yaw={snap.orientation_yaw:.0f}°  "
            f"mode={snap.flight_mode or '?'}  "
            f"armed={snap.armed}"
        )
    lines.append("--- End of History ---")
    return "\n".join(lines)


def _format_ego_block(ego_state: RobotState) -> str:
    """Format ego-vehicle state (position, orientation, flight status) into a compact block."""
    return (
        f"Position: ({ego_state.position_x:.2f}m, {ego_state.position_y:.2f}m, {ego_state.position_z:.2f}m), "
        f"Yaw={ego_state.orientation_yaw:.1f}°\n"
        f"Flight Mode: {ego_state.flight_mode or 'UNKNOWN'}, Armed: {ego_state.armed}"
    )


def _format_world_block(world_model: WorldModel) -> str:
    """Format world observations (detections, LiDAR) from a WorldModel into a compact block."""
    detections_summary = _format_detections(world_model.detections)
    lidar_summary = _format_lidar(world_model.lidar_ranges or {})
    return (
        f"Detected Objects:\n{detections_summary}\n\n"
        f"Obstacle Distances (LiDAR):\n{lidar_summary}"
    )


def format_world_context(
    ego_state: RobotState,
    world_model: WorldModel,
    history: list[RobotState] | None = None,
) -> str:
    """
    Format current ego state and world model into a context string for the LLM.

    Args:
        ego_state: Current ego-vehicle state snapshot.
        world_model: Current world observation snapshot.
        history: Optional list of past ego state snapshots for temporal context.

    Returns:
        Formatted context string combining ego, world, and optional memory blocks.
    """
    ego_block = _format_ego_block(ego_state)
    world_block = _format_world_block(world_model)
    context = f"{ego_block}\n\n{world_block}"
    if history:
        memory = _format_memory_block(history)
        return f"{context}\n\n{memory}"
    return context
