"""
Configuration settings for PEACE using pydantic-settings.
"""

import requests

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PeaceSettings(BaseSettings):
    """
    Configuration settings (from .env at project root) for the PEACE.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # DEPLOYMENT CONFIGURATION

    # Agent Configuration
    agent_name: str = Field(
        default="PEACE",
        description="Name of the agent",
    )
    agent_max_replan_depth: int = Field(
        default=2,
        description="Maximum number of full-replan cycles per mission (prevents infinite replan loops)",
    )
    agent_enable_memory: bool = Field(
        default=True,
        description="Include world model history in LLM prompt for temporal context",
    )
    agent_memory_window: int = Field(
        default=6,
        description="Number of past world model snapshots to include in LLM prompt",
    )

    # Ollama Configuration
    ollama_endpoint: str = Field(
        default="http://host.docker.internal:11434",
        description="Ollama API endpoint URL",
    )
    ollama_api_key: str = Field(
        default="",
        description="API key for Ollama cloud (leave empty for local instances)",
    )

    # Vision Node Configuration
    vision_model: str = Field(
        default="llava:13b",
        description="Ollama model to use for vision/scene interpretation",
    )
    vision_model_temperature: float = Field(
        default=0.1,
        description="Temperature for vision model (0.0-2.0, lower = more deterministic)",
    )
    vision_model_num_predict: int = Field(
        default=800,
        description="Maximum number of tokens to predict for vision model",
    )
    vision_model_timeout: int = Field(
        default=120,
        description="Timeout for vision model API calls (seconds)",
    )

    # YOLO Configuration
    use_yolo: bool = Field(
        default=False,
        description="Use YOLO object detection instead of VLM (faster, more accurate bboxes)",
    )
    yolo_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model name or path to a .pt file. Auto-downloaded if just a model name (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt).",
    )
    yolo_device: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda', or 'mps' (Apple Silicon)",
    )
    yolo_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence threshold for YOLO detections (0.0-1.0)",
    )

    # Language Node Configuration
    language_model: str = Field(
        default="qwen2.5:14b",
        description="Ollama model to use for language/reasoning",
    )
    language_model_temperature: float = Field(
        default=0.7,
        description="Temperature for language model (0.0-2.0, lower = more deterministic)",
    )
    language_model_num_predict: int = Field(
        default=2000,
        description="Maximum number of tokens to predict for language model",
    )
    language_model_timeout: int = Field(
        default=120,
        description="Timeout for language model API calls (seconds)",
    )

    # ROS2 Node Configuration
    node_namespace: str = Field(
        default="peace",
        description="ROS2 namespace for the agent's nodes",
    )

    # Topic Configuration
    topic_namespace: str = Field(
        default="/mavros",
        description="MAVROS topic namespace",
    )
    topic_imu: str = Field(
        default="/mavros/imu/data",
        description="IMU data topic",
    )
    topic_altitude: str = Field(
        default="/mavros/altitude",
        description="Altitude data topic",
    )
    topic_odom: str = Field(
        default="/mavros/local_position/odom",
        description="Odometry topic",
    )
    topic_global_position: str = Field(
        default="/mavros/global_position/global",
        description="GPS global position topic",
    )
    topic_camera_info: str = Field(
        default="/camera/color/camera_info",
        description="RGB camera camera info topic",
    )
    topic_camera_color_image: str = Field(
        default="/camera/color/image",
        description="RGB camera image topic",
    )
    topic_camera_depth_image: str = Field(
        default="/camera/depth/image",
        description="Depth camera image topic",
    )
    topic_camera_depth_points: str = Field(
        default="/camera/depth/points",
        description="Depth camera point cloud topic",
    )
    topic_lidar_scan: str = Field(
        default="/lidar/scan",
        description="LiDAR scan topic",
    )

    # RUNTIME CONFIGURATION
    # These parameters control node behavior and can be overridden via environment variables

    # Vision Node Runtime Parameters
    vision_inference_rate: float = Field(
        default=0.2,
        description="Vision inference rate in Hz (how often to run detection)",
    )
    vision_confidence_threshold: float = Field(
        default=0.6,
        description="Confidence threshold for vision detections (0.0-1.0)",
    )
    max_depth: float = Field(
        default=100.0,
        description="Maximum depth distance in meters",
    )
    min_depth: float = Field(
        default=0.1,
        description="Minimum depth distance in meters",
    )

    # Data Logging
    log_enable: bool = Field(
        default=True,
        description="Enable CSV data logging for task, world model, decisions, and reasoning",
    )
    log_output_dir: str = Field(
        default="./logs",
        description="Directory for CSV log files (created automatically)",
    )

    # Debug
    debug_save_images: bool = Field(
        default=False,
        description="Save annotated detection images to disk for debugging",
    )
    debug_output_dir: str = Field(
        default="./debug_images",
        description="Directory to write annotated debug images (created automatically)",
    )

    # Language Node Runtime Parameters
    language_context_window: int = Field(
        default=10,
        description="Number of world model snapshots to maintain in context window",
    )
    language_max_detections: int = Field(
        default=10,
        description="Maximum number of object detections to include in context",
    )

    # Action Node Runtime Parameters
    action_enable_control: bool = Field(
        default=False,
        description="Enable flight control commands (safety override, disabled by default)",
    )
    action_waypoint_tolerance: float = Field(
        default=0.5,
        description="Distance tolerance for waypoint completion in meters",
    )
    action_max_altitude: float = Field(
        default=50.0,
        description="Maximum altitude limit in meters",
    )
    action_min_altitude: float = Field(
        default=0.5,
        description="Minimum altitude limit in meters",
    )
    action_horizontal_limit: float = Field(
        default=500.0,
        description="Maximum horizontal distance from home in meters (±x and ±y)",
    )
    action_command_timeout: float = Field(
        default=30.0,
        description="Timeout for command execution in seconds",
    )
    action_setpoint_stream_rate: float = Field(
        default=10.0,
        description="Setpoint streaming rate in Hz (must be >= 2 for OFFBOARD mode)",
    )
    action_status_timer_interval: float = Field(
        default=5.0,
        description="Interval in seconds for periodic status updates",
    )

    # VALIDATORS & UTILITIES

    @field_validator("ollama_endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Ensure endpoint starts with http:// or https://."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama endpoint must start with http:// or https://")
        return v.rstrip("/")

    def validate_ollama_connection(self) -> bool:
        """Validate that Ollama is accessible."""
        try:
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


@lru_cache(maxsize=None)
def get_settings() -> PeaceSettings:
    """Return the singleton settings instance, creating it on first call."""
    return PeaceSettings()
