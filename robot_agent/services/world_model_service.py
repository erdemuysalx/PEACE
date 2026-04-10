"""World model aggregation service — thread-safe environment observation collection."""

import math
import threading
from collections import deque

import numpy as np

from robot_agent.schemas import WorldModel


class WorldModelAggregatorService:
    """
    Thread-safe aggregator for environment observations (detections, obstacles).

    Nodes call the update_*() methods from ROS2 callbacks.
    The agent calls get_world_model() to obtain a complete snapshot for LLM reasoning.

    A single _world_model_lock covers the WorldModel (including detections and lidar_ranges)
    and the history deque.
    """

    # Sector definitions: name → (angle_min_deg, angle_max_deg)
    # 0° = forward, positive CCW when viewed from above
    LIDAR_SECTORS: dict[str, tuple[float, float]] = {
        "front":       (-22.5,   22.5),
        "front_right": (-67.5,  -22.5),
        "right":       (-112.5, -67.5),
        "back_right":  (-157.5, -112.5),
        "back":        (157.5, -157.5),   # wraps: start=157.5°, end=-157.5° across ±180°
        "back_left":   (112.5,  157.5),
        "left":        (67.5,   112.5),
        "front_left":  (22.5,   67.5),
    }

    def __init__(self, context_window: int = 10, max_detections: int = 10) -> None:
        """Initialise the world model snapshot, bounded history deque, and thread lock."""
        self._max_detections = max_detections
        self._world_model = WorldModel()
        self._history: deque[WorldModel] = deque(maxlen=context_window)
        self._world_model_lock = threading.Lock()

    # ── Update methods (called from ROS2 callbacks) ──────────────────────────

    def update_lidar(
        self,
        ranges: list[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
    ) -> None:
        """Parse LiDAR scan into named sectors and store min distances."""
        arr = np.array(ranges, dtype=np.float32)
        valid = np.isfinite(arr) & (arr >= range_min) & (arr <= range_max)
        if not np.any(valid):
            return

        num = len(arr)
        angles = angle_min + np.arange(num) * angle_increment

        new_ranges: dict[str, float] = {}
        for name, (a_start_deg, a_end_deg) in self.LIDAR_SECTORS.items():
            a_start = math.radians(a_start_deg)
            a_end = math.radians(a_end_deg)
            if name == "back":  # sector wraps around ±π: start > end
                mask = valid & ((angles >= a_start) | (angles <= a_end))
            else:
                mask = valid & (angles >= a_start) & (angles < a_end)
            new_ranges[name] = float(np.min(arr[mask])) if np.any(mask) else float("inf")

        new_ranges["min_range"] = float(np.min(arr[valid]))

        with self._world_model_lock:
            self._world_model.lidar_ranges = new_ranges

    def update_detections(self, detections: list[dict]) -> None:
        """Store the latest object detections (capped at max_detections) and append a snapshot to history."""
        with self._world_model_lock:
            self._world_model.detections = detections[: self._max_detections]
            self._history.append(self._world_model.model_copy())

    # ── Query methods (called from agent / LLM thread) ───────────────────────

    def get_world_model(self) -> WorldModel:
        """Return a complete snapshot of the world model including detections and lidar."""
        with self._world_model_lock:
            return self._world_model.model_copy()

    def get_world_model_history(self, n: int) -> list[WorldModel]:
        """Return the last n detection-stamped world model snapshots, oldest first."""
        with self._world_model_lock:
            history = list(self._history)
        return history[-n:] if n < len(history) else history
