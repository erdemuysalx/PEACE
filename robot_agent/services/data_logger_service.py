"""Data logging service — appends structured mission data to a CSV file.

A single fixed file (agent_log.csv) is appended across all launches. One row
is written per mission at completion, capturing the query, plan, execution
steps, final state, ego state, and world model observed at query-receive time.
"""

import csv
import json
import os
import threading
from typing import Optional

from robot_agent.schemas import RobotState, WorldModel

# CSV column order — fixed for easy pandas/numpy import
_COLUMNS = [
    "mission_id",
    "query",
    "plan_reasoning",
    "steps_json",
    "final_state",
    "ego_state_json",
    "world_model_json",
]


class DataLoggerService:
    """
    Thread-safe CSV logger for mission data collection.

    All public methods are safe to call from multiple threads simultaneously.
    """

    def __init__(self, output_dir: str, enabled: bool = True) -> None:
        """Open (or create) the CSV log file and write the header if the file is empty."""
        self._enabled = enabled
        self._lock = threading.Lock()
        self._file = None
        self._writer = None

        if not enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "agent_log.csv")
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

        self._file = open(log_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    # ── Public log method ─────────────────────────────────────────────────────

    def log_mission(
        self,
        mission_id: str,
        query: str,
        plan_reasoning: str,
        executed_steps: list[dict],
        final_state: str,
        ego_state: Optional[RobotState],
        world_model: Optional[WorldModel],
    ) -> None:
        """Log the complete outcome of a mission as a single row."""
        if not self._enabled or self._writer is None:
            return

        row = {
            "mission_id": mission_id,
            "query": query,
            "plan_reasoning": plan_reasoning,
            "steps_json": json.dumps(executed_steps),
            "final_state": final_state,
            "ego_state_json": ego_state.model_dump_json() if ego_state else "",
            "world_model_json": world_model.model_dump_json() if world_model else "",
        }

        with self._lock:
            if self._writer is not None:
                self._writer.writerow(row)
                self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file. Called on node shutdown."""
        with self._lock:
            if self._file is None:
                return
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
