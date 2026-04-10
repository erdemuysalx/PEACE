"""
Agent node — implementing planner and executor pattern.

The LLM outputs a full MissionPlan (List[ToolCall]) in a single call;
DroneTools executes the steps deterministically with no LLM in the execution loop.
"""

import concurrent.futures
import json
import threading
import time
import uuid
from typing import Optional

import rclpy
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from robot_agent.backends.llm import OllamaClient
from robot_agent.core.config import get_settings
from robot_agent.exceptions import InferenceError
from robot_agent.services.drone_tools import DroneTools
from robot_agent.system_prompts import LANGUAGE_SYSTEM_PROMPT, format_world_context
from robot_agent.schemas import RobotState, ExecutionStatus, MissionPlan, ToolCall, WorldModel
from robot_agent.services.data_logger_service import DataLoggerService
from robot_agent.services.robot_state_service import RobotStateAggregatorService
from robot_agent.services.safety_service import SafetyValidatorService
from robot_agent.services.world_model_service import WorldModelAggregatorService


class AgentNode(Node):
    """
    Mission planning and flight execution node using the planner-executor pattern.

    Subscribes to:
      - MAVROS state, odometry, LiDAR — forwarded to RobotStateAggregatorService /
        WorldModelAggregatorService respectively
      - /agent/vision/detections — object detections from VisionNode
      - /agent/query             — natural-language mission queries from the CLI

    Publishes:
      - /agent/execution_status  — JSON-serialised ExecutionStatus for the CLI

    On each query the LLM produces a full MissionPlan (list of ToolCalls) in a
    single inference call. DroneTools then executes each step deterministically
    with no LLM involvement. If a step fails and the replan budget allows,
    a new plan is generated for the remaining objectives (_replan_remaining).

    Telemetry callbacks delegate immediately to the appropriate aggregator service
    so no business logic lives inside ROS callbacks.
    """

    def __init__(self) -> None:
        """Set up LLM backend, services, subscribers, publishers, and tools."""
        super().__init__("agent_node")
        self.settings = get_settings()

        if not self.settings.action_enable_control:
            self.get_logger().warning(
                "Control disabled — set ACTION_ENABLE_CONTROL=True to enable flight commands"
            )

        # ── Execution state ──────────────────────────────────────────────────
        self._lock = threading.RLock()
        self._executing = False
        self._prev_mode: Optional[str] = None

        # ── Worker pool for planner + executor) ──────────────────────────────
        self._worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._inference_running = False

        # ── LLM backend ──────────────────────────────────────────────────────
        self.llm = OllamaClient(
            endpoint=self.settings.ollama_endpoint,
            model=self.settings.language_model,
            temperature=self.settings.language_model_temperature,
            num_predict=self.settings.language_model_num_predict,
            timeout=self.settings.language_model_timeout,
            api_key=self.settings.ollama_api_key,
        )
        if not self.llm.is_ready:
            available = ", ".join(self.llm.available_models) or "none"
            self.get_logger().error(
                f"LLM '{self.settings.language_model}' not found at "
                f"{self.settings.ollama_endpoint}. Available: {available}"
            )

        # ── Services ─────────────────────────────────────────────────────────
        self.ego_state_svc = RobotStateAggregatorService(
            context_window=self.settings.language_context_window,
        )
        self.world_model_svc = WorldModelAggregatorService(
            context_window=self.settings.language_context_window,
            max_detections=self.settings.language_max_detections,
        )
        self.safety_svc = SafetyValidatorService(
            min_altitude=self.settings.action_min_altitude,
            max_altitude=self.settings.action_max_altitude,
            horizontal_limit=self.settings.action_horizontal_limit,
        )
        self.data_logger = DataLoggerService(
            output_dir=self.settings.log_output_dir,
            enabled=self.settings.log_enable,
        )

        # ── QoS profiles ─────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Callback groups ──────────────────────────────────────────────────
        reentrant_cb = ReentrantCallbackGroup()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            State,
            self.settings.topic_namespace + "/state",
            self._state_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            Odometry,
            self.settings.topic_odom,
            self._odom_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            LaserScan,
            self.settings.topic_lidar_scan,
            self._lidar_cb,
            best_effort_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            String,
            "/agent/vision/detections",
            self._detections_cb,
            reliable_qos,
            callback_group=reentrant_cb,
        )
        self.create_subscription(
            String, "/agent/query", self._query_cb, reliable_qos, callback_group=reentrant_cb
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/agent/execution_status", reliable_qos)

        # Periodic status timer
        self.create_timer(self.settings.action_status_timer_interval, self._status_timer_cb)

        # ── DroneTools (owns MAVROS publisher + service clients) ──────────────
        self.tools = DroneTools(
            self, self.ego_state_svc, self.world_model_svc, self.safety_svc, self.settings
        )

        self.get_logger().info(
            f"Agent Node ready | model={self.settings.language_model} | "
            f"control={'enabled' if self.settings.action_enable_control else 'disabled'} | "
            f"alt=[{self.settings.action_min_altitude}, {self.settings.action_max_altitude}]m"
        )

    def destroy_node(self) -> None:
        """Stop the tools, flush the data logger, and destroy the ROS node."""
        self.tools.stop_safely()
        self.data_logger.close()
        super().destroy_node()

    # ── Telemetry callbacks ───────────────────────────────────────────────────

    def _state_cb(self, msg: State) -> None:
        """Forward MAVROS State to DroneTools and RobotStateAggregatorService."""
        self.tools.update_state(msg)
        self.ego_state_svc.update_flight_mode(msg.mode, msg.armed, msg.connected)

    def _odom_cb(self, msg: Odometry) -> None:
        """Forward full odometry (pose + twist + orientation) to RobotStateAggregatorService."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.ego_state_svc.update_odom(
            p.x, p.y, p.z,
            v.x, v.y, v.z,
            q.x, q.y, q.z, q.w,
        )

    def _lidar_cb(self, msg: LaserScan) -> None:
        """Forward LiDAR scan to the world model service for sector parsing."""
        self.world_model_svc.update_lidar(
            list(msg.ranges),
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
        )

    def _detections_cb(self, msg: String) -> None:
        """Parse JSON detections from the vision node and update the world model."""
        try:
            data = json.loads(msg.data)
            objects = data.get("objects", [])
            detections = []
            for obj in objects:
                world_pos = obj.get("world_pos", [0.0, 0.0, 0.0])
                detections.append(
                    {
                        "id": obj.get("id", -1),
                        "label": obj.get("label", "unknown"),
                        "confidence": obj.get("confidence", 0) / 100.0,
                        "world_x": world_pos[0] if len(world_pos) > 0 else 0.0,
                        "world_y": world_pos[1] if len(world_pos) > 1 else 0.0,
                        "world_z": world_pos[2] if len(world_pos) > 2 else 0.0,
                        "depth_meters": obj.get("depth_meters", 0.0),
                        "depth_valid": obj.get("depth_valid", False),
                    }
                )
            self.world_model_svc.update_detections(detections)
            self.get_logger().debug(
                f"Detections updated: {len(detections)} objects — "
                + ", ".join(
                    f"{d['label']}(depth_valid={d['depth_valid']}, "
                    f"pos=({d['world_x']:.1f},{d['world_y']:.1f}))"
                    for d in detections
                )
            )
        except Exception as e:
            self.get_logger().error(f"Failed to parse detections: {e}")

    # ── Periodic status timer ─────────────────────────────────────────────────

    def _status_timer_cb(self) -> None:
        """Log a flight-mode change if the mode has changed since the last tick."""
        ego = self.ego_state_svc.get_ego_state()
        mode = ego.flight_mode or "UNKNOWN"
        if mode != self._prev_mode:
            self.get_logger().info(f"Mode: {self._prev_mode} -> {mode}")
            self._prev_mode = mode

    # ── Query handling ────────────────────────────────────────────────────────

    def _query_cb(self, msg: String) -> None:
        """Accept an incoming query and dispatch it to the worker pool if idle."""
        query = msg.data.strip()
        if not query:
            return
        with self._lock:
            if self._inference_running or self._executing:
                self.get_logger().warning("Query dropped — inference or execution already in progress")
                return
            if not self.llm.is_ready:
                self.get_logger().error("LLM not available — check Ollama connection")
                return
            self._inference_running = True

        self.get_logger().info(f"Query: {query}")
        self._worker_pool.submit(self._process_query, query)

    # ── Planner-executor pattern ────────────────────────────────────────────────

    def _process_query(self, user_query: str) -> None:
        """Call the LLM to produce a MissionPlan, then hand off to _execute_plan."""
        mission_id = ""
        ego_state = None
        world_model = None
        try:
            ego_state = self.ego_state_svc.get_ego_state()
            world_model = self.world_model_svc.get_world_model()
            ego_history = (
                self.ego_state_svc.get_ego_state_history(self.settings.agent_memory_window)
                if self.settings.agent_enable_memory
                else None
            )
            context = format_world_context(ego_state, world_model, ego_history)
            messages = [
                {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"{user_query}\n\n{context}"},
            ]
            mission_id = str(uuid.uuid4())
            self._publish_execution_status(mission_id, "executing", "Planning mission")
            raw = self.llm.chat(messages=messages, format=MissionPlan.model_json_schema())
            plan = MissionPlan.model_validate_json(raw)
            plan.user_query = user_query
            plan.mission_id = mission_id
            if not plan.steps:
                self._publish_execution_status(mission_id, "failed", "LLM returned empty plan")
                self.data_logger.log_mission(
                    mission_id, user_query, "", [], "failed", ego_state, world_model
                )
                with self._lock:
                    self._inference_running = False
                return
            self._publish_execution_status(
                mission_id, "executing", "Executing plan", reasoning=plan.reasoning
            )
            with self._lock:
                self._executing = True
                self._inference_running = False
            self._worker_pool.submit(self._execute_plan, plan, ego_state, world_model)
        except InferenceError as e:
            self.get_logger().error(f"LLM inference failed: {e}")
            self._publish_execution_status("", "failed", str(e))
            self.data_logger.log_mission(
                mission_id, user_query, "", [], "failed", ego_state, world_model
            )
            with self._lock:
                self._inference_running = False
        except Exception as e:
            self.get_logger().error(f"Query processing error: {type(e).__name__}: {e}")
            self._publish_execution_status("", "failed", str(e))
            self.data_logger.log_mission(
                mission_id, user_query, "", [], "failed", ego_state, world_model
            )
            with self._lock:
                self._inference_running = False

    # ── Plan execution ────────────────────────────────────────────────────────

    def _execute_plan(
        self,
        plan: MissionPlan,
        ego_state: RobotState,
        world_model: WorldModel,
        replan_depth: int = 0,
    ) -> None:
        """Execute plan steps sequentially, triggering a replan on step failure if allowed."""
        mission_id = plan.mission_id
        steps = list(plan.steps)
        completed: list[ToolCall] = []
        executed_steps: list[dict] = []
        final_state = "failed"
        try:
            for i, step in enumerate(steps):
                self._publish_execution_status(
                    mission_id, "executing",
                    f"Step {i + 1}/{len(steps)}: {step.tool}",
                )
                result = self.tools.execute(step.tool, step.args)
                self.get_logger().info(f"[STEP {i + 1}/{len(steps)}] {step.tool} -> {result}")
                executed_steps.append(
                    {"tool": step.tool, "args": step.args, "result": result, "reasoning": step.reasoning}
                )
                if "failed" in result.lower() or result.startswith("ERROR"):
                    if replan_depth < self.settings.agent_max_replan_depth:
                        new_plan = self._replan_remaining(plan, completed, result)
                        if new_plan:
                            final_state = "replanning"
                            self._execute_plan(
                                new_plan, ego_state, world_model, replan_depth + 1
                            )
                            return
                    self._publish_execution_status(mission_id, "failed", result)
                    return
                completed.append(step)
            self._publish_execution_status(mission_id, "completed", "Mission complete")
            final_state = "completed"
        except Exception as e:
            self.get_logger().error(f"Plan execution error: {e}")
            self._publish_execution_status(mission_id, "failed", str(e))
        finally:
            self.data_logger.log_mission(
                mission_id, plan.user_query or "", plan.reasoning,
                executed_steps, final_state, ego_state, world_model,
            )
            if final_state != "replanning":
                with self._lock:
                    self._executing = False
                self.tools.stop_safely()

    def _replan_remaining(
        self,
        plan: MissionPlan,
        completed: list[ToolCall],
        failure_reason: str,
    ) -> Optional[MissionPlan]:
        """
        Issue a replanning LLM call describing the mission state and failure reason.

        Returns a new MissionPlan for the remaining objectives, or None on error.
        """
        ego_state = self.ego_state_svc.get_ego_state()
        world_model = self.world_model_svc.get_world_model()
        context = format_world_context(ego_state, world_model)

        completed_summary = (
            "\n".join(f"  - {s.tool}({s.args}): {s.reasoning}" for s in completed)
            if completed
            else "  None"
        )
        remaining_summary = (
            "\n".join(
                f"  - {s.tool}({s.args}): {s.reasoning}"
                for s in plan.steps
                if s not in completed
            )
            if plan.steps
            else "  None"
        )

        replan_user = (
            f"Original query: {plan.user_query or ''}\n\n"
            f"Completed steps:\n{completed_summary}\n\n"
            f"Remaining steps (not yet executed):\n{remaining_summary}\n\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Current world context:\n{context}\n\n"
            "Produce a revised MissionPlan for the remaining objectives only. "
            "Return ONLY the JSON mission plan, no additional text."
        )
        try:
            raw = self.llm.chat(
                messages=[
                    {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": replan_user},
                ],
                format=MissionPlan.model_json_schema(),
            )
            new_plan = MissionPlan.model_validate_json(raw)
            new_plan.mission_id = new_plan.mission_id or str(uuid.uuid4())
            new_plan.user_query = new_plan.user_query or plan.user_query
            self.get_logger().info(
                f"[REPLAN REASONING] {new_plan.reasoning}"
            )
            self.get_logger().info(
                f"Replanned mission: {len(new_plan.steps)} step(s)"
            )
            return new_plan
        except Exception as e:
            self.get_logger().error(f"Replanning failed: {type(e).__name__}: {e}")
            return None

    # ── Status publishing ─────────────────────────────────────────────────────

    def _publish_execution_status(
        self, mission_id: str, state: str, message: str, reasoning: str = ""
    ) -> None:
        """Publish an execution status update."""
        status = ExecutionStatus(
            mission_id=mission_id,
            current_waypoint_index=0,
            total_waypoints=0,
            state=state,
            position_error=0.0,
            message=message,
            timestamp=time.time(),
            reasoning=reasoning,
        )
        msg = String()
        msg.data = status.model_dump_json()
        self._status_pub.publish(msg)


def main(args=None) -> None:
    """Initialise ROS2, spin the AgentNode, and shut down on exit."""
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
