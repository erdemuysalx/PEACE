"""Command-line interface for Robot Agent."""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from std_msgs.msg import String

import robot_agent
from robot_agent.core.config import get_settings

_ACCENT = "#ffb86c"

_TERMINAL_STATES = frozenset({"completed", "aborted", "failed"})
_STATE_STYLE = {
    "completed": "green",
    "failed": "red",
    "aborted": "red",
    "executing": "dim white",
    "idle": "dim white",
    "paused": "yellow",
}

_console = Console()

_DRONE_ART = """\
  o     o
   \\   /
  --+--
   /   \\
  o     o"""


def _print_welcome(model: str) -> None:
    """Print the Claude Code-style welcome panel."""
    left = Text()
    left.append("\n  Welcome to Robot Agent!\n\n", style=f"bold {_ACCENT}")
    left.append(_DRONE_ART + "\n\n", style=_ACCENT)
    left.append(f"  {model}\n", style="dim")
    left.append("  ROS 2 / MAVROS\n", style="dim")

    right = Text()
    right.append("\n  Tips for getting started\n", style=f"bold {_ACCENT}")
    right.append("  Describe your mission in plain language.\n", style="")
    right.append('  Try: "take off to 5m and hover for 10s"\n\n', style="dim")
    right.append("  Shortcuts\n", style=f"bold {_ACCENT}")
    right.append("  exit / q    quit the session\n", style="")
    right.append("  Ctrl+C      interrupt current mission\n", style="")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(left, right)

    _console.print(
        Panel(
            grid,
            border_style=_ACCENT,
            title=f"[bold {_ACCENT}]Robot Agent v{robot_agent.__version__}[/bold {_ACCENT}]",
            title_align="left",
            padding=(0, 1),
        )
    )
    _console.print(f"[dim]  ? for shortcuts · type exit to quit[/dim]\n")


class AgentCLI(Node):
    """CLI client — publishes queries and displays execution status updates."""

    def __init__(self) -> None:
        """Create the query publisher, status subscriber, and mission state fields."""
        super().__init__("agent_cli")

        self._status_sub = self.create_subscription(
            String, "/agent/execution_status", self._status_cb, 10
        )
        self._query_pub = self.create_publisher(String, "/agent/query", 10)

        self._mission_done = False
        self._current_step = "Planning..."
        self._terminal_state: str = ""
        self._terminal_message: str = ""

        # Brief pause so publishers register before first message
        time.sleep(0.5)

    def _status_cb(self, msg: String) -> None:
        """Parse an ExecutionStatus JSON message and update mission state flags."""
        try:
            status = json.loads(msg.data)
            state = status.get("state", "unknown")
            message = status.get("message", "")
            reasoning = status.get("reasoning", "")
            if reasoning:
                _console.print(f"[dim italic]  {reasoning}[/dim italic]")
            self._current_step = message or state
            if state in _TERMINAL_STATES:
                self._terminal_state = state
                self._terminal_message = message
                self._mission_done = True
        except Exception as e:
            self.get_logger().debug(f"Failed to parse status: {e}")

    def send_query(self, query: str) -> bool:
        """Publish a query string to the agent and reset mission state. Returns False if empty."""
        if not query.strip():
            _console.print("[red]Query cannot be empty[/red]")
            return False
        self._mission_done = False
        self._terminal_state = ""
        self._terminal_message = ""
        self._current_step = "Planning..."
        msg = String()
        msg.data = query
        self._query_pub.publish(msg)
        return True

    def wait_for_completion(self, timeout: float = 300.0) -> None:
        """Block until a terminal status is received or timeout expires."""
        deadline = time.time() + timeout
        with Live(console=_console, refresh_per_second=10, transient=True) as live:
            while not self._mission_done and time.time() < deadline:
                live.update(Spinner("dots", text=f" {self._current_step}"))
                rclpy.spin_once(self, timeout_sec=0.1)

        if not self._mission_done:
            _console.print("[yellow]No completion status received — returning to prompt[/yellow]")
            return

        style = _STATE_STYLE.get(self._terminal_state, "white")
        _console.print(Text(f"  {self._terminal_message}", style=style))


def main(args=None) -> None:
    """Entry point for interactive and single-query CLI modes."""
    parser = argparse.ArgumentParser(description="Robot Agent CLI")
    parser.add_argument("--query", nargs="*", help="Query to send (omit for interactive mode)")
    parser.add_argument(
        "--monitor", action="store_true", help="Keep running to monitor execution status"
    )

    # Separate CLI args from ROS2 args
    cli_args: list[str] = []
    ros_args: list[str] = []
    in_ros = False
    for arg in sys.argv[1:]:
        if arg == "--ros-args":
            in_ros = True
        (ros_args if in_ros else cli_args).append(arg)

    parsed = parser.parse_args(cli_args)

    rclpy.init(args=ros_args or None)
    cli = AgentCLI()

    try:
        if parsed.query:
            query_text = " ".join(parsed.query)
            _console.print(f"\n[dim]{query_text}[/dim]")
            if cli.send_query(query_text):
                if parsed.monitor:
                    _console.print("[dim]Monitoring execution (Ctrl+C to exit)...[/dim]")
                    rclpy.spin(cli)
                else:
                    cli.wait_for_completion()
        else:
            model = get_settings().language_model
            _print_welcome(model)
            while True:
                try:
                    _console.print(Rule(style=_ACCENT))
                    query = _console.input(f"[bold {_ACCENT}]> [/bold {_ACCENT}]").strip()
                    if not query:
                        continue
                    if query.lower() in ("exit", "quit", "q"):
                        break
                    if cli.send_query(query):
                        cli.wait_for_completion()
                        _console.print()
                except KeyboardInterrupt:
                    break
    finally:
        cli.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
