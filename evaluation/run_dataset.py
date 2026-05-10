"""Reference example: drive a batch of pre-defined missions through PEACE.

This script is **not** the canonical way to use the agent — that is the
``peace-cli`` REPL (or ``peace-cli "<one-shot query>"``). It is a worked
example of how to publish prompts onto ``/agent/query`` programmatically and
wait for each mission to terminate, so the same agent can be exercised across
a dataset for evaluation or benchmarking.

Dataset
-------
The default dataset is the public HuggingFace repo
``erdemuysalx/uav-mission-prompts``. It is loaded with the ``datasets``
library when available; otherwise the script falls back to fetching the raw
``prompts.json`` over HTTPS. A local file may be supplied with
``--prompts /path/to/prompts.json``.

Each row is expected to expose ``id``, ``category``, and ``prompt`` keys.

Adapting to your own dataset
----------------------------
Replace ``_load_prompts`` with whatever loader your dataset needs and adjust
``_metadata_for`` to choose which row fields you want recorded on the CSV
``metadata_json`` column. Nothing else in the agent needs to change — the
mission logger writes whatever metadata dict you hand it, opaquely.

Usage examples
--------------
::

    # run every prompt from the default HF dataset
    python -m evaluation.run_dataset

    # only takeoff/landing, 10 s gap between missions
    python -m evaluation.run_dataset --category takeoff_and_landing --delay 10

    # use a local copy
    python -m evaluation.run_dataset --prompts ./prompts.json --limit 5

    # use a different HF dataset
    python -m evaluation.run_dataset --prompts other-user/other-dataset
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rich.console import Console

from peace.nodes.cli import AgentCLI

_DEFAULT_DATASET = "erdemuysalx/uav-mission-prompts"
_HF_RAW_URL = "https://huggingface.co/datasets/{repo}/resolve/main/prompts.json"
_console = Console()


def _load_prompts(source: str) -> list[dict]:
    """Load prompts from a HuggingFace dataset id, a URL, or a local JSON path.

    HF id detection is heuristic: anything containing ``/`` and not pointing
    at an existing path is treated as a HF repo. The script prefers the
    ``datasets`` library and falls back to a plain HTTPS fetch of
    ``prompts.json`` so installs without ``datasets`` still work.
    """
    local = Path(source)
    if local.exists():
        data = json.loads(local.read_text(encoding="utf-8"))
        prompts = data.get("prompts", data) if isinstance(data, dict) else data
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"No prompts found in {local}")
        return prompts

    # Treat anything else as a HF dataset id.
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset(source, split="train")
        return [dict(row) for row in ds]
    except ImportError:
        pass

    import urllib.request

    url = _HF_RAW_URL.format(repo=source)
    _console.print(f"[dim]datasets not installed; fetching {url}[/dim]")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prompts = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found at {url}")
    return prompts


def _filter_prompts(
    prompts: list[dict],
    category: Optional[str],
    start_id: Optional[str],
    limit: Optional[int],
) -> list[dict]:
    """Apply category / start-id / limit filters in dataset order."""
    selected = list(prompts)
    if category:
        selected = [p for p in selected if p.get("category") == category]
    if start_id:
        for i, p in enumerate(selected):
            if p.get("id") == start_id:
                selected = selected[i:]
                break
        else:
            raise ValueError(f"start-id '{start_id}' not found in dataset (after filters)")
    if limit is not None:
        selected = selected[:limit]
    return selected


def _metadata_for(prompt: dict) -> dict[str, str]:
    """Project a dataset row into the metadata dict that rides with each mission.

    These keys end up serialised into the ``metadata_json`` CSV column. The
    agent does not interpret them; downstream scoring scripts join on them.
    """
    return {
        "prompt_id": str(prompt.get("id", "")),
        "task_category": str(prompt.get("category", "")),
    }


def _rtl_and_wait(cli: AgentCLI, rtl_timeout: float = 90.0) -> bool:
    """Engage AUTO.RTL and wait for PX4 to land + auto-disarm at home."""
    set_mode = cli.create_client(SetMode, "/mavros/set_mode")
    if not set_mode.wait_for_service(timeout_sec=5.0):
        _console.print("[yellow]  /mavros/set_mode unavailable — skipping RTL[/yellow]")
        return False
    req = SetMode.Request()
    req.custom_mode = "AUTO.RTL"
    future = set_mode.call_async(req)
    deadline = time.time() + 5.0
    while not future.done() and time.time() < deadline:
        rclpy.spin_once(cli, timeout_sec=0.1)
    if not future.done() or not (future.result() and future.result().mode_sent):
        _console.print("[red]  Failed to engage AUTO.RTL[/red]")
        return False

    armed = {"value": True}

    def _state_cb(msg: State) -> None:
        armed["value"] = msg.armed

    sub = cli.create_subscription(State, "/mavros/state", _state_cb, 10)
    deadline = time.time() + rtl_timeout
    try:
        while time.time() < deadline:
            rclpy.spin_once(cli, timeout_sec=0.2)
            if not armed["value"]:
                return True
    finally:
        cli.destroy_subscription(sub)
    _console.print(f"[yellow]  RTL did not complete within {rtl_timeout:.0f}s[/yellow]")
    return False


def _force_disarm(cli: AgentCLI, timeout: float = 5.0) -> bool:
    """Call /mavros/cmd/arming to force-disarm the drone before a teleport."""
    client = cli.create_client(CommandBool, "/mavros/cmd/arming")
    if not client.wait_for_service(timeout_sec=timeout):
        _console.print("[yellow]  /mavros/cmd/arming unavailable — skipping disarm[/yellow]")
        return False
    req = CommandBool.Request()
    req.value = False
    future = client.call_async(req)
    deadline = time.time() + timeout
    while not future.done() and time.time() < deadline:
        rclpy.spin_once(cli, timeout_sec=0.1)
    if not future.done():
        return False
    return bool(future.result() and future.result().success)


def _teleport_to_origin(model: str, world: str, spawn_z: float) -> bool:
    """Teleport the drone model back to (0, 0, spawn_z) via Gazebo Harmonic's set_pose service."""
    req = (
        f'name: "{model}", position: {{x: 0.0, y: 0.0, z: {spawn_z}}}, '
        'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}'
    )
    cmd = [
        "gz", "service", "-s", f"/world/{world}/set_pose",
        "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000", "--req", req,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _console.print(f"[red]  gz set_pose failed: {e}[/red]")
        return False
    ok = result.returncode == 0 and "true" in result.stdout.lower()
    if not ok:
        _console.print(f"[red]  gz set_pose rejected: {result.stdout or result.stderr}[/red]")
    return ok


def _reset_drone(
    cli: AgentCLI,
    mode: str,
    model: str,
    world: str,
    spawn_z: float,
    settle: float,
) -> None:
    """Reset the drone using the chosen strategy, then wait for the stack to settle."""
    _console.print(f"[dim]  Resetting drone via {mode}...[/dim]")
    if mode == "rtl":
        _rtl_and_wait(cli)
    elif mode == "teleport":
        _force_disarm(cli)
        time.sleep(0.5)
        _teleport_to_origin(model, world, spawn_z)
    else:
        _console.print(f"[red]  Unknown reset mode '{mode}' — skipping[/red]")
        return
    time.sleep(settle)


def _run(
    prompts: list[dict],
    delay: float,
    timeout: float,
    reset_between: bool,
    reset_mode: str,
    model: str,
    world: str,
    spawn_z: float,
    settle: float,
) -> dict[str, int]:
    """Send each prompt to the agent and wait for completion."""
    rclpy.init()
    cli = AgentCLI()
    counts = {"completed": 0, "failed": 0, "aborted": 0, "no_status": 0}
    try:
        for i, prompt in enumerate(prompts, start=1):
            pid = prompt.get("id", "")
            cat = prompt.get("category", "")
            text = prompt.get("prompt", "")
            _console.rule(f"[bold]{i}/{len(prompts)}  {pid}  ({cat})[/bold]")
            _console.print(f"[dim]{text}[/dim]")

            if reset_between:
                _reset_drone(cli, reset_mode, model, world, spawn_z, settle)
            if not cli.send_query(text, metadata=_metadata_for(prompt)):
                counts["failed"] += 1
                continue
            cli.wait_for_completion(timeout=timeout)
            terminal = cli._terminal_state or "no_status"
            counts[terminal] = counts.get(terminal, 0) + 1
            if i < len(prompts) and delay > 0:
                time.sleep(delay)
    finally:
        cli.destroy_node()
        rclpy.shutdown()
    return counts


def main() -> None:
    """Entry point — parse arguments, load dataset, run, print summary."""
    parser = argparse.ArgumentParser(description="Run evaluation prompts through PEACE")
    parser.add_argument(
        "--prompts", type=str, default=_DEFAULT_DATASET,
        help=(
            "HuggingFace dataset id (default: "
            f"{_DEFAULT_DATASET}) or path to a local prompts.json"
        ),
    )
    parser.add_argument("--category", type=str, default=None, help="Only run this task category")
    parser.add_argument("--start-id", type=str, default=None, help="Resume from this prompt id")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N missions")
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Seconds to wait between missions (default: 5)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-mission timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--reset-between", action="store_true",
        help="Reset the drone before each prompt (see --reset-mode)",
    )
    parser.add_argument(
        "--reset-mode", type=str, choices=("rtl", "teleport"), default="rtl",
        help="rtl: AUTO.RTL — drone flies home, lands, auto-disarms (default); "
             "teleport: gz set_pose to (0,0,spawn-z) after force-disarm",
    )
    parser.add_argument(
        "--model", type=str, default="x500_depth",
        help="Gazebo model name to teleport (default: x500_depth)",
    )
    parser.add_argument(
        "--world", type=str, default="default",
        help="Gazebo world name (default: default)",
    )
    parser.add_argument(
        "--spawn-z", type=float, default=0.2,
        help="Z coordinate for teleport target — keep slightly above 0 to avoid ground clip (default: 0.2)",
    )
    parser.add_argument(
        "--settle", type=float, default=10.0,
        help="Seconds to wait after teleport for EKF to converge (default: 10)",
    )
    args = parser.parse_args()

    prompts = _filter_prompts(
        _load_prompts(args.prompts), args.category, args.start_id, args.limit
    )
    if not prompts:
        _console.print("[red]No prompts selected after filters[/red]")
        sys.exit(2)

    _console.print(f"[bold]Running {len(prompts)} prompt(s) from {args.prompts}[/bold]")
    counts = _run(
        prompts, delay=args.delay, timeout=args.timeout,
        reset_between=args.reset_between, reset_mode=args.reset_mode,
        model=args.model, world=args.world, spawn_z=args.spawn_z, settle=args.settle,
    )

    _console.rule("[bold]Summary[/bold]")
    for state, n in counts.items():
        _console.print(f"  {state}: {n}")


if __name__ == "__main__":
    main()
