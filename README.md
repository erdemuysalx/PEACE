# PEACE: Planner-Executor Agent with Constraint Enforcement

A multi-modal AI agent that integrates robotics platforms with LLM-based mission planning and visual perception for natural-language-driven robot control. It uses ROS 2 as the communication gateway between the AI layer and firmware.

## Features

- Natural language mission planning via Ollama (local or cloud)
- Deterministic planner-executor pattern: LLM plans once, `UAVTools` executes without re-inference
- Pluggable object detection: YOLO (ultralytics) or VLM (Ollama vision model)
- 3D depth projection of detected objects from 2D bounding boxes
- Automatic replanning on step failure (configurable max depth)
- Thread-safe world model aggregation across all telemetry sources
- Safety validation: altitude limits, horizontal geofence, per-command checks
- CSV mission logging for post-flight analysis
- Docker-based SITL environment: ROS 2 Jazzy, Gazebo Harmonic, MAVROS, NoVNC browser UI
- Keyboard and joystick teleoperation

## Prerequisites

1. **Docker** and **Docker Compose**
2. **Ollama** — accessible from inside the container (local or cloud endpoint)
3. **YOLO model** — `yolov8n.pt` (auto-downloaded by ultralytics) or a custom `.pt` file placed in `models/`

For native (non-Docker) usage:
- Python 3.10–3.12
- ROS 2 Jazzy
- MAVROS (`ros-jazzy-mavros`)
- Gazebo Harmonic
- PX4 Autopilot (SITL)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd peace
```

### 2. Start the SITL container

```bash
docker compose up -d
```

This builds the image (ROS 2 Jazzy + Gazebo Harmonic + MAVROS + NoVNC) and mounts the project into `/root/peace` inside the container. The NoVNC interface is available at `http://localhost:6080`.

### 3. Install inside the container

```bash
docker exec -it px4_sitl bash
cd /root/peace
./install.sh
source .venv/bin/activate
```

### 4. Configure

Edit `peace/.env`:

```bash
# Ollama
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_API_KEY=                          # leave empty for local Ollama

# Models
LANGUAGE_MODEL=llama3.2
VISION_MODEL=llava:13b

# Object detection backend
USE_YOLO=true
YOLO_MODEL=yolov8n.pt                    # name (auto-download) or path to .pt file
YOLO_DEVICE=cpu                          # cpu | cuda | mps

# Flight control
ACTION_ENABLE_CONTROL=false              # set to true to allow MAVROS commands

# Safety limits
AGENT_MIN_ALTITUDE=1.0
AGENT_MAX_ALTITUDE=50.0
AGENT_HORIZONTAL_LIMIT=500.0

# Replanning
AGENT_MAX_REPLAN_DEPTH=2
AGENT_ENABLE_MEMORY=false

# Logging
LOG_ENABLE=true
LOG_OUTPUT_DIR=./logs

# Debug
DEBUG_SAVE_IMAGES=false
DEBUG_OUTPUT_DIR=./debug_images
```

## Usage

### Start PX4 SITL and MAVROS

Inside the container:

```bash
# Terminal 1 — PX4 SITL
cd /root/PX4-Autopilot && make px4_sitl gz_x500

# Terminal 2 — MAVROS
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14540@14557
```

### Launch agent nodes

```bash
peace-vision
peace-planner-executor
```

Starts the **Vision Node** and **Planner-Executor Node** in the background. Ctrl+C stops both cleanly.

### Send queries (interactive)

```bash
peace-cli
```

Opens an interactive prompt. Type natural language mission queries:

```
> takeoff to 5 meters and fly north 20 meters
> scan for humans at the current position
> approach the nearest vehicle and stop 3 meters away
> land
```

### Single query

```bash
peace-cli "takeoff to 10 meters and hold position"
```

Add `--monitor` to keep the terminal open and watch execution status after the query is sent.

## Evaluation

A reference batch runner lives at [`evaluation/run_dataset.py`](evaluation/run_dataset.py). It drives PEACE through a dataset of pre-defined prompts and writes one CSV row per mission via the agent's `MissionLoggerService`. The default dataset is the public HuggingFace repo [erdemuysalx/uav-mission-prompts](https://huggingface.co/datasets/erdemuysalx/uav-mission-prompts).

```bash
uv pip install -e '.[eval]'   # adds the optional `datasets` dependency
peace-vision
peace-planner-executor
python -m evaluation.run_dataset --limit 5
```

This script is not the canonical way to use PEACE — it is a worked example of how to publish queries onto `/agent/query` programmatically. Each row's `id` and `category` ride along inside the agent's opaque `metadata` field, end up serialised in the CSV's `metadata_json` column, and can be joined back to the dataset for scoring.

## Citation

If you use PEACE in your research, please cite our paper ([arXiv:2606.00104](https://arxiv.org/abs/2606.00104)):

```bibtex
@misc{uysal2026peaceplannerexecutoragentconstraint,
      title={PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs},
      author={Erdem Uysal and Timo Kehrer and Sebastiano Panichella},
      year={2026},
      eprint={2606.00104},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.00104},
}
```
