#!/usr/bin/env bash
# Quick setup script for Robot Agent inside Docker container
# Run this inside your Robot SITL Docker container

set -euo pipefail

# Color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Logging functions with colors
log_info()    { echo -e "${GREEN}[INFO]    $*${NC}"; }
log_warning() { echo -e "${YELLOW}[WARNING] $*${NC}"; }
log_error()   { echo -e "${RED}[ERROR]   $*${NC}"; }

echo "=================================="
log_info "Robot Agent Docker Installation"
echo "=================================="
echo ""

# Get the directory where this script is located
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if we're in the correct directory
if [ ! -f "$WORK_DIR/pyproject.toml" ]; then
    log_error "pyproject.toml not found!"
    log_error "Please run this script from the robot-agent directory"
    echo ""
    log_info "Expected location: /root/robot-agent or wherever the package is located"
    exit 1
fi

# Install uv if not present
log_info "[1/5] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    log_info "uv installed successfully"
else
    log_info "uv already installed"
fi

# Create virtual environment
echo ""
log_info "[2/5] Creating Python virtual environment..."
cd "$WORK_DIR" || exit 1

if [ ! -f ".venv/bin/activate" ]; then
    uv venv
fi

log_info "Sourcing virtual environment..."
# shellcheck source=/dev/null
source ".venv/bin/activate"

# Install dependencies
echo ""
log_info "[3/5] Installing Python dependencies..."
uv sync

# Install package
echo ""
log_info "[4/5] Installing robot-agent package..."
uv pip install -e .

# Source ROS2 if available
echo ""
log_info "[5/5] Checking ROS2 environment..."
if [ -z "${ROS_DISTRO:-}" ]; then
    log_warning "ROS_DISTRO environment variable not set. Skipping ROS2 sourcing."
elif [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    # Temporarily disable -u for ROS2 setup
    set +u
    # shellcheck source=/dev/null
    source "/opt/ros/$ROS_DISTRO/setup.bash"
    set -u
    log_info "ROS2 $ROS_DISTRO sourced successfully"
else
    log_warning "ROS2 not found at /opt/ros/$ROS_DISTRO"
fi

echo ""
echo "=================================="
log_info "Installation Complete!"
echo "=================================="
echo ""
log_info "Script finished. To activate the virtual environment in your shell, run:"
echo "  source \"$WORK_DIR/.venv/bin/activate\""
echo ""
log_info "To launch the agent nodes (vision + agent):"
echo "  ./launch_nodes.sh"
echo ""
log_info "To interact with the agent (interactive mode):"
echo "  ./launch_cli.sh"
echo ""
log_info "To send a single query:"
echo '  ./launch_cli.sh "What is the current altitude?"'
echo ""
log_info "For manual keyboard control:"
echo "  ./launch_teleop.sh"
echo ""
