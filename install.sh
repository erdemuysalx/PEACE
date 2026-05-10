#!/usr/bin/env bash
# Quick setup script for PEACE environment setup

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
log_info "PEACE Environment Setup"
echo "=================================="
echo ""

# Get the directory where this script is located
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check the current working directory for pyproject.toml
if [ ! -f "$WORK_DIR/pyproject.toml" ]; then
    log_error "pyproject.toml not found!"
    log_error "Please run this script from the PEACE directory"
    echo ""
    log_info "Expected location: /root/peace or wherever the package is located"
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
log_info "[4/5] Installing peace package..."
uv pip install -e .

# Source ROS if available
echo ""
log_info "[5/5] Checking ROS environment..."
if [ -z "${ROS_DISTRO:-}" ]; then
    log_warning "ROS_DISTRO environment variable not set. Skipping ROS sourcing."
elif [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    # Temporarily disable -u for ROS setup
    set +u
    # shellcheck source=/dev/null
    source "/opt/ros/$ROS_DISTRO/setup.bash"
    set -u
    log_info "ROS $ROS_DISTRO sourced successfully"
else
    log_warning "ROS not found at /opt/ros/$ROS_DISTRO"
fi

echo ""
echo "=================================="
log_info "Environment Setup Complete!"
echo "=================================="
echo ""
log_info "Script finished. To activate the virtual environment in your shell, run:"
echo "  source \"$WORK_DIR/.venv/bin/activate\""
echo ""
log_info "To launch the vision node:"
echo "  peace-vision"
echo ""
log_info "To launch the planner-executor node:"
echo "  peace-planner-executor"
echo ""
log_info "To interact with the agent"
echo '  peace-cli'
echo ""
