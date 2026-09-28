#!/bin/bash
# ./scripts/python_setup.sh [--no-externals]
# Sets up the Python environment for the World Model project using uv.

# Get the project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Defaults
VENV_PATH="${WM_VENV:-$PROJECT_ROOT/.venvs/wm}"
PYTHON_BIN="${WM_PYTHON:-/usr/bin/python3.12}"
INSTALL_EXTERNALS=true

while :; do
    case "$1" in
        --no-externals)
            INSTALL_EXTERNALS=false
            shift
            ;;
        -h|-?|--help)
            echo "Usage: ./python_setup.sh [--no-externals]"
            echo "Environment variables:"
            echo "  WM_VENV: Path to the virtual environment (default: $PROJECT_ROOT/.venvs/wm)"
            echo "  WM_PYTHON: Python 3.12 interpreter (default: /usr/bin/python3.12)"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

echo "========================================="
echo "Setting up environment in $VENV_PATH..."
echo "========================================="

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed. Please install it first: https://github.com/astral-sh/uv"
    exit 1
fi

# 1. Create virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment with Python 3.12 and system site packages..."
    uv venv "$VENV_PATH" --python "$PYTHON_BIN" --system-site-packages || exit 1
fi

# Activate environment
. "$VENV_PATH/bin/activate"
if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    echo "Error: $VENV_PATH is not a Python 3.12 environment. Set WM_VENV to a new path."
    exit 1
fi
if [ "${ROS_DISTRO:-}" = "jazzy" ] && ! python -c 'from rclpy.node import Node' 2>/dev/null; then
    echo "Error: $VENV_PATH cannot load Jazzy rclpy. Source Jazzy and use a fresh system-site-packages Python 3.12 environment."
    exit 1
fi

# 2. Install main project and dependencies
echo "Installing project dependencies (including dev)..."
cd "$PROJECT_ROOT" || exit 1
uv pip install -e ".[dev]" || exit 1
uv pip install --upgrade pyarrow datasets

# 3. Install external packages (DINOv2, V-JEPA2)
if [ "$INSTALL_EXTERNALS" = true ]; then
    # Upstream pins target older PyTorch/Python releases. Install their sources
    # and the ROS bridge runtime dependencies separately.
    uv pip install --no-deps -e "$PROJECT_ROOT/external/dino" -e "$PROJECT_ROOT/external/vjepa2" || exit 1
    uv pip install fvcore iopath torchmetrics peft fire python-box || exit 1
fi

# 4. Initialize .env file
echo "Initializing .env file..."
DOTENV="$PROJECT_ROOT/.env"

if [ ! -f "$DOTENV" ]; then
    echo "PROJECT_ROOT=\"$PROJECT_ROOT\"" > "$DOTENV"
    echo "PYTHONPATH=\"\$PROJECT_ROOT/src\"" >> "$DOTENV"
    echo "Created .env file"
else
    echo ".env file already exists"
fi

echo "========================================="
echo "Setup complete!"
echo "To activate the environment: source $VENV_PATH/bin/activate"
echo "To source environment variables: set -a; source .env; set +a"
echo "========================================="
