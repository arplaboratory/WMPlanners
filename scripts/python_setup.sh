#!/bin/bash
# ./scripts/python_setup.sh [--no-externals]
# Sets up the Python environment for the World Model project using uv.

# Get the project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Defaults
VENV_PATH="${WM_VENV:-$PROJECT_ROOT/.venv}"
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
            echo "  WM_VENV: Path to the virtual environment (default: $PROJECT_ROOT/.venv)"
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
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_PATH" --python 3.11 || exit 1
fi

# Activate environment
. "$VENV_PATH/bin/activate"

# 2. Install main project and dependencies
echo "Installing project dependencies (including dev)..."
cd "$PROJECT_ROOT" || exit 1
uv pip install -e ".[dev]" || exit 1
uv pip install --upgrade pyarrow datasets

# 3. Install external packages (DINOv2, V-JEPA2)
if [ "$INSTALL_EXTERNALS" = true ]; then
    # Patch for Python 3.11 and dependency compatibility:
    # - xformers==0.0.18 is not supported on 3.11.
    # - torch/torchvision pins conflict with uv resolution and platform availability.
    # - Platform-specific index URLs can cause resolution failures.
    if [ -f "$PROJECT_ROOT/external/dino/requirements.txt" ]; then
        echo "Patching external/dino/requirements.txt for Python 3.11 and dependency compatibility..."
        sed -i '/xformers==0.0.18/d' "$PROJECT_ROOT/external/dino/requirements.txt"
        sed -i 's/torch==2.0.0/torch>=2.0/g' "$PROJECT_ROOT/external/dino/requirements.txt"
        sed -i 's/torchvision==0.15.0/torchvision>=0.15/g' "$PROJECT_ROOT/external/dino/requirements.txt"
        sed -i '/--extra-index-url https:\/\/download.pytorch.org\/whl\/cu117/d' "$PROJECT_ROOT/external/dino/requirements.txt"
        sed -i '/--extra-index-url https:\/\/pypi.nvidia.com/d' "$PROJECT_ROOT/external/dino/requirements.txt"
        sed -i '/cuml-cu11/d' "$PROJECT_ROOT/external/dino/requirements.txt"
    fi

    bash "$PROJECT_ROOT/scripts/install_externals.sh" || exit 1
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
