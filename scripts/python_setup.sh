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

echo "Setting up environment in $VENV_PATH..."

if command -v uv >/dev/null 2>&1; then
    CREATE_VENV_WITH_UV=true
else
    CREATE_VENV_WITH_UV=false
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment"
    PYTHON_EXEC="$(command -v python3.11 || command -v python3)"
    if [ -z "$PYTHON_EXEC" ]; then
        echo "Error: no suitable python binary found (python3.11 or python3). Install one or install 'uv' to control the venv creation."
        exit 1
    fi
    if [ "$CREATE_VENV_WITH_UV" = true ]; then
        uv venv "$VENV_PATH" --python "$PYTHON_EXEC" || exit 1
    else
        "$PYTHON_EXEC" -m venv "$VENV_PATH" || exit 1
    fi
fi

# Activate environment
. "$VENV_PATH/bin/activate"

echo "Upgrading pip/setuptools/wheel and installing uv into the venv"
python -m pip install --upgrade pip setuptools wheel || exit 1
python -m pip install --upgrade uv || exit 1

echo "Installing project dependencies"
cd "$PROJECT_ROOT" || exit 1
uv pip install -e ".[dev]" || exit 1

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

    export PYTHON_CMD="$VENV_PATH/bin/python3"
    export PATH="$VENV_PATH/bin:$PATH"
    PYTHON_CMD="$PYTHON_CMD" PATH="$PATH" bash "$PROJECT_ROOT/scripts/install_externals.sh" || exit 1
fi

echo "Initializing .env file..."
DOTENV="$PROJECT_ROOT/.env"

if [ ! -f "$DOTENV" ]; then
    echo "PROJECT_ROOT=\"$PROJECT_ROOT\"" > "$DOTENV"
    echo "PYTHONPATH=\"\$PROJECT_ROOT/src\"" >> "$DOTENV"
    echo "Created .env file"
else
    echo ".env file already exists"
fi

echo "Setup complete!"
echo "To activate the environment: source $VENV_PATH/bin/activate"
echo "To source environment variables: set -a; source .env; set +a"
