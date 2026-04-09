#!/usr/bin/env bash
# ./scripts/install_externals.sh
# Installs external dependencies in editable mode using uv or pip.

set -euo pipefail

# Get the project root (directory where this script is located's parent)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Installing external packages from $PROJECT_ROOT/external..."

if [ -n "${PYTHON_CMD:-}" ]; then
    :
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
elif [ -x "/isaac-sim/kit/python/bin/python3" ]; then
    PYTHON_CMD="/isaac-sim/kit/python/bin/python3"
else
    echo "Error: python3 not found."
    exit 1
fi
echo "Using Python: $PYTHON_CMD"

# Locate uv: prefer PATH (venv/bin/uv), then python -m uv
if command -v uv >/dev/null 2>&1; then
    UV_CMD="uv"
elif "$PYTHON_CMD" -m uv --version >/dev/null 2>&1; then
    UV_CMD="$PYTHON_CMD -m uv"
else
    UV_CMD=""
fi

if [ -n "$UV_CMD" ]; then
    echo "Using uv: $UV_CMD"
else
    echo "uv not found; will fall back to pip (may be slower or require extra deps)."
fi

# Determine whether the target python is a virtualenv
# (sys.prefix != getattr(sys, 'base_prefix', sys.prefix))
IS_VENV=0
if "$PYTHON_CMD" -c 'import sys; print(1 if getattr(sys, "base_prefix", sys.prefix) != sys.prefix else 0)'; then
    IS_VENV=$("$PYTHON_CMD" -c 'import sys; print(1 if getattr(sys, "base_prefix", sys.prefix) != sys.prefix else 0)')
fi

# Externals to install
EXTERNALS=("dino" "vjepa2")

for pkg in "${EXTERNALS[@]}"; do
    pkg_dir="$PROJECT_ROOT/external/$pkg"
    if [ ! -d "$pkg_dir" ]; then
        echo "Warning: External directory not found: $pkg_dir"
        continue
    fi

    # Skip directories that are not installable Python projects
    if [ ! -f "$pkg_dir/pyproject.toml" ] && [ ! -f "$pkg_dir/setup.py" ] && [ ! -f "$pkg_dir/setup.cfg" ]; then
        echo "Skipping $pkg — no pyproject.toml/setup.py/setup.cfg found (not an installable Python package)"
        continue
    fi

    echo "Installing $pkg in editable mode..."
    cd "$pkg_dir" || exit 1

    # Create COLCON_IGNORE to avoid ROS2 colcon scanning if applicable
    touch COLCON_IGNORE

    INSTALL_OK=0

    if [ -n "$UV_CMD" ]; then
        if [ "$IS_VENV" -eq 1 ]; then
            echo "Target python is a venv — using uv to install into that venv"
            $UV_CMD pip install --python "$PYTHON_CMD" -e . --no-build-isolation && INSTALL_OK=1 || true
            if [ "$INSTALL_OK" -eq 0 ]; then
                echo "Warning: uv venv install failed for $pkg, retrying without --no-build-isolation..."
                $UV_CMD pip install --python "$PYTHON_CMD" -e . && INSTALL_OK=1 || true
            fi
        else
            echo "Target python is NOT a venv — using uv --system to install into ambient Python"
            $UV_CMD pip install --system --python "$PYTHON_CMD" -e . --no-build-isolation && INSTALL_OK=1 || true
            if [ "$INSTALL_OK" -eq 0 ]; then
                echo "Warning: uv system install failed for $pkg, retrying without --no-build-isolation..."
                $UV_CMD pip install --system --python "$PYTHON_CMD" -e . && INSTALL_OK=1 || true
            fi
        fi
    fi

    if [ "$INSTALL_OK" -eq 0 ]; then
        echo "Falling back to pip for $pkg..."
        "$PYTHON_CMD" -m pip install -e . --no-build-isolation || \
            "$PYTHON_CMD" -m pip install -e .
    fi
done

cd "$PROJECT_ROOT" || exit 1
echo "Finished installing external packages."
