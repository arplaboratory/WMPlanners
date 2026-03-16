#!/bin/bash
# ./scripts/install_externals.sh
# Installs external dependencies in editable mode using uv.

# Get the project root (directory where this script is located's parent)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================="
echo "Installing external packages from $PROJECT_ROOT/external..."
echo "========================================="

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed. Please install it first: https://github.com/astral-sh/uv"
    exit 1
fi

# List of externals to install
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

    # Install using uv.
    # --no-build-isolation is often needed for research repos with complex dependencies
    uv pip install -e . --no-build-isolation || {
        echo "Warning: Standard install failed for $pkg, trying without --no-build-isolation..."
        uv pip install -e .
    }
done

cd "$PROJECT_ROOT" || exit 1
echo "Finished installing external packages."
