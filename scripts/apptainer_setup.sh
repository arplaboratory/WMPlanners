#!/usr/bin/env bash
# scripts/apptainer_setup.sh
# Build a WM-capable SIF on top of a base SIF.
#
# The base SIF can be:
#   cuda    - any CUDA-enabled SIF with system Python (e.g. nvcr.io/nvidia/cuda:12.8-devel)
#
# Key environment variables:
#   BASE_SIF_IMAGE   Path to the base .sif file          [required]
#   SIF_IMAGE        Output .sif path                    [default: <BASE_SIF_IMAGE dir>/wm.sif]
#   SANDBOX_DIR      Writable sandbox directory          [default: <BASE_SIF_IMAGE dir>/wm_container]
#   PROJECT_ROOT     Path to this repo on the host       [default: directory two levels up from this script]
#   WM_VENV_NAME     Name for the wm venv inside /opt/venvs [default: wm]
#   SCRATCH_TMP      Temp dir to bind-mount as /tmp      [default: <BASE_SIF_IMAGE dir>/.wm_scratch]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WM_WS="${WM_WS:-$(cd "$PROJECT_ROOT/../.." && pwd)}"

: "${BASE_SIF_IMAGE:?'BASE_SIF_IMAGE must be set to the base .sif file path'}"
BASE_SIF_IMAGE="$(cd "$(dirname "$BASE_SIF_IMAGE")" && pwd)/$(basename "$BASE_SIF_IMAGE")"

SIF_DIR="$(dirname "$BASE_SIF_IMAGE")"
: "${SIF_IMAGE:=$SIF_DIR/wm.sif}"
: "${SANDBOX_DIR:=$SIF_DIR/wm_container}"
: "${WM_VENV_NAME:=wm}"
: "${SCRATCH_TMP:=$SIF_DIR/.wm_scratch}"
: "${APPTAINER_TMPDIR:=$SIF_DIR/.apptainer_tmp}"
: "${APPTAINER_CACHEDIR:=$SIF_DIR/.apptainer_cache}"

export APPTAINER_TMPDIR APPTAINER_CACHEDIR

if [ ! -f "$BASE_SIF_IMAGE" ]; then
    echo "ERROR: BASE_SIF_IMAGE not found: $BASE_SIF_IMAGE" >&2
    exit 1
fi

echo "Creating host directories..."
mkdir -p "$SCRATCH_TMP" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Building sandbox from $BASE_SIF_IMAGE..."
    apptainer build --fakeroot --sandbox "$SANDBOX_DIR" "$BASE_SIF_IMAGE"
else
    echo "Sandbox $SANDBOX_DIR already exists, skipping build."
fi

mkdir -p "$SANDBOX_DIR/opt/venvs"
mkdir -p "$SANDBOX_DIR/workspace/run"
chmod 1777 "$SCRATCH_TMP"

echo "Installing wm_planner dependencies..."

EXEC_FLAGS=(
    --fakeroot --writable
    -B "$SCRATCH_TMP":/tmp:rw
    -B "$WM_WS":/workspace/run:rw
)

apptainer exec "${EXEC_FLAGS[@]}" "$SANDBOX_DIR" bash -c "
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export TMPDIR=/tmp
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
WM_REPO_NAME='$(basename "$PROJECT_ROOT")'
VENV=\"/opt/venvs/${WM_VENV_NAME}\"

echo \"Checking required system dependencies...\"
MISSING_PKGS=\"\"
for pkg in python3 python3-pip python3-venv python3-dev curl; do
    if ! dpkg -s \"\$pkg\" >/dev/null 2>&1; then
        MISSING_PKGS=\"\$MISSING_PKGS \$pkg\"
    fi
done
if [ -n \"\$MISSING_PKGS\" ]; then
    echo \"Installing missing system packages:\$MISSING_PKGS\"
    apt-get -o APT::Sandbox::User=root update -qq
    apt-get -o APT::Sandbox::User=root install -y -qq \$MISSING_PKGS
else
    echo \"All required system packages already present, skipping apt-get.\"
fi

echo \"Checking for uv...\"
export HOME=/root
export PATH=\"/root/.local/bin:\$PATH\"
if ! command -v uv >/dev/null 2>&1; then
    echo \"uv not found, installing via official installer...\"
    curl -sSf https://astral.sh/uv/install.sh | sh
else
    echo \"uv already installed: \$(uv --version)\"
fi

# Create venv with Python 3.11 if possible, else fall back to system default
# --allow-existing skips recreation if the venv already exists
if [ -d \"\$VENV\" ]; then
    echo \"Venv already exists at \$VENV, skipping creation.\"
elif uv venv --seed --python 3.11 \"\$VENV\" 2>/dev/null; then
    echo \"Created venv with Python 3.11\"
else
    echo \"Python 3.11 not available; falling back to system Python\"
    uv venv --seed \"\$VENV\"
fi

VENV_PY=\"\$VENV/bin/python\"
echo \"Venv Python: \$(\$VENV_PY --version)\"

echo \"Installing wm_planner into \$VENV...\"
uv pip install --python \"\$VENV_PY\" --no-cache-dir setuptools wheel hatchling editables
# On Jetson (aarch64) pre-install PyTorch from NVIDIA JetPack index before
# resolving the rest of the deps (PyPI torch won't work on ARM)
if [ "\$(uname -m)" = "aarch64" ]; then
    JP_TORCH_INDEX="\${JP_TORCH_INDEX:-https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/}"
    echo "[Jetson] Pre-installing PyTorch from \$JP_TORCH_INDEX"
    uv pip install --python "\$VENV_PY" \
        --index-url "\$JP_TORCH_INDEX" \
        "torch>=2.3" "torchvision" --no-deps || \
        echo "[WARN] Jetson torch install failed — falling back to uv"
fi
uv pip install --python \"\$VENV_PY\" \
    --no-cache-dir \
    --no-build-isolation \
    -e "/workspace/run/src/\$WM_REPO_NAME[dev]"

# Install external packages if present
for ext in dino vjepa2; do
    ext_dir="/workspace/run/src/\$WM_REPO_NAME/external/\$ext"
    if [ -d \"\$ext_dir\" ]; then
        echo \"Installing external: \$ext\"
        if [ -f \"\$ext_dir/requirements.txt\" ]; then
            sed -i '/xformers==0.0.18/d' \"\$ext_dir/requirements.txt\"
            sed -i 's/torch==2.0.0/torch>=2.0/g' \"\$ext_dir/requirements.txt\"
            sed -i 's/torchvision==0.15.0/torchvision>=0.15/g' \"\$ext_dir/requirements.txt\"
            sed -i '/--extra-index-url.*pytorch.org/d' \"\$ext_dir/requirements.txt\"
            sed -i '/--extra-index-url.*nvidia.com/d' \"\$ext_dir/requirements.txt\"
            sed -i '/cuml-cu11/d' \"\$ext_dir/requirements.txt\"
        fi
        uv pip install --python \"\$VENV_PY\" \
            --no-cache-dir --no-build-isolation -e \"\$ext_dir\" || echo \"Warning: external \$ext install failed (non-fatal)\"
    fi
done

echo \"Verifying installation...\"
\"\$VENV_PY\" -c 'import torch; print(\"torch:\", torch.__version__)'
\"\$VENV_PY\" -m pip check || echo \"Warning: pip check reported conflicts (may be non-fatal)\"
echo \"wm venv ready at \$VENV\"
"

echo "Converting sandbox to SIF: $SIF_IMAGE"
[ -f "$SIF_IMAGE" ] && rm -f "$SIF_IMAGE"
apptainer build --fakeroot "$SIF_IMAGE" "$SANDBOX_DIR"

echo ""
echo "Build complete!"

exit 0
