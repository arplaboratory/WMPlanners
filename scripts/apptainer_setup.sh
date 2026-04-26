#!/usr/bin/env bash
# scripts/apptainer_setup.sh
# Build a WM-capable SIF on top of a base SIF.
#
# The base SIF can be:
#   isaac   - raw isaac-sim_X.Y.Z.sif (no ROS2, but Isaac Sim Python available)
#   cuda    - any CUDA-enabled SIF with system Python (e.g. nvcr.io/nvidia/cuda:12.8-devel)
#
# Key environment variables:
#   SIF_IMAGE        Path to the base .sif file          [required]
#   WM_SIF_IMAGE     Output .sif path                    [default: <SIF_IMAGE dir>/wm.sif]
#   SANDBOX_DIR      Writable sandbox directory          [default: <SIF_IMAGE dir>/wm_container]
#   PROJECT_ROOT     Path to this repo on the host       [default: directory two levels up from this script]
#   WM_VENV_NAME     Name for the wm venv inside /opt/venvs [default: wm]
#   SCRATCH_TMP      Temp dir to bind-mount as /tmp      [default: <SIF_IMAGE dir>/.wm_scratch]

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

: "${SIF_IMAGE:?'SIF_IMAGE must be set to the base .sif file path'}"

SIF_DIR="$(dirname "$SIF_IMAGE")"
: "${WM_SIF_IMAGE:=$SIF_DIR/wm.sif}"
: "${SANDBOX_DIR:=$SIF_DIR/wm_container}"
: "${WM_VENV_NAME:=wm}"
: "${SCRATCH_TMP:=$SIF_DIR/.wm_scratch}"
: "${APPTAINER_TMPDIR:=$SIF_DIR/.apptainer_tmp}"
: "${APPTAINER_CACHEDIR:=$SIF_DIR/.apptainer_cache}"

export APPTAINER_TMPDIR APPTAINER_CACHEDIR

echo "================================================="
echo " wm_planner apptainer_setup.sh"
echo "================================================="
echo "  SIF_IMAGE    : $SIF_IMAGE"
echo "  SANDBOX_DIR  : $SANDBOX_DIR"
echo "  WM_SIF_IMAGE : $WM_SIF_IMAGE"
echo "  PROJECT_ROOT : $PROJECT_ROOT"
echo "  WM_VENV_NAME : $WM_VENV_NAME"
echo "================================================="

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: SIF_IMAGE not found: $SIF_IMAGE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Host directories
# ---------------------------------------------------------------------------
echo "[local] Creating host directories..."
mkdir -p "$SCRATCH_TMP" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---------------------------------------------------------------------------
# Build sandbox from base SIF
# ---------------------------------------------------------------------------
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "[local] Building sandbox from $SIF_IMAGE..."
    apptainer build --fakeroot --sandbox "$SANDBOX_DIR" "$SIF_IMAGE"
else
    echo "[local] Sandbox $SANDBOX_DIR already exists, skipping build."
fi

# Ensure /opt/venvs mount point exists inside sandbox
mkdir -p "$SANDBOX_DIR/opt/venvs"
mkdir -p "$SANDBOX_DIR/workspace/run"
chmod 1777 "$SCRATCH_TMP"

# ---------------------------------------------------------------------------
# Install wm_planner dependencies inside sandbox
# ---------------------------------------------------------------------------
echo "[local] Installing wm_planner dependencies..."

# Common apptainer exec flags
EXEC_FLAGS=(
    --fakeroot --writable
    -B "$SCRATCH_TMP":/tmp:rw
    -B "$PROJECT_ROOT":/workspace/run:rw
)

apptainer exec "${EXEC_FLAGS[@]}" "$SANDBOX_DIR" bash -c "
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export TMPDIR=/tmp
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy

VENV=\"/opt/venvs/${WM_VENV_NAME}\"

echo \"Installing system Python and build tools...\"
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv python3-dev curl

echo \"Installing uv via official installer...\"
export HOME=/root
curl -sSf https://astral.sh/uv/install.sh | sh
export PATH=\"/root/.local/bin:\$PATH\"

# Create venv with Python 3.11 if possible, else fall back to system default
if uv venv --seed --python 3.11 \"\$VENV\" 2>/dev/null; then
    echo \"Created venv with Python 3.11\"
else
    echo \"Python 3.11 not available; falling back to system Python\"
    uv venv --seed \"\$VENV\"
fi

VENV_PY=\"\$VENV/bin/python\"

echo \"Installing wm_planner into \$VENV...\"
uv pip install --python \"\$VENV_PY\" \
    --no-cache-dir \
    -e '/workspace/run[dev]'

# Install external packages if present
for ext in dino vjepa2; do
    ext_dir=\"/workspace/run/external/\$ext\"
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
            --no-cache-dir -e \"\$ext_dir\" || echo \"Warning: external \$ext install failed (non-fatal)\"
    fi
done

echo \"Verifying installation...\"
\"\$VENV_PY\" -c 'import torch; print(\"torch:\", torch.__version__)'
\"\$VENV_PY\" -m pip check || echo \"Warning: pip check reported conflicts (may be non-fatal)\"
echo \"wm venv ready at \$VENV\"
"

# ---------------------------------------------------------------------------
# Convert sandbox to final SIF
# ---------------------------------------------------------------------------
echo "[local] Converting sandbox to SIF: $WM_SIF_IMAGE"
apptainer build --fakeroot "$WM_SIF_IMAGE" "$SANDBOX_DIR"

echo ""
echo "[local] Build complete!"
echo "  Output SIF : $WM_SIF_IMAGE"
echo "  wm venv    : /opt/venvs/$WM_VENV_NAME (inside container)"
echo ""
echo "To run with GPU:"
echo "  apptainer exec --nv -B $PROJECT_ROOT:/workspace/run:ro \\"
echo "    $WM_SIF_IMAGE /opt/venvs/$WM_VENV_NAME/bin/python <script>"
echo ""

exit 0
