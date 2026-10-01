#!/usr/bin/env bash
# scripts/update_sandbox.sh
# Update an existing wm_planner sandbox with latest Python deps,
# then optionally re-squash it into a SIF image.
#
# Usage:
#   bash src/wm_planner/scripts/update_sandbox.sh [OPTIONS]
#
# Options:
#   --sandbox DIR        Writable sandbox directory   [default: $WM_WS/wm_container]
#   --sif PATH           Output SIF path to rebuild   [default: $WM_WS/wm.sif]
#   --venv-name NAME     Venv name under /opt/venvs   [default: wm]
#   --no-squash          Skip re-squashing to SIF after update
#   --no-sync            Skip wm_planner package sync/install
#   --no-externals       Skip scripts/install_externals.sh
#   -h, --help           Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WM_WS="${WM_WS:-$(cd "$WM_ROOT/../.." && pwd)}"

SANDBOX_DIR="${SANDBOX_DIR:-$WM_WS/wm_container}"
SIF_IMAGE="${SIF_IMAGE:-$WM_WS/wm.sif}"
WM_VENV_NAME="${WM_VENV_NAME:-wm}"

DO_SQUASH=true
DO_SYNC=true
DO_EXTERNALS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sandbox) SANDBOX_DIR="$2"; shift 2 ;;
        --sif) SIF_IMAGE="$2"; shift 2 ;;
        --venv-name) WM_VENV_NAME="$2"; shift 2 ;;
        --no-squash) DO_SQUASH=false; shift ;;
        --no-sync) DO_SYNC=false; shift ;;
        --no-externals) DO_EXTERNALS=false; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

SANDBOX_DIR="$(cd "$SANDBOX_DIR" && pwd)"

if [ ! -d "$SANDBOX_DIR" ]; then
    echo "ERROR: Sandbox not found: $SANDBOX_DIR" >&2
    exit 1
fi

SCRATCH_TMP="$WM_WS/.wm_scratch"
mkdir -p "$SCRATCH_TMP"
chmod 1777 "$SCRATCH_TMP"

EXEC_FLAGS=(
    --fakeroot --writable
    -B "$SCRATCH_TMP":/tmp:rw
    -B "$WM_WS":/workspace/run:rw
)

if [ "$DO_SYNC" = true ]; then
    echo "==> Updating wm_planner package in /opt/venvs/${WM_VENV_NAME}..."
    apptainer exec "${EXEC_FLAGS[@]}" "$SANDBOX_DIR" bash -c '
        set -euo pipefail
        export TMPDIR=/tmp
        export UV_CACHE_DIR=/tmp/uv-cache
        export UV_LINK_MODE=copy
        export HOME=/root
        export PATH="/root/.local/bin:$PATH"

        WM_REPO=/workspace/run/src/wm_planner
        VENV="/opt/venvs/'"$WM_VENV_NAME"'"

        if [ ! -d "$VENV" ]; then
            echo "Venv missing at $VENV; creating..."
            if command -v uv >/dev/null 2>&1; then
                uv venv --seed --managed-python --python 3.12 "$VENV"
            else
                python3.12 -m venv "$VENV"
                "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
            fi
        fi

        VENV_PY="$VENV/bin/python"
        "$VENV_PY" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"

        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV_PY" --no-cache-dir setuptools wheel hatchling editables
            uv pip install --python "$VENV_PY" --no-cache-dir --no-build-isolation -e "$WM_REPO[dev]"
            uv pip install --python "$VENV_PY" --no-cache-dir --upgrade pyarrow datasets
        else
            "$VENV_PY" -m pip install --no-cache-dir --upgrade pip setuptools wheel
            "$VENV_PY" -m pip install --no-cache-dir -e "$WM_REPO[dev]"
            "$VENV_PY" -m pip install --no-cache-dir --upgrade pyarrow datasets
        fi

        "$VENV_PY" -m pip check || true
    '
fi

if [ "$DO_EXTERNALS" = true ]; then
    echo "==> Running wm_planner/scripts/install_externals.sh..."
    apptainer exec "${EXEC_FLAGS[@]}" "$SANDBOX_DIR" bash -c '
        set -euo pipefail
        export TMPDIR=/tmp
        export UV_CACHE_DIR=/tmp/uv-cache
        export UV_LINK_MODE=copy

        WM_REPO=/workspace/run/src/wm_planner
        VENV="/opt/venvs/'"$WM_VENV_NAME"'"
        export PYTHON_CMD="$VENV/bin/python"

        cd "$WM_REPO"
        bash scripts/install_externals.sh
    '
fi

if [ "$DO_SQUASH" = true ]; then
    echo "==> Re-squashing sandbox to $SIF_IMAGE..."
    [ -f "$SIF_IMAGE" ] && rm -f "$SIF_IMAGE"
    apptainer build --fakeroot "$SIF_IMAGE" "$SANDBOX_DIR"
    echo "==> Done. SIF updated at $SIF_IMAGE"
else
    echo "==> Done. Skipped re-squash (--no-squash was set)."
fi
