#!/usr/bin/env bash
set -euo pipefail

# Helper: create two venvs inside a writable sandbox and optionally install packages.
# Usage: scripts/create_venvs_in_sandbox.sh --sandbox /path/to/sandbox [options]

SANDBOX_DIR=""
PROJECT_ROOT=""
RUNTIME_NAME="cjepa_runtime"
TRAIN_NAME="cjepa_train"
KIT_PYTHON="/isaac-sim/kit/python/bin/python3"
USE_FAKEROOt=false
INSTALL_RUNTIME_PACKAGES=false
INSTALL_TRAIN_PACKAGES=false

show_usage() {
  cat <<EOF
Usage: $0 --sandbox SANDBOX_DIR [options]

Options:
  --sandbox PATH           Path to writable sandbox directory (required)
  --project PATH           Path on host to project (will be bind-mounted to /workspace/run)
  --kit-python PATH        Path to kit python inside container (default: $KIT_PYTHON)
  --runtime-name NAME      Name for runtime venv under /opt/venvs (default: $RUNTIME_NAME)
  --train-name NAME        Name for train venv under /opt/venvs (default: $TRAIN_NAME)
  --fakeroot               Use --fakeroot when running apptainer exec
  --install-runtime        Install minimal runtime packages into runtime venv (cjepa)
  --install-train          Install training packages into train venv (pins numpy to 1.26 and installs project)
  -h, --help               Show this message

Example:
  $0 --sandbox /home/user/isaac_container --project /home/user/ds_ga_3001 --fakeroot --install-train
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox) SANDBOX_DIR="$2"; shift 2;;
    --project) PROJECT_ROOT="$2"; shift 2;;
    --kit-python) KIT_PYTHON="$2"; shift 2;;
    --runtime-name) RUNTIME_NAME="$2"; shift 2;;
    --train-name) TRAIN_NAME="$2"; shift 2;;
    --fakeroot) USE_FAKEROOt=true; shift;;
    --install-runtime) INSTALL_RUNTIME_PACKAGES=true; shift;;
    --install-train) INSTALL_TRAIN_PACKAGES=true; shift;;
    -h|--help) show_usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; show_usage; exit 2;;
  esac
done

if [[ -z "$SANDBOX_DIR" ]]; then
  echo "ERROR: --sandbox is required" >&2
  show_usage
  exit 2
fi

if [[ ! -d "$SANDBOX_DIR" ]]; then
  echo "ERROR: sandbox directory '$SANDBOX_DIR' does not exist" >&2
  exit 2
fi

# Build apptainer args
APPTAINER_ARGS=(--writable)
$USE_FAKEROOt && APPTAINER_ARGS+=(--fakeroot)

# Add bind for project if provided
if [[ -n "$PROJECT_ROOT" ]]; then
  if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: project root '$PROJECT_ROOT' does not exist" >&2
    exit 2
  fi
  BIND_PROJECT=( -B "$PROJECT_ROOT":/workspace/run:rw )
else
  BIND_PROJECT=()
fi

echo "Creating venvs inside sandbox: $SANDBOX_DIR"
echo "Kit python inside container: $KIT_PYTHON"

apptainer exec "${APPTAINER_ARGS[@]}" "${BIND_PROJECT[@]}" "$SANDBOX_DIR" bash -s <<'INNER'
set -euo pipefail
PYKIT="${KIT_PYTHON:-/isaac-sim/kit/python/bin/python3}"
RUNTIME_NAME="${RUNTIME_NAME:-cjepa_runtime}"
TRAIN_NAME="${TRAIN_NAME:-cjepa_train}"

echo "Inside container: using Python at $PYKIT"

mkdir -p /opt/venvs

create_venv() {
  local name="$1"; shift
  local path="/opt/venvs/$name"
  if [[ -d "$path" ]]; then
    echo "venv $name already exists at $path, skipping creation"
  else
    echo "Creating venv $name at $path"
    "$PYKIT" -m venv "$path"
  fi
  # Upgrade packaging tools inside venv
  "$path/bin/python" -m pip install --upgrade pip setuptools wheel
}

create_venv "$RUNTIME_NAME"
create_venv "$TRAIN_NAME"

if [[ "${INSTALL_RUNTIME_PACKAGES:-false}" = "true" ]]; then
  echo "Installing runtime packages into /opt/venvs/$RUNTIME_NAME"
  source "/opt/venvs/$RUNTIME_NAME/bin/activate"
  python -m pip install --no-cache-dir cjepa || echo "cjepa install failed (continuing)"
  deactivate
fi

if [[ "${INSTALL_TRAIN_PACKAGES:-false}" = "true" ]]; then
  echo "Installing training packages into /opt/venvs/$TRAIN_NAME"
  source "/opt/venvs/$TRAIN_NAME/bin/activate"
  # pin numpy to 1.26 for Isaac Sim compatibility
  python -m pip install --no-cache-dir "numpy~=1.26.0"
  # if project was bound to /workspace/run, install it editable
  if [[ -d /workspace/run ]]; then
    python -m pip install --no-cache-dir -e /workspace/run || echo "project install failed (continuing)"
  fi
    # Install common training externals if present (dino, vjepa2).
    for ext in dino vjepa2; do
      if [[ -d /workspace/run/external/$ext ]]; then
        echo "Installing external $ext into train venv"
        (cd /workspace/run/external/$ext && python -m pip install --no-cache-dir -e .) || {
          echo "Install $ext failed; retrying with --no-deps"
          (cd /workspace/run/external/$ext && python -m pip install --no-deps --no-cache-dir -e .) || echo "Install $ext failed (continuing)"
        }
      fi
    done

    # verify installed packages and report conflicts
    python -m pip check || echo "pip check reported issues (see above)"
  deactivate
fi

# create a small wrapper to start Isaac with the runtime venv prefixed
cat > /usr/local/bin/isaac-with-cjepa <<'WRAP'
#!/bin/bash
export PATH=/opt/venvs/${RUNTIME_NAME}/bin:$PATH
exec /isaac-sim/isaac-sim.streaming.sh "$@"
WRAP
chmod +x /usr/local/bin/isaac-with-cjepa || true

echo "Done creating venvs and wrapper"
INNER

echo "Finished: venvs created in sandbox. Verify with:"
echo "  apptainer exec --nv --containall -B $PROJECT_ROOT:/workspace/run:ro $SANDBOX_DIR /opt/venvs/$TRAIN_NAME/bin/python -c 'import numpy,cjepa; print(numpy.__version__)'"

exit 0
