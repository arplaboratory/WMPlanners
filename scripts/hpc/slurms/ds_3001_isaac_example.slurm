#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --job-name=isaac_gui
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

# Load user environment variables (e.g. WANDB_API_KEY)
# shellcheck source=/dev/null
[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="rloclab"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
SIF_IMAGE="/scratch/$USER/isaac-sim_ds3001.sif"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/interactive_${SLURM_JOB_ID}"

DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Must match ds_3001_install.slurm so pre-warmed caches are reused
CACHE_ROOT="/scratch/$USER/docker-isaac-sim-cache"
SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"
mkdir -p \
  "${CACHE_ROOT}/cache/kit" \
  "${CACHE_ROOT}/cache/ov" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/glcache" \
  "${CACHE_ROOT}/cache/computecache" \
  "${CACHE_ROOT}/cache/wandb" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/documents" \
  "${CACHE_ROOT}/kit-data"

mkdir -p "${PERSISTENT_LOGS_DIR}"

# Must match ds_3001_install.slurm — reuse the pre-built environment
VENV_DIR="/scratch/$USER/venvs/ds_ga_3001"
mkdir -p "${VENV_DIR}"

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# -------------------------------
# Livestream Tunnel Instructions
# -------------------------------
COMPUTE_NODE=$(hostname)

echo "=========================================================================="
echo "                   ISAAC LAB INTERACTIVE LIVESTREAM RUNNING                "
echo "=========================================================================="
echo ""
echo "To view the stream on your local browser, open a terminal on your laptop"
echo "and run the following SSH port-forwarding command:"
echo ""
echo "    ssh -N -L 8211:${COMPUTE_NODE}:8211 ${USER}@dtn.torch.hpc.nyu.edu"
echo ""
echo "Then, open your web browser (Chrome/Firefox) and navigate to:"
echo "    http://localhost:8211/streaming/webrtc-demo/"
echo ""
echo "=========================================================================="

CMD="--nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${CACHE_ROOT}/kit-data:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw \
  -B ${CACHE_ROOT}/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
  -B ${CACHE_ROOT}/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
  -B ${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${CACHE_ROOT}/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
  -B ${CACHE_ROOT}/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
  -B ${CACHE_ROOT}/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
  -B ${CACHE_ROOT}/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
  -B ${CACHE_ROOT}/documents:${DOCKER_USER_HOME}/Documents:rw \
  -B ${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}/.cache/wandb:rw \
  -B ${HOME}/.config:${DOCKER_USER_HOME}/.config:rw \
  -B ${ISAACLAB_DIR}:/workspace/isaaclab:rw \
  -B ${PERSISTENT_LOGS_DIR}:/workspace/isaaclab/logs:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw \
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env WANDB_API_KEY=${WANDB_API_KEY:-} \
  --env LIVESTREAM=1 \
  --env ENABLE_CAMERAS=1"

  echo "[NODE] creating import check script in SCRATCH_TMP: ${SCRATCH_TMP}"
  cat > "${SCRATCH_TMP}/isaac_import_check.sh" <<'SH'
#!/bin/bash
set -euo pipefail
source "${VENV_DIR}/bin/activate"
export ISAACLAB_PATH=/workspace/isaaclab
export PYTHONPATH="/workspace/isaaclab:${PYTHONPATH:-}"
echo "[APPTAINER] listing /workspace/isaaclab inside container:"
ls -la /workspace/isaaclab || true
echo "[APPTAINER] trying to import isaaclab inside venv (will not fail job):"
python - <<'PYCODE'
import importlib, sys, traceback
try:
  m = importlib.import_module('isaaclab')
  print('isaaclab imported from', getattr(m, '__file__', repr(m)))
except Exception:
  traceback.print_exc()
PYCODE
SH
  chmod +x "${SCRATCH_TMP}/isaac_import_check.sh" || true

  singularity exec \
  $CMD \
  --env VENV_DIR="$VENV_DIR" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

echo "[APPTAINER] /tmp usage:"; df -h /tmp || true

export ISAACLAB_PATH=/workspace/isaaclab

# ---- Venv bootstrap (persists in scratch cache between runs) ----
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[APPTAINER] Creating venv at ${VENV_DIR}..."
  /isaac-sim/python.sh -m venv "${VENV_DIR}" --system-site-packages
fi
source "${VENV_DIR}/bin/activate"

# Ensure Python can import the isaaclab package from the mounted workspace
export PYTHONPATH="/workspace/isaaclab:${PYTHONPATH:-}"

if [ -d /workspace/run ]; then
  echo "[APPTAINER] Installing /workspace/run into venv..."
  pip install -e /workspace/run || true
fi

# Run host-created import-check script (written to SCRATCH_TMP, bound to /tmp inside container)
if [ -x /tmp/isaac_import_check.sh ]; then
  /tmp/isaac_import_check.sh || true
else
  echo "[APPTAINER] import check script not found in /tmp"
fi

echo "[APPTAINER] Launching Isaac Lab Test GUI (WebRTC on port 8211)..."

# Run the official empty testing environment from Isaac Lab
python /workspace/isaaclab/scripts/tutorials/00_sim/create_empty.py 2>&1 | sed -n '1,200p'
rc=${PIPESTATUS[0]:-0}
echo "[APPTAINER] create_empty.py exit code: ${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[APPTAINER] create_empty.py failed; printing last 400 lines of logs (if any):" >&2
  tail -n 400 /root/.nvidia-omniverse/logs/* 2>/dev/null || true
  exit ${rc}
fi
'
