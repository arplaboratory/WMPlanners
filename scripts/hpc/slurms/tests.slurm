#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --job-name=ds3001_tests
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
PROJECT_NAME="ds_ga_3001"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
SIF_IMAGE="/scratch/$USER/isaac-sim_ds3001.sif"
SANDBOX_DIR="/scratch/$USER/isaac_container"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Isaac Sim container paths
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Node-local cache & execution
# Must match ds_3001_install.slurm so pre-warmed caches are reused
CACHE_ROOT="/scratch/$USER/singularity-cache"
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

# keep logs dir
mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

# Prefer scratch-backed tmp for apptainer/singularity scratch
export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# Prefer baked SIF; fall back to sandbox directory if SIF not yet built
if [ -f "${SIF_IMAGE}" ]; then
  APPTAINER_IMAGE="${SIF_IMAGE}"
elif [ -d "${SANDBOX_DIR}" ]; then
  echo "[NODE] SIF not found; using sandbox at ${SANDBOX_DIR}"
  APPTAINER_IMAGE="${SANDBOX_DIR}"
else
  echo "[NODE] ERROR: neither SIF (${SIF_IMAGE}) nor sandbox (${SANDBOX_DIR}) found." >&2; exit 1
fi

CMD="--nv --containall \
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
  -B ${RUN_DIR}:/workspace/code:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw"

echo "[remote] CMD to run: $CMD"

singularity exec \
  $CMD \
  "${APPTAINER_IMAGE}" bash -lc '
set -euo pipefail

export ISAACLAB_PATH=/workspace/isaaclab
export PYTHONPATH="/workspace/code/src:/workspace/isaaclab:${PYTHONPATH:-}"

echo "[test] /tmp usage:"; df -h /tmp || true

echo "[test] Verifying core packages..."
python -c "import torch; print('PyTorch ' + torch.__version__ + ' | CUDA: ' + str(torch.cuda.is_available()))"
python -c "import numpy; print('NumPy ' + numpy.__version__)"

echo "[test] Running PyTest suite..."
python -m pytest /workspace/code/tests/ -v

echo "[test] All tests completed!"
'
