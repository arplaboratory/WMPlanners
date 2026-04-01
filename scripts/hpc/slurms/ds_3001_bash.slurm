#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --job-name=ds3001_bash
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err
#SBATCH --gres=gpu:2

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
SANDBOX_DIR="/scratch/$USER/isaac_container"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"

DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

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

# keep no runtime venv in baked-SIF workflow; code will be bound to /workspace/code

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# -------------------------------
# Connection instructions
# -------------------------------
COMPUTE_NODE=$(hostname)

echo "=========================================================================="
echo "               INTERACTIVE BASH SESSION (${COMPUTE_NODE})                 "
echo "=========================================================================="
echo ""
echo "Your Singularity container is running with the venv activated."
echo "To connect from another terminal, SSH into the compute node:"
echo ""
echo "    ssh ${COMPUTE_NODE}"
echo ""
echo "The session will stay alive until the job walltime expires or you cancel it."
echo "=========================================================================="

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
  -B ${RUN_DIR}:/workspace/run:rw \
  -B ${RUN_DIR}:/workspace/code:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw \
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env WANDB_API_KEY=${WANDB_API_KEY:-}"

singularity exec \
  $CMD \
  "${APPTAINER_IMAGE}" bash -lc '
set -euo pipefail

export ISAACLAB_PATH=/workspace/isaaclab

# Make sure Python can find the mounted project code
export PYTHONPATH="/workspace/code/src:/workspace/isaaclab:${PYTHONPATH:-}"

echo "[APPTAINER] python: $(which python)"
echo "[APPTAINER] Python version: $(python --version)"
echo "[APPTAINER] Sleeping until job expires. Use ssh to connect and work interactively."

sleep infinity
'
