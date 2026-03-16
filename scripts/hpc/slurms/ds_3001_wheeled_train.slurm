#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=wheeled_rl
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

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
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Isaac Sim container paths
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Node-local cache & execution
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
# Use a persistent scratch-backed cache root so pip/builds don't exhaust node-local /tmp
# Must match ds_3001_install.slurm so the pre-warmed pip/kit caches are reused
CACHE_ROOT="/scratch/$USER/singularity-cache"
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

# Scratch-backed tmp for builds and apptainer cache
SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"
export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# -------------------------------
# Wheeled RL defaults (MuSHR drift)
# -------------------------------
export ISAAC_ARGS="$*"
TASK_NAME="${TASK_NAME:-Template-Rloclab-Mushr-DriftRL-v0}"
AGENT_NAME="${AGENT_NAME:-rsl_rl_cfg_entry_point}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mushr_drift_rl}"

EXPERIMENT_NAME_GREP=$(echo "$ISAAC_ARGS" | tr -s ' ' | grep -oP '(?<=--experiment_name )\S+' || true)
if [[ -n "$EXPERIMENT_NAME_GREP" ]]; then
  EXPERIMENT_NAME="$EXPERIMENT_NAME_GREP"
fi
scontrol update JobId="${SLURM_JOB_ID}" Name="${EXPERIMENT_NAME}" 2>/dev/null || true

# WheeledLab assets path used by src/rloclab/wheeled/assets/*.py
# Override from environment as needed:
#   export RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=/path/to/WheeledLab/source/wheeledlab_assets/data
WHEELED_ASSETS_DATA_DIR="${RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR:-${RUN_DIR}/src/WheeledLab/source/wheeledlab_assets/data}"

# WandB
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

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
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env WANDB_API_KEY=${WANDB_API_KEY:-} \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_RUN_NAME=${WANDB_RUN_NAME} \
  --env TASK_NAME=${TASK_NAME} \
  --env AGENT_NAME=${AGENT_NAME} \
  --env EXPERIMENT_NAME=${EXPERIMENT_NAME} \
  --env RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=${WHEELED_ASSETS_DATA_DIR}"

# Ensure container sees node-local temp/caches and pip cache location
CMD="${CMD} \
  --env APPTAINER_TMPDIR=${APPTAINER_TMPDIR} \
  --env APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR} \
  --env TMPDIR=${SCRATCH_TMP} \
  --env PIP_CACHE_DIR=${DOCKER_USER_HOME}/.cache/pip \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw"

# XDG cache points to the bound location so all tools use the persistent cache
CMD="${CMD} \
  --env XDG_CACHE_HOME=${DOCKER_USER_HOME}/.cache"

echo "[NODE] ISAAC_ARGS=${ISAAC_ARGS}"
echo "[NODE] TASK_NAME=${TASK_NAME}"
echo "[NODE] AGENT_NAME=${AGENT_NAME}"
echo "[NODE] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[NODE] RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=${WHEELED_ASSETS_DATA_DIR}"
echo "[NODE] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unknown}"
echo "[NODE] RUN_DIR=${RUN_DIR}"

echo "[NODE] listing RUN_DIR before singularity exec:"
ls -la "${RUN_DIR}" || true

echo "[NODE] listing RUN_DIR/scripts path before singularity exec:"
ls -la "${RUN_DIR}/scripts" || true

singularity exec \
  $CMD \
  --env ISAAC_ARGS="$ISAAC_ARGS" \
  "${APPTAINER_IMAGE}" bash -lc '
set -euo pipefail

cd /workspace/isaaclab || true
export ISAACLAB_PATH=/workspace/isaaclab

# Ensure PYTHON can see mounted project code and isaaclab
export PYTHONPATH="/workspace/code/src:/workspace/isaaclab:${PYTHONPATH:-}"

echo "[APPTAINER] /tmp usage:"; df -h /tmp || true
echo "[APPTAINER] python: $(which python)"
echo "[APPTAINER] listing /workspace/code:"
ls -la /workspace/code || true
echo "[APPTAINER] checking for train script:"
ls -la /workspace/code/scripts/rl || true

echo "[APPTAINER] TASK_NAME=${TASK_NAME}"
echo "[APPTAINER] AGENT_NAME=${AGENT_NAME}"
echo "[APPTAINER] ISAAC_ARGS=${ISAAC_ARGS}"

python /workspace/code/scripts/rl/wheeledlab_rl/scripts/train_rl.py \
  --task="${TASK_NAME}" \
  --agent="${AGENT_NAME}" \
  --headless \
  ${ISAAC_ARGS}
'
