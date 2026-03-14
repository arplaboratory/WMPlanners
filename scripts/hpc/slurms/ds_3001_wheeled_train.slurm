#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

set -euo pipefail

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="rloclab"
LOCAL_PROJECT="${HOME}/${PROJECT_NAME}"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
REMOTE_CONFIG="${HOME}/.config"
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"

# -------------------------------
# Set cluster paths based on submission context
# -------------------------------
RUN_DIR="${SLURM_SUBMIT_DIR}"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Isaac Sim container paths
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Node-local cache & execution
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CACHE_ROOT="${NODE_TMP}/docker-isaac-sim"
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
touch "${PERSISTENT_LOGS_DIR}/.keep"

# Prefer node-local tmp for apptainer/singularity scratch
export APPTAINER_TMPDIR="${NODE_TMP}"
export APPTAINER_CACHEDIR="${NODE_TMP}/apptainer-cache"

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

# WheeledLab assets path used by src/rloclab/wheeled/assets/*.py
# Override from environment as needed:
#   export RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=/path/to/WheeledLab/source/wheeledlab_assets/data
WHEELED_ASSETS_DATA_DIR="${RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR:-${RUN_DIR}/src/WheeledLab/source/wheeledlab_assets/data}"

# WandB
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_API_KEY="$(cat "${REMOTE_CONFIG}/wandb/.key" 2>/dev/null || true)"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

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
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env WANDB_API_KEY=${WANDB_API_KEY} \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_RUN_NAME=${WANDB_RUN_NAME} \
  --env TASK_NAME=${TASK_NAME} \
  --env AGENT_NAME=${AGENT_NAME} \
  --env EXPERIMENT_NAME=${EXPERIMENT_NAME} \
  --env RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=${WHEELED_ASSETS_DATA_DIR}"

echo "[slurm] ISAAC_ARGS=${ISAAC_ARGS}"
echo "[slurm] TASK_NAME=${TASK_NAME}"
echo "[slurm] AGENT_NAME=${AGENT_NAME}"
echo "[slurm] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[slurm] RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR=${WHEELED_ASSETS_DATA_DIR}"

singularity exec \
  $CMD \
  --env ISAAC_ARGS="$ISAAC_ARGS" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

cd /workspace/isaaclab
export ISAACLAB_PATH=/workspace/isaaclab
export PYTHONPATH="/workspace/run/src:${PYTHONPATH:-}"

/isaac-sim/python.sh -m pip install -e /workspace/run || true

echo "[container] TASK_NAME=${TASK_NAME}"
echo "[container] AGENT_NAME=${AGENT_NAME}"
echo "[container] ISAAC_ARGS=${ISAAC_ARGS}"

/isaac-sim/python.sh /workspace/run/scripts/rsl_rl/train.py \
  --task="${TASK_NAME}" \
  --agent="${AGENT_NAME}" \
  --headless \
  ${ISAAC_ARGS}
'
