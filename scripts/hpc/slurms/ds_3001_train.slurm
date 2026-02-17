#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

set -euo pipefail

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="rloclab"
REMOTE_HOST="burst"
LOCAL_PROJECT="${HOME}/${PROJECT_NAME}"
REMOTE_PROJECT="${HOME}/${PROJECT_NAME}"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
REMOTE_CONFIG="${HOME}/.config"
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"
RCLONE_SIF="/scratch/$USER/rclone.sif"
LOG_CSV="${LOCAL_PROJECT}/logs/runs.csv"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-go2_flat_direct}"

scp -o StrictHostKeyChecking=accept-new "${REMOTE_HOST}:${RCLONE_SIF}" "${RCLONE_SIF}"
singularity exec --bind "${LOCAL_PROJECT}" "${RCLONE_SIF}" \
  rclone copy \
  --sftp-host="${REMOTE_HOST}" \
  --sftp-user="${USER}" \
  --sftp-key-file="${HOME}/.ssh/id_rsa" \
  --sftp-ask-password=false \
  ":sftp:${REMOTE_PROJECT}/" \
  "${LOCAL_PROJECT}/"

singularity exec --bind "${ISAACLAB_DIR}" "${RCLONE_SIF}" \
  rclone copy \
  --sftp-host="${REMOTE_HOST}" \
  --sftp-user="${USER}" \
  --sftp-key-file="${HOME}/.ssh/id_rsa" \
  --sftp-ask-password=false \
  ":sftp:${ISAACLAB_DIR}/" \
  "${ISAACLAB_DIR}/"

if [ ! -d "${HOME}/.config" ]; then
  mkdir -p "${HOME}/.config"
fi

singularity exec --bind "${HOME}/.config" "${RCLONE_SIF}" \
  rclone copy \
  --sftp-host="${REMOTE_HOST}" \
  --sftp-user="${USER}" \
  --sftp-key-file="${HOME}/.ssh/id_rsa" \
  --sftp-ask-password=false \
  ":sftp:${REMOTE_CONFIG}/" \
  "${HOME}/.config/"

singularity exec --bind "/scratch/${USER}" "${RCLONE_SIF}" \
  rclone copy \
  --sftp-host="${REMOTE_HOST}" \
  --sftp-user="${USER}" \
  --sftp-key-file="${HOME}/.ssh/id_rsa" \
  --sftp-ask-password=false \
  ":sftp:${SIF_IMAGE}" \
  "/scratch/${USER}/"

# -------------------------------
# Hardcode your cluster paths
# -------------------------------
RUN_DIR="${LOCAL_PROJECT}"                       # run inside the mirrored project
PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
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

# Forward all user args to eval.py
export ISAAC_ARGS="$*"
TASK_NAME="${TASK_NAME:-Template-Rloclab-Go2-Direct-v0}"

# nasty
EXPERIMENT_NAME_GREP=$(echo "$ISAAC_ARGS" | tr -s ' ' | grep -oP '(?<=--experiment_name )\S+' || true)
if [[ -n "$EXPERIMENT_NAME_GREP" ]]; then
      EXPERIMENT_NAME="$EXPERIMENT_NAME_GREP"
fi
echo "[burst] ISAAC_ARGS=${ISAAC_ARGS}"
echo "[burst] TASK_NAME=${TASK_NAME}"
echo "[burst] EXPERIMENT_NAME=${EXPERIMENT_NAME}"

# WandB
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_API_KEY="$(cat "${REMOTE_CONFIG}/wandb/.key" 2>/dev/null || true)"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

# Execute with GPU and required binds
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
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env WANDB_PROJECT=$WANDB_PROJECT \
  --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
  --env EXPERIMENT_NAME=$EXPERIMENT_NAME \
  --env TASK_NAME=$TASK_NAME"

echo "[burst] CMD to run: $CMD"

singularity exec \
  $CMD \
  --env ISAAC_ARGS="$ISAAC_ARGS" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

cd /workspace/isaaclab
export ISAACLAB_PATH=/workspace/isaaclab

# Ensure the local package is installed in the container Python

/isaac-sim/python.sh -m pip install -e /workspace/run[dev]

echo "[singularity] ISAAC_ARGS=${ISAAC_ARGS}"

/isaac-sim/python.sh /workspace/run/scripts/rsl_rl/train.py \
  --task="${TASK_NAME}" \
  --headless \
  ${ISAAC_ARGS}

echo "[singularity] TASK_NAME=${TASK_NAME}"
echo "[singularity] EXPERIMENT_NAME=${EXPERIMENT_NAME}"

# Identify the latest-created/modified subdirectory under go2_<experiment_name>
LATEST_DIR="$(ls -td /workspace/isaaclab/logs/rsl_rl/${EXPERIMENT_NAME}/*/ 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_DIR:-}" ]]; then
  echo "[singularity] No subdirectories found under /workspace/isaaclab/logs/rsl_rl/${EXPERIMENT_NAME}" >&2
  exit 1
fi

echo "[singularity] LATEST_DIR=${LATEST_DIR}"
LATEST_DIR="${LATEST_DIR%/}"

# find latest checkpoint
LATEST_CHECKPOINT="$(ls -t ${LATEST_DIR}/model_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_CHECKPOINT:-}" ]]; then
  echo "[singularity] No checkpoint files found in ${LATEST_DIR}" >&2
  exit 1
fi

echo "[singularity] LATEST_CHECKPOINT=${LATEST_CHECKPOINT}"

# Run evaluation with the discovered checkpoint
/isaac-sim/python.sh /workspace/run/scripts/rsl_rl/play.py \
  --task="${TASK_NAME}" \
  --checkpoint "${LATEST_CHECKPOINT}" \
  --video \
  --video_length 1000 \
  --headless \

# WandB sync, TODO: we could use the built in WandB logger option
/isaac-sim/python.sh -m wandb sync \
    --sync-tensorboard \
    --project ${WANDB_PROJECT} \
    --id ${WANDB_RUN_NAME} \
    ${LATEST_DIR}/
'

rsync -az \
  --exclude='*.err' \
  --exclude='*.out' \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
  "${PERSISTENT_LOGS_DIR}/../${SLURM_JOB_ID}/" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/logs/"

RUN_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
echo "${SLURM_JOB_ID}, \
    ${RUN_TIME}, \
    ${EXPERIMENT_NAME}, \
    ${TASK_NAME}" \
    >> "${LOG_CSV}"
