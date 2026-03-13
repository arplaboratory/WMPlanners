#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1

set -euo pipefail

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="rloclab"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/interactive_${SLURM_JOB_ID}"

DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

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

VENV_DIR="/scratch/$USER/venvs/ds_ga_3001"

export APPTAINER_TMPDIR="${NODE_TMP}"
export APPTAINER_CACHEDIR="${NODE_TMP}/apptainer-cache"

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
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env LIVESTREAM=1 \
  --env ENABLE_CAMERAS=1"

singularity exec \
  $CMD \
  --env VENV_DIR="$VENV_DIR" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

source "${VENV_DIR}/bin/activate"
export ISAACLAB_PATH=/workspace/isaaclab

echo "[Interactive] Launching Isaac Lab Test GUI (WebRTC on port 8211)..."

# Run the official empty testing environment from Isaac Lab
/isaac-sim/python.sh /workspace/isaaclab/source/standalone/tutorials/00_sim/create_empty.py
'
