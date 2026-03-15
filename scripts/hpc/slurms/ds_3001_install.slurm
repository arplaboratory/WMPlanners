### Deprecated since we're not caching venv in /scratch

#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --job-name=ds3001_install
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

set -euo pipefail

cd /scratch/$USER

if [ -d IsaacLab/.git ]; then
  echo "[NODE] IsaacLab exists, updating safely..."
  cd IsaacLab
  if ! git diff --quiet || ! git diff --cached --quiet; then
    STASH_FILE="../isaaclab_stash_$(date +%Y%m%d_%H%M%S).patch"
    echo "[NODE] Local changes detected, stashing and saving to $STASH_FILE"
    git stash push -u -m "Auto-stash before update on $(date)"
    git stash show -p > "$STASH_FILE" || echo "[WARN] No diff to save in stash."
  fi
  # Fetch and fast-forward only
  git fetch origin
  git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD)
  cd ..
else
  echo "[NODE] Cloning IsaacLab..."
  git clone https://github.com/isaac-sim/IsaacLab.git
fi

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="ds_ga_3001"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
SIF_IMAGE="/scratch/$USER/isaac-sim_ds3001.sif"

mkdir -p "${HOME}/.config"

# -------------------------------
# Set cluster paths based on submission context
# -------------------------------
RUN_DIR="${SLURM_SUBMIT_DIR}"
PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Isaac Sim container paths
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Node-local cache & execution
CACHE_ROOT="/scratch/$USER/docker-isaac-sim-cache"
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

# Python envs
VENV_DIR="/scratch/$USER/venvs/ds_ga_3001"
mkdir -p "${VENV_DIR}"

BUILD_TMP="/scratch/$USER/pip_build_tmp"
mkdir -p "$BUILD_TMP"

SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"
export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"
export ISAAC_ARGS="$*"

echo "[NODE] ISAAC_ARGS=${ISAAC_ARGS}"

# Execute with GPU and required binds
CMD="--nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${BUILD_TMP}:/tmp:rw \
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
  "

echo "[NODE] CMD to run: $CMD"

singularity exec \
  $CMD \
  --env ISAAC_ARGS="$ISAAC_ARGS" \
  --env VENV_DIR="$VENV_DIR" \
  --env TMPDIR="/tmp" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

if [ ! -d "${VENV_DIR}/bin" ]; then
    echo "[APPTAINER] Creating venv with Python"
    /isaac-sim/python.sh -m venv ${VENV_DIR} --system-site-packages
fi
source "${VENV_DIR}/bin/activate"

cd /workspace/isaaclab
export ISAACLAB_PATH=/workspace/isaaclab

# Ensure the local package is installed in the container Python

echo "[APPTAINER] Disk space inside container:"
df -h /tmp
df -h ${VENV_DIR}

pip install --upgrade pip
pip install -e /workspace/run[dev]

echo "Using: $(which python)"
echo "[APPTAINER] ISAAC_ARGS=${ISAAC_ARGS}"

# -------------------------------------------------------------
# Post-Install Verification Tests
# -------------------------------------------------------------
echo "[APPTAINER] Verifying core packages..."
python -c "import torch; print(\"PyTorch \" + torch.__version__ + \" | CUDA Available: \" + str(torch.cuda.is_available()))"
python -c "import numpy; print(\"NumPy \" + numpy.__version__)"

# echo "[APPTAINER] Running PyTest suite..."
# python -m pytest /workspace/run/tests/ -v

echo "[APPTAINER] Environment successfully built and verified!"
'