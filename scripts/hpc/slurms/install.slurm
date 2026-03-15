#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --job-name=install
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

CONTAINER_PATH="/scratch/$USER/isaac_container"
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"
BUILT_SIF="/scratch/$USER/isaac-sim_ds3001.sif"  # final baked image
OVERLAY_FILE="/scratch/$USER/my_env.ext3"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/singularity"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

ROOT_PATH="/isaac-sim"
USER_HOME="/root"

# Cache and Temp setup
CACHE_ROOT="/scratch/$USER/singularity-cache"
mkdir -p "${CACHE_ROOT}/cache/pip"

SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

export ISAAC_ARGS="$*"
echo "[NODE] ISAAC_ARGS=${ISAAC_ARGS}"

if [ ! -f "${OVERLAY_FILE}" ]; then
  echo "[NODE] Overlay not found, creating ${OVERLAY_FILE} ..."
  cd /scratch/$USER
  rsync -aP /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
  gunzip overlay-50G-10M.ext3.gz
  mv overlay-50G-10M.ext3 ${OVERLAY_FILE}
  echo "[NODE] Overlay created at ${OVERLAY_FILE}"
else
  echo "[NODE] Overlay already exists at ${OVERLAY_FILE}, skipping."
fi

SANDBOX_DIR="/scratch/$USER/isaac_container"

if [ -f "${BUILT_SIF}" ]; then
  echo "[NODE] Built SIF already exists at ${BUILT_SIF}, skipping build."
else
  if [ ! -d "${SANDBOX_DIR}" ]; then
    echo "[NODE] Building sandbox from ${SIF_IMAGE} -> ${SANDBOX_DIR} ..."
    singularity build --sandbox "${SANDBOX_DIR}" "${SIF_IMAGE}"
    echo "[NODE] Sandbox build complete."
  else
    echo "[NODE] Sandbox already exists at ${SANDBOX_DIR}, reusing for install step."
  fi

  echo "[NODE] Installing packages into sandbox..."

  mkdir -p "${SANDBOX_DIR}/root/.cache/pip"
  mkdir -p "${SANDBOX_DIR}/tmp"
  mkdir -p "${SANDBOX_DIR}/scratch"
  mkdir -p "${SANDBOX_DIR}/share/apps"
  mkdir -p "${SANDBOX_DIR}/workspace/isaaclab"
  mkdir -p "${SANDBOX_DIR}/workspace/run"

  singularity exec --writable --no-home \
    -B ${CACHE_ROOT}/cache/pip:${USER_HOME}/.cache/pip:rw \
    -B ${SCRATCH_TMP}:/tmp:rw \
    -B ${ISAACLAB_DIR}:/workspace/isaaclab:rw \
    -B ${RUN_DIR}:/workspace/run:rw \
    "${SANDBOX_DIR}" bash -lc '
  set -euo pipefail
  export PYTHON_EXE="/isaac-sim/kit/python/bin/python3"
  export PROJECT_ROOT="/workspace/run"

  echo "[SANDBOX] Installing uv..."
  $PYTHON_EXE -m pip install --no-cache-dir uv

  if [ -f "$PROJECT_ROOT/external/dino/requirements.txt" ]; then
    echo "[SANDBOX] Fixing DINO requirements..."
    sed -i "/xformers==0.0.18/d" "$PROJECT_ROOT/external/dino/requirements.txt"
    sed -i "s/torch==2.0.0/torch>=2.0/g" "$PROJECT_ROOT/external/dino/requirements.txt"
    sed -i "s/torchvision==0.15.0/torchvision>=0.15/g" "$PROJECT_ROOT/external/dino/requirements.txt"
    sed -i "/--extra-index-url/d" "$PROJECT_ROOT/external/dino/requirements.txt"
  fi

  echo "[SANDBOX] Installing project dependencies..."
  $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$PROJECT_ROOT[dev]"

  for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
    $PYTHON_EXE -m pip install --no-cache-dir -e "/workspace/isaaclab/source/${pkg}"
  done

  echo "[SANDBOX] Running scripts/install_externals.sh"
  export UV_PYTHON="$PYTHON_EXE"
  bash "$PROJECT_ROOT/scripts/install_externals.sh"

  echo "[SANDBOX] Verifying install..."
  $PYTHON_EXE -c "import torch; import isaaclab; print(torch.__version__, isaaclab.__file__)"
'

  echo "[NODE] Converting sandbox -> SIF: ${BUILT_SIF} ..."
  singularity build --fakeroot "${BUILT_SIF}" "${SANDBOX_DIR}"
  echo "[NODE] SIF build complete: ${BUILT_SIF}"

  echo "[NODE] Removing sandbox to free inodes..."
  rm -rf "${SANDBOX_DIR}"
  echo "[NODE] Sandbox removed."
fi

echo "[NODE] Final SIF ready at: ${BUILT_SIF}"

echo "[NODE] Verifying built SIF: ${BUILT_SIF}"
singularity exec --nv --containall \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${ISAACLAB_DIR}:/workspace/isaaclab:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  "${BUILT_SIF}" /isaac-sim/kit/python/bin/python3 -c \
  "import torch, isaaclab; print('[verify] torch', torch.__version__, '| isaaclab', isaaclab.__file__)"