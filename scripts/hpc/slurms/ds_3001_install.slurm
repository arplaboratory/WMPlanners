#!/bin/bash
# Run scripts use: --overlay /scratch/$USER/my_env.ext3:ro

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
    echo "[NODE] Local changes detected, stashing to $STASH_FILE"
    git stash push -u -m "Auto-stash before update on $(date)"
    git stash show -p > "$STASH_FILE" || echo "[WARN] No diff to save in stash."
  fi
  git fetch origin
  git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD)
  cd ..
else
  echo "[NODE] Cloning IsaacLab..."
  git clone https://github.com/isaac-sim/IsaacLab.git
fi

# -----------------------------------------------
# Paths
# -----------------------------------------------
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"      # base image, never modified
OVERLAY_FILE="/scratch/$USER/my_env.ext3"             # persistent writable layer
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
RUN_DIR="${SLURM_SUBMIT_DIR}"

DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

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

PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"
mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"
export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"
mkdir -p "${HOME}/.config"

# -----------------------------------------------
# Create overlay if it doesn't exist
# -----------------------------------------------
if [ ! -f "${OVERLAY_FILE}" ]; then
  echo "[NODE] Creating overlay at ${OVERLAY_FILE} ..."
  rsync -aP /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz /scratch/$USER/
  gunzip /scratch/$USER/overlay-50G-10M.ext3.gz
  mv /scratch/$USER/overlay-50G-10M.ext3 "${OVERLAY_FILE}"
  echo "[NODE] Overlay created."
else
  echo "[NODE] Overlay already exists at ${OVERLAY_FILE}"
fi

echo "[NODE] Installing into overlay: ${OVERLAY_FILE}"

# -----------------------------------------------
# Run install inside container
#   --no-home     : avoids --containall UID-mapping bug that breaks /isaac-sim access
#   --overlay :rw : writes packages into ext3 layer; persists after container exits
# -----------------------------------------------
singularity exec --nv --no-home \
  --overlay "${OVERLAY_FILE}:rw" \
  -B "${SCRATCH_TMP}":/tmp:rw \
  -B "${CACHE_ROOT}/kit-data":"${DOCKER_ISAACSIM_ROOT_PATH}"/kit/data:rw \
  -B "${CACHE_ROOT}/cache/kit":"${DOCKER_ISAACSIM_ROOT_PATH}"/kit/cache:rw \
  -B "${CACHE_ROOT}/cache/ov":"${DOCKER_USER_HOME}"/.cache/ov:rw \
  -B "${CACHE_ROOT}/cache/pip":"${DOCKER_USER_HOME}"/.cache/pip:rw \
  -B "${CACHE_ROOT}/cache/glcache":"${DOCKER_USER_HOME}"/.cache/nvidia/GLCache:rw \
  -B "${CACHE_ROOT}/cache/computecache":"${DOCKER_USER_HOME}"/.nv/ComputeCache:rw \
  -B "${CACHE_ROOT}/logs":"${DOCKER_USER_HOME}"/.nvidia-omniverse/logs:rw \
  -B "${CACHE_ROOT}/data":"${DOCKER_USER_HOME}"/.local/share/ov/data:rw \
  -B "${CACHE_ROOT}/documents":"${DOCKER_USER_HOME}"/Documents:rw \
  -B "${CACHE_ROOT}/cache/wandb":"${DOCKER_USER_HOME}"/.cache/wandb:rw \
  -B "${HOME}/.config":"${DOCKER_USER_HOME}"/.config:rw \
  -B "${ISAACLAB_DIR}":/workspace/isaaclab:rw \
  -B "${PERSISTENT_LOGS_DIR}":/workspace/isaaclab/logs:rw \
  -B "${RUN_DIR}":/workspace/run:rw \
  "${SIF_IMAGE}" bash -s <<'SCRIPT'
set -euo pipefail

PYTHON_EXE="/isaac-sim/kit/python/bin/python3"
PROJECT_ROOT="/workspace/run"
export UV_LINK_MODE=copy   # avoid hardlink-across-filesystem errors

# -----------------------------------------------
# Install uv into the overlay
# -----------------------------------------------
echo "[CONTAINER] Installing uv..."
"$PYTHON_EXE" -m pip install --no-cache-dir uv

# Add uv's script dir to PATH so install_externals.sh can find it
SCRIPTDIR="$("$PYTHON_EXE" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
export PATH="${SCRIPTDIR}:/usr/local/bin:${PATH}"
export UV_PYTHON="$PYTHON_EXE"
echo "[CONTAINER] uv: $(command -v uv 2>/dev/null || echo 'not found on PATH')"

if [ -f "$PROJECT_ROOT/external/dino/requirements.txt" ]; then
  echo "[CONTAINER] Patching DINO requirements..."
  sed -i '/xformers==0.0.18/d'                        "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i 's/torch==2.0.0/torch>=2.0/g'               "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i 's/torchvision==0.15.0/torchvision>=0.15/g' "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i '/--extra-index-url/d'                       "$PROJECT_ROOT/external/dino/requirements.txt"
fi

# -----------------------------------------------
echo "[CONTAINER] Installing cjepa project (editable, with dev extras)..."
"$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir \
  -e "${PROJECT_ROOT}[dev]" || \
"$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir \
  -e "${PROJECT_ROOT}"

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  if [ -d "$target" ] && ( [ -f "$target/pyproject.toml" ] || [ -f "$target/setup.py" ] ); then
    echo "[CONTAINER] Installing $pkg..."
    "$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$target" || \
      echo "[WARN] Install of $pkg failed, continuing."
  else
    echo "[CONTAINER] Skipping $pkg — not found or not a Python project"
  fi
done

echo "[CONTAINER] Running install_externals.sh..."
bash "$PROJECT_ROOT/scripts/install_externals.sh"

echo "[CONTAINER] Verifying..."
"$PYTHON_EXE" - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
except Exception as e:
    print("[WARN] torch:", e)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("[WARN] numpy:", e)
try:
    import isaaclab
    print("isaaclab ->", isaaclab.__file__)
except Exception as e:
    print("[WARN] isaaclab:", e)
try:
    import cjepa
    print("cjepa ->", cjepa.__file__)
except Exception as e:
    print("[WARN] cjepa:", e)
PY

echo "[CONTAINER] Install complete."
SCRIPT

echo "[NODE] Done."
echo "[NODE] Add to your run scripts: --overlay ${OVERLAY_FILE}:ro"