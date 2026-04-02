#!/bin/bash
# DS-GA 3001 — Install persistent venv for IsaacLab RL training
# Uses the base Isaac Sim 5.1.0 SIF on /scratch.
# Run this ONCE before submitting any wheeled_train jobs.

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --job-name=ds3001_install_isaac
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

set -euo pipefail

echo "[NODE] Ensuring IsaacLab is cloned to /scratch..."
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
  git fetch origin
  git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD)
  cd ..
else
  echo "[NODE] Cloning IsaacLab..."
  git clone https://github.com/isaac-sim/IsaacLab.git
fi

SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0_fixed.sif"
if [ ! -f "${SIF_IMAGE}" ]; then
  echo "[NODE] Base Isaac Sim SIF not found at ${SIF_IMAGE}."
  echo "[NODE] Please pull it first: singularity pull ${SIF_IMAGE} docker://nvcr.io/nvidia/isaac-sim:5.1.0"
  exit 1
fi

RUN_DIR="${SLURM_SUBMIT_DIR}"
PERSISTENT_CACHE_DIR="/scratch/$USER/isaaclab_cache"
VENV_DIR="${PERSISTENT_CACHE_DIR}/venv"
SCRATCH_TMP="/scratch/$USER/tmp"

mkdir -p "${VENV_DIR}" "${SCRATCH_TMP}" "${PERSISTENT_CACHE_DIR}/cache/pip"
export APPTAINER_TMPDIR="${SCRATCH_TMP}"

echo "[NODE] Creating IsaacLab environment in ${VENV_DIR}..."

singularity exec --nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${PERSISTENT_CACHE_DIR}/cache/pip:/root/.cache/pip:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  -B /scratch/$USER/IsaacLab:/workspace/isaaclab:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  --env TMPDIR=${SCRATCH_TMP} \
  --env PIP_CACHE_DIR=/root/.cache/pip \
  "${SIF_IMAGE}" bash -lc "
set -euo pipefail

PYTHON_EXE='/isaac-sim/python.sh'

if [ ! -f \"${VENV_DIR}/bin/activate\" ]; then
  echo '[APPTAINER] Creating venv at ${VENV_DIR}...'
  \$PYTHON_EXE -m venv --without-pip --system-site-packages \"${VENV_DIR}\"
  curl -sS https://bootstrap.pypa.io/get-pip.py | \"${VENV_DIR}/bin/python3\"
fi

source \"${VENV_DIR}/bin/activate\"
python -m pip install --upgrade pip 'setuptools<82' uv

echo '[APPTAINER] Installing main project dependencies...'
python -m uv pip install -e \"/workspace/run[dev]\" || \
  python -m uv pip install -e \"/workspace/run\" || \
  echo \"[APPTAINER] Warning: install of project root failed.\"

echo '[APPTAINER] Installing external packages (dino, vjepa2)...'
/workspace/run/scripts/install_externals.sh

echo '[APPTAINER] Installing IsaacLab packages...'
for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target=\"/workspace/isaaclab/source/\${pkg}\"
  if [ -d \"\$target\" ]; then
    python -m uv pip install -e \"\$target\"
  fi
done

echo '[APPTAINER] IsaacLab environment ready.'
"
