#!/bin/bash
# DS-GA 3001 — Install venv for world model training
# Uses Ubuntu 24.04 CUDA SIF on /scratch.
# Run this ONCE before submitting any ds_3001_train_wm_*.slurm experiment.

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --job-name=ds3001_install
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

set -euo pipefail

# Load user environment variables (e.g. HF_TOKEN)
# shellcheck source=/dev/null
[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# -----------------------------------------------
# Paths
# -----------------------------------------------
SIF_IMAGE="/scratch/lq2146/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/world_model_cache"
DOCKER_USER_HOME="/root"
SCRATCH_TMP="/scratch/$USER/tmp"
VENV_DIR="${PERSISTENT_CACHE_DIR}/venv"

mkdir -p \
  "${PERSISTENT_CACHE_DIR}" \
  "${PERSISTENT_CACHE_DIR}/cache/pip" \
  "${PERSISTENT_CACHE_DIR}/cache/wandb" \
  "${PERSISTENT_CACHE_DIR}/cache/huggingface" \
  "${VENV_DIR}" \
  "${SCRATCH_TMP}"

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"
mkdir -p "${HOME}/.config"

echo "[NODE] SIF_IMAGE=${SIF_IMAGE}"
echo "[NODE] RUN_DIR=${RUN_DIR}"
echo "[NODE] VENV_DIR=${VENV_DIR}"

# -----------------------------------------------
# Run install inside container
# -----------------------------------------------
singularity exec --nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${PERSISTENT_CACHE_DIR}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${PERSISTENT_CACHE_DIR}/cache/huggingface:${DOCKER_USER_HOME}/.cache/huggingface:rw \
  -B ${HOME}/.config:${DOCKER_USER_HOME}/.config:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw \
  --env APPTAINER_TMPDIR=${APPTAINER_TMPDIR} \
  --env TMPDIR=${SCRATCH_TMP} \
  --env PIP_CACHE_DIR=${DOCKER_USER_HOME}/.cache/pip \
  --env HF_TOKEN=${HF_TOKEN:-} \
  --env HF_HOME=${DOCKER_USER_HOME}/.cache/huggingface \
  --env XDG_CACHE_HOME=${DOCKER_USER_HOME}/.cache \
  --env VENV_DIR=${VENV_DIR} \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

PROJECT_ROOT="/workspace/run"

# ---- Create or reuse venv ----
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[APPTAINER] Creating venv at ${VENV_DIR}..."
  python3 -m venv --without-pip "${VENV_DIR}" --system-site-packages
  
  echo "[APPTAINER] Bootstrapping pip..."
  curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3"
else
  echo "[APPTAINER] Reusing existing venv at ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "[APPTAINER] Python: $(which python) ($(python --version))"

# ---- Upgrade pip ----
pip install --upgrade pip "setuptools<82" wheel

# ---- Patch DINO requirements if present ----
if [ -f "$PROJECT_ROOT/external/dino/requirements.txt" ]; then
  echo "[APPTAINER] Patching DINO requirements..."
  sed -i "/xformers==0.0.18/d"                        "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "s/torch==2.0.0/torch>=2.0/g"               "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "s/torchvision==0.15.0/torchvision>=0.15/g" "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "/--extra-index-url/d"                       "$PROJECT_ROOT/external/dino/requirements.txt"
fi

# ---- Install project (editable, with dev extras) ----
echo "[APPTAINER] Installing cjepa project..."
pip install -e "${PROJECT_ROOT}[dev]" || \
  pip install -e "${PROJECT_ROOT}"

# ---- Install externals (DINO, V-JEPA2, etc.) ----
if [ -f "$PROJECT_ROOT/scripts/install_externals.sh" ]; then
  echo "[APPTAINER] Running install_externals.sh..."
  bash "$PROJECT_ROOT/scripts/install_externals.sh"
fi

# ---- Install wandb ----
pip install wandb

# ---- Verify installation ----
echo ""
echo "========== Installation Verification =========="
python - <<PY
import sys
print(f"python {sys.version.split()[0]}")

try:
    import torch
    print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[WARN] torch: {e}")

try:
    import numpy
    print(f"numpy {numpy.__version__}")
except Exception as e:
    print(f"[WARN] numpy: {e}")

try:
    from world_model.architecture import WorldModel
    print("world_model.architecture -> OK")
except Exception as e:
    print(f"[WARN] world_model: {e}")

try:
    from world_model.data import HFWindowDataset
    print("world_model.data -> OK")
except Exception as e:
    print(f"[WARN] world_model.data: {e}")

try:
    import wandb
    print(f"wandb {wandb.__version__}")
except Exception as e:
    print(f"[WARN] wandb: {e}")

try:
    import datasets
    print(f"datasets {datasets.__version__}")
except Exception as e:
    print(f"[WARN] datasets: {e}")
PY

echo "========== Install complete =========="
'

echo "[NODE] Done."
echo "[NODE] Venv installed at: ${VENV_DIR}"
echo "[NODE] Training SLURM files will reuse this venv automatically."