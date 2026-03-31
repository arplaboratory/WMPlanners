#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=collect0_vjepa2_exp
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

set -euo pipefail

# Load user environment variables (e.g. WANDB_API_KEY, HF_TOKEN)
# shellcheck source=/dev/null
[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="world_model"
SIF_IMAGE="/scratch/lq2146/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/world_model_cache"
PERSISTENT_LOGS_DIR="/scratch/$USER/world_model/logs/${SLURM_JOB_ID}"

DOCKER_USER_HOME="/root"

# Node-local cache & execution
SCRATCH_TMP="/scratch/$USER/tmp"
CACHE_ROOT="${PERSISTENT_CACHE_DIR}"
mkdir -p "${PERSISTENT_CACHE_DIR}" "${PERSISTENT_CACHE_DIR}/cache/pip" "${PERSISTENT_CACHE_DIR}/logs" "${PERSISTENT_CACHE_DIR}/data"
mkdir -p "${SCRATCH_TMP}"
mkdir -p \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/wandb" \
  "${CACHE_ROOT}/cache/huggingface" \
  "${CACHE_ROOT}/logs"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

VENV_DIR="${PERSISTENT_CACHE_DIR}/venv"
mkdir -p "${VENV_DIR}"

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# -------------------------------
# Experiment configuration
# -------------------------------
EXPERIMENT_NAME="collect0_vjepa2_exp"
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

scontrol update JobId="${SLURM_JOB_ID}" Name="${EXPERIMENT_NAME}" 2>/dev/null || true

CMD="--nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}/.cache/wandb:rw \
  -B ${CACHE_ROOT}/cache/huggingface:${DOCKER_USER_HOME}/.cache/huggingface:rw \
  -B ${HOME}/.config:${DOCKER_USER_HOME}/.config:rw \
  -B ${PERSISTENT_LOGS_DIR}:/workspace/worldmodel/logs:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  -B ${RUN_DIR}:/workspace/code:rw \
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env WANDB_API_KEY=${WANDB_API_KEY:-} \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_RUN_NAME=${WANDB_RUN_NAME} \
  --env HF_TOKEN=${HF_TOKEN:-} \
  --env HF_HOME=${DOCKER_USER_HOME}/.cache/huggingface \
  --env EXPERIMENT_NAME=${EXPERIMENT_NAME}"

CMD="${CMD} \
  --env APPTAINER_TMPDIR=${APPTAINER_TMPDIR} \
  --env APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR} \
  --env TMPDIR=${SCRATCH_TMP} \
  --env PIP_CACHE_DIR=${DOCKER_USER_HOME}/.cache/pip \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw"

CMD="${CMD} \
  --env XDG_CACHE_HOME=${DOCKER_USER_HOME}/.cache"

echo "[NODE] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[NODE] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[NODE] RUN_DIR=${RUN_DIR}"

singularity exec \
  $CMD \
  --env VENV_DIR="${VENV_DIR}" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

# ---- Venv bootstrap (persists in scratch cache between runs) ----
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[APPTAINER] Creating venv at ${VENV_DIR}..."
  python3 -m venv --without-pip "${VENV_DIR}" --system-site-packages
  curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3"
fi
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/workspace/run/src:${PYTHONPATH:-}"

echo "[APPTAINER] venv: $(which python)"

echo "[APPTAINER] installing /workspace/run into venv..."
pip install -e /workspace/run

echo "[APPTAINER] Starting experiment: ${EXPERIMENT_NAME}"

python /workspace/run/scripts/world_model/train_world_model.py \
  --dataset lquan9/collect0 \
  --backbone vjepa2 \
  --backbone_name vjepa2_vitl16 \
  --freeze_backbone \
  --experiment_name collect0_vjepa2_exp \
  --epochs 20 \
  --batch_size 64 \
  --lr 3e-4 \
  --horizon 8 \
  --img_size 224 \
  --wandb \
  --out_dir /workspace/worldmodel/logs/collect0_vjepa2_exp
'
