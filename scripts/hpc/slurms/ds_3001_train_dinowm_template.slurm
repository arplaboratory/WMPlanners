#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --gres={{GRES}}
#SBATCH --job-name={{EXPERIMENT_NAME}}
#SBATCH --mail-type=END
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
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
mkdir -p "${PERSISTENT_CACHE_DIR}" "${PERSISTENT_CACHE_DIR}/cache/pip" "${PERSISTENT_CACHE_DIR}/logs"
mkdir -p "${SCRATCH_TMP}"
mkdir -p \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/wandb" \
  "${CACHE_ROOT}/cache/huggingface" \
  "${CACHE_ROOT}/cache/torch" \
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
EXPERIMENT_NAME="{{EXPERIMENT_NAME}}"
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

scontrol update JobId="${SLURM_JOB_ID}" Name="${EXPERIMENT_NAME}" 2>/dev/null || true

CMD="--nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}/.cache/wandb:rw \
  -B ${CACHE_ROOT}/cache/huggingface:${DOCKER_USER_HOME}/.cache/huggingface:rw \
  -B ${CACHE_ROOT}/cache/torch:${DOCKER_USER_HOME}/.cache/torch:rw \
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
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TORCH_HOME=${DOCKER_USER_HOME}/.cache/torch \
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

# ---- Activate venv (created by ds_3001_install.slurm) ----
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[ERROR] Virtual environment not found at ${VENV_DIR}."
  echo "Please run: sbatch scripts/hpc/slurms/ds_3001_install.slurm"
  exit 1
fi
source "${VENV_DIR}/bin/activate"
echo "[APPTAINER] Fast project re-install..."
pip install -e /workspace/run --no-deps
export PYTHONPATH="/workspace/run/src:${PYTHONPATH:-}"

echo "[APPTAINER] Starting experiment: ${EXPERIMENT_NAME}"

# Compute trajectory_length = num_hist + num_pred (num_pred is always 1 in dino-wm)
DINOWM_HORIZON={{HORIZON}}
DINOWM_TRAJ_LEN=$(( DINOWM_HORIZON + 1 ))

# Translate GUI backbone choice to the Hydra encoder config-group name.
# The config group files live under configs/dino-wm/encoder/:
#   encoder/dino.yaml   (_target_: world_model.backbones.dino.build_dinov2)
#   encoder/vjepa.yaml  (_target_: world_model.backbones.vjepa.build_vjepa2)
# The GUI populates BACKBONE as "dinov2" or "vjepa2", so we map here.
case "{{BACKBONE}}" in
  dinov2) DINOWM_ENCODER="dino"  ;;
  vjepa2) DINOWM_ENCODER="vjepa" ;;
  *)      DINOWM_ENCODER="{{BACKBONE}}" ;;  # already the config-group name
esac

# dino-wm uses Hydra config; all hyperparams are Hydra key=value overrides.
# {{EXTRA_FLAGS}} must also be Hydra overrides (e.g. model.train_encoder=False),
# NOT argparse-style flags like --freeze_backbone or --dynamics_type.

accelerate launch --num_processes={{NPROC}} --multi_gpu \
  /workspace/run/scripts/world_model/train_world_model_dinowm.py \
  "hydra.run.dir=/workspace/worldmodel/logs/{{EXPERIMENT_NAME}}" \
  "env.dataset.data_path={{DATASET}}" \
  "encoder=${DINOWM_ENCODER}" \
  "encoder.model_name={{BACKBONE_NAME}}" \
  "training.epochs={{EPOCHS}}" \
  "training.batch_size={{BATCH_SIZE}}" \
  "env.num_workers={{NUM_WORKERS}}" \
  "training.predictor_lr={{LR}}" \
  "training.action_encoder_lr={{LR}}" \
  "training.decoder_lr={{LR}}" \
  "num_hist=${DINOWM_HORIZON}" \
  "trajectory_length=${DINOWM_TRAJ_LEN}" \
  "img_size={{IMG_SIZE}}" \
  {{EXTRA_FLAGS}}
'
