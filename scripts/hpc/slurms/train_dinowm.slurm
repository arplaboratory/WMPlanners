#!/bin/bash
#SBATCH --job-name=dinowm-train
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/dinowm_%j.out
#SBATCH --error=/scratch/%u/logs/dinowm_%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
NETID=$(whoami)
SCRATCH=/scratch/${NETID}
OVERLAY=${SCRATCH}/overlay-15GB-500K.ext3
SIF=${SCRATCH}/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Per-job cache and log directories
CACHE_ROOT="${SCRATCH}/world_model_cache"
PERSISTENT_LOGS_DIR="${SCRATCH}/world_model/logs/${SLURM_JOB_ID}"
SCRATCH_TMP="${SCRATCH}/tmp"
VENV_DIR="${CACHE_ROOT}/venv"

mkdir -p \
  "${SCRATCH}/logs" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/wandb" \
  "${CACHE_ROOT}/cache/huggingface" \
  "${CACHE_ROOT}/cache/torch" \
  "${PERSISTENT_LOGS_DIR}" \
  "${SCRATCH_TMP}"

# ── Load user secrets (WANDB_API_KEY, HF_TOKEN) ────────────────────────────────
[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# ── Experiment configuration ───────────────────────────────────────────────────
EXPERIMENT_NAME="dinowm_${SLURM_JOB_ID}"
ENCODER="dino"                      # "dino" or "vjepa" (maps to configs/dino-wm/encoder/)
BACKBONE_MODEL="dinov2_vits14"      # dinov2_vits14 | dinov2_vitb14 | vjepa2_vit_large
DATASET="lquan9/collect1"           # HF Hub ID or absolute path to local dataset
EPOCHS=50
BATCH_SIZE=16
NUM_HIST=3
NUM_PRED=1
FRAMESKIP=5
IMG_SIZE=224
LR=5e-4
NUM_WORKERS=4
NPROC=1                             # number of GPUs (adjust for multi-GPU)

TRAJ_LEN=$(( NUM_HIST + NUM_PRED ))

scontrol update JobId="${SLURM_JOB_ID}" Name="${EXPERIMENT_NAME}" 2>/dev/null || true

echo "[SLURM] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[SLURM] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[SLURM] NODE=${SLURMD_NODENAME:-unknown}"
echo "[SLURM] ENCODER=${ENCODER}  DATASET=${DATASET}"

# ── Singularity binds ──────────────────────────────────────────────────────────
DOCKER_USER_HOME="/root"

SING_CMD="--nv --containall \
  -B ${VENV_DIR}:${VENV_DIR}:rw \
  -B ${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}/.cache/wandb:rw \
  -B ${CACHE_ROOT}/cache/huggingface:${DOCKER_USER_HOME}/.cache/huggingface:rw \
  -B ${CACHE_ROOT}/cache/torch:${DOCKER_USER_HOME}/.cache/torch:rw \
  -B ${HOME}/.config:${DOCKER_USER_HOME}/.config:rw \
  -B ${PERSISTENT_LOGS_DIR}:/workspace/dinowm/logs:rw \
  -B ${SLURM_SUBMIT_DIR}:/workspace/code:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw \
  --env WANDB_API_KEY=${WANDB_API_KEY:-} \
  --env WANDB_PROJECT=dino_wm \
  --env WANDB_RUN_NAME=${EXPERIMENT_NAME} \
  --env WANDB_CACHE_DIR=${DOCKER_USER_HOME}/.cache/wandb \
  --env WANDB_CONFIG_DIR=${DOCKER_USER_HOME}/.config/wandb \
  --env HF_TOKEN=${HF_TOKEN:-} \
  --env HF_HOME=${DOCKER_USER_HOME}/.cache/huggingface \
  --env TORCH_HOME=${DOCKER_USER_HOME}/.cache/torch \
  --env PIP_CACHE_DIR=${DOCKER_USER_HOME}/.cache/pip \
  --env XDG_CACHE_HOME=${DOCKER_USER_HOME}/.cache \
  --env TMPDIR=${SCRATCH_TMP} \
  --env APPTAINER_TMPDIR=${SCRATCH_TMP} \
  --env VENV_DIR=${VENV_DIR} \
  --env EXPERIMENT_NAME=${EXPERIMENT_NAME} \
  --env ENCODER=${ENCODER} \
  --env BACKBONE_MODEL=${BACKBONE_MODEL} \
  --env DATASET=${DATASET} \
  --env EPOCHS=${EPOCHS} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_HIST=${NUM_HIST} \
  --env FRAMESKIP=${FRAMESKIP} \
  --env IMG_SIZE=${IMG_SIZE} \
  --env TRAJ_LEN=${TRAJ_LEN} \
  --env LR=${LR} \
  --env NUM_WORKERS=${NUM_WORKERS} \
  --env NPROC=${NPROC}"

# ── Launch ─────────────────────────────────────────────────────────────────────
singularity exec \
  ${SING_CMD} \
  "${SIF}" bash -lc '
set -euo pipefail

# Create venv if it does not exist, then install all requirements
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[INFO] Creating virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "[INFO] Installing requirements..."
pip install --upgrade pip -q
pip install -r /workspace/code/requirements_dinowm.txt -q

# Install local project in editable mode (no extra deps)
pip install -e /workspace/code --no-deps -q

# Patch external/dino for Python ≥ 3.10 union-type syntax
for f in \
  /workspace/code/external/dino/dinov2/layers/attention.py \
  /workspace/code/external/dino/dinov2/layers/block.py
do
  if [ -f "$f" ] && ! grep -q "from __future__ import annotations" "$f"; then
    sed -i '1s/^/from __future__ import annotations\n/' "$f"
    echo "[INFO] Patched: $f"
  fi
done

export PYTHONPATH="/workspace/code/src:${PYTHONPATH:-}"

echo "[APPTAINER] Python: $(python --version)"
echo "[APPTAINER] PyTorch: $(python -c "import torch; print(torch.__version__)")"
echo "[APPTAINER] CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "[APPTAINER] Starting training: ${EXPERIMENT_NAME}"

accelerate launch \
  --num_processes=${NPROC} \
  $([ "${NPROC}" -gt 1 ] && echo "--multi_gpu" || echo "") \
  /workspace/code/scripts/world_model/train_world_model_dinowm.py \
  "hydra.run.dir=/workspace/dinowm/logs/${EXPERIMENT_NAME}" \
  "encoder=${ENCODER}" \
  "encoder.model_name=${BACKBONE_MODEL}" \
  "env.dataset.data_path=${DATASET}" \
  "training.epochs=${EPOCHS}" \
  "training.batch_size=${BATCH_SIZE}" \
  "env.num_workers=${NUM_WORKERS}" \
  "num_hist=${NUM_HIST}" \
  "trajectory_length=${TRAJ_LEN}" \
  "frameskip=${FRAMESKIP}" \
  "img_size=${IMG_SIZE}" \
  "training.predictor_lr=${LR}" \
  "training.action_encoder_lr=${LR}"
' &

CHILD_PID=$!
trap 'kill -TERM ${CHILD_PID}; wait ${CHILD_PID}' SIGTERM SIGUSR1
wait ${CHILD_PID}
