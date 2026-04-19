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

# ── Environment ────────────────────────────────────────────────────────────────
NETID=$(whoami)
SCRATCH=/scratch/${NETID}
OVERLAY=${SCRATCH}/overlay-15GB-500K.ext3
SIF=${SCRATCH}/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# ── Experiment configuration ───────────────────────────────────────────────────
ENCODER="dino"                 # "dino" or "vjepa"
BACKBONE_MODEL="dinov2_vits14" # dinov2_vits14 | dinov2_vitb14 | vjepa2_vit_large
DATASET="lquan9/collect1"      # HF Hub ID or absolute path to local dataset
EPOCHS=50
BATCH_SIZE=16
NUM_HIST=3
NUM_PRED=1
FRAMESKIP=5
IMG_SIZE=224
LR=5e-4
NUM_WORKERS=4
NPROC=1

TRAJ_LEN=$(( NUM_HIST + NUM_PRED ))

mkdir -p ${SCRATCH}/logs

# ── Launch via Singularity ─────────────────────────────────────────────────────
singularity exec \
    --nv \
    --overlay ${OVERLAY}:rw \
    ${SIF} \
    /bin/bash -c "

        export WANDB_API_KEY=\$(cat ~/.wandb_key 2>/dev/null || echo '')
        export HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null || echo '')
        export PYTHONPATH=/scratch/${NETID}/NYU_DS_GA_3001/src:\$PYTHONPATH
        export PATH=\$HOME/.local/bin:\$PATH

        cd /scratch/${NETID}/NYU_DS_GA_3001

        # Install requirements and local package
        pip install --user --break-system-packages -q -r requirements_dinowm.txt
        pip install --user --break-system-packages -q --no-deps -e .

        # Patch external/dino for Python >= 3.10 union-type syntax
        for f in external/dino/dinov2/layers/attention.py external/dino/dinov2/layers/block.py; do
            if [ -f \"\$f\" ] && ! grep -q 'from __future__ import annotations' \"\$f\"; then
                sed -i '1s/^/from __future__ import annotations\n/' \"\$f\"
            fi
        done

        accelerate launch \
            --num_processes=${NPROC} \
            scripts/world_model/train_world_model_dinowm.py \
            encoder=${ENCODER} \
            encoder.model_name=${BACKBONE_MODEL} \
            env.dataset.data_path=${DATASET} \
            training.epochs=${EPOCHS} \
            training.batch_size=${BATCH_SIZE} \
            env.num_workers=${NUM_WORKERS} \
            num_hist=${NUM_HIST} \
            trajectory_length=${TRAJ_LEN} \
            frameskip=${FRAMESKIP} \
            img_size=${IMG_SIZE} \
            training.predictor_lr=${LR} \
            training.action_encoder_lr=${LR}
    " &

CHILD_PID=$!
trap 'kill -TERM ${CHILD_PID}; wait ${CHILD_PID}' SIGTERM SIGUSR1
wait ${CHILD_PID}
