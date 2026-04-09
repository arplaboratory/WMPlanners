#!/bin/bash
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=EXPERIMENT_NAME_PLACEHOLDER
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

OVERLAY=/scratch/$USER/wm_env.ext3
SIF=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif
PROJECT=/home/$USER/NYU_DS_GA_3001
LOGS_DIR=/scratch/$USER/wm_experiments_v2/__EXPERIMENT_NAME__

mkdir -p "$LOGS_DIR"

# Source environment variables if available
if [ -f ~/.envs ]; then
    source ~/.envs
fi

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT}:${PROJECT}" \
    --bind "${LOGS_DIR}:${LOGS_DIR}" \
    "$SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}

        # Create temp caches to avoid home directory quota issues
        export HF_HOME=/tmp/hf_cache_\$\$
        export TORCH_HOME=/tmp/torch_cache_\$\$
        mkdir -p \$HF_HOME \$TORCH_HOME

        echo '=== Training __EXPERIMENT_NAME__ ==='
        python ${PROJECT}/train_world_model.py \
            --dataset __DATASET__ \
            --backbone __BACKBONE__ \
            --backbone_name __BACKBONE_NAME__ \
            --dynamics __DYNAMICS__ \
            --batch_size __BATCH_SIZE__ \
            --img_size __IMG_SIZE__ \
            --epochs 20 \
            --lr 3e-4 \
            --horizon 16 \
            --num_workers 8 \
            --freeze_backbone \
            --experiment_name __EXPERIMENT_NAME__ \
            --log_dir ${LOGS_DIR}

        echo '=== Evaluating __EXPERIMENT_NAME__ ==='
        python ${PROJECT}/eval_trajectory.py \
            --checkpoint ${LOGS_DIR}/latest.pt \
            --dataset __DATASET__ \
            --horizons 8 16 25 \
            --img_size __IMG_SIZE__

        echo '=== Done __EXPERIMENT_NAME__ ==='
    "
