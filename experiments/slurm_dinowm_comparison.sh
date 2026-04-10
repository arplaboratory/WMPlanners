#!/bin/bash
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dinowm_compare
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

OVERLAY=/scratch/$USER/wm_env.ext3
SIF=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif
PROJECT=/home/$USER/NYU_DS_GA_3001
LOGS_DIR=/scratch/$USER/wm_experiments_v2/dinowm_comparison

mkdir -p "$LOGS_DIR"

[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT}:${PROJECT}:ro" \
    --bind "${LOGS_DIR}:${LOGS_DIR}:rw" \
    "$SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT}/src:\${PYTHONPATH:-}
        export HF_HOME=/tmp/hf_cache_\$\$
        export TORCH_HOME=/tmp/torch_cache_\$\$
        mkdir -p \$HF_HOME \$TORCH_HOME

        # Install einops if not present
        pip install --quiet einops 2>/dev/null || true

        echo '=== Training DINO-WM (Akash architecture) for comparison ==='
        python ${PROJECT}/experiments/train_dinowm_comparison.py \
            --dataset lquan9/collect1 \
            --epochs 20 \
            --batch_size 8 \
            --lr 5e-4 \
            --num_hist 3 \
            --vit_depth 4 \
            --horizon 8 \
            --img_size 224 \
            --num_workers 4 \
            --out_dir ${LOGS_DIR}

        echo '=== Done ==='
    "
