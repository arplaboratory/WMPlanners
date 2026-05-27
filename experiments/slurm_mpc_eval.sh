#!/bin/bash
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mpc_eval
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

OVERLAY=/scratch/$USER/wm_env.ext3
SIF=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif
PROJECT=/home/$USER/NYU_DS_GA_3001

[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# Run MPC eval on best model (DINOv2+RSSM+unfiltered) and CNN baseline
for EXPERIMENT in dino_rssm_c1u cnn_mlp_c1u; do
    CKPT=/scratch/$USER/wm_experiments_v2/${EXPERIMENT}/latest.pt
    DATASET="lquan9/collect1_unfiltered"
    IMG_SIZE=224
    if [[ "$EXPERIMENT" == cnn* ]]; then
        IMG_SIZE=128
    fi

    echo "=== MPC eval: ${EXPERIMENT} ==="

    singularity exec --nv \
        --overlay "${OVERLAY}:ro" \
        --bind "${PROJECT}:${PROJECT}:ro" \
        --bind "/scratch/$USER:/scratch/$USER:rw" \
        "$SIF" \
        /bin/bash -c "
            source /ext3/env.sh
            export PYTHONPATH=${PROJECT}/src:\${PYTHONPATH:-}
            export HF_HOME=/tmp/hf_cache_\$\$
            export TORCH_HOME=/tmp/torch_cache_\$\$
            mkdir -p \$HF_HOME \$TORCH_HOME

            python ${PROJECT}/scripts/world_model/eval_mpc_replay.py \
                --checkpoint ${CKPT} \
                --dataset ${DATASET} \
                --horizon 16 \
                --num_episodes 50 \
                --cem_population 128 \
                --cem_iters 5 \
                --img_size ${IMG_SIZE}
        "
    echo ""
done

echo "=== All MPC evals complete ==="
