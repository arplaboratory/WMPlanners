#!/usr/bin/env bash
TASK_NAME="${TASK_NAME:-Template-Rloclab-Go2-Direct-v0}"
ssh -o StrictHostKeyChecking=accept-new burst \
    "cd ~/rloclab && \
    export TASK_NAME=\"$TASK_NAME\" && \
    sbatch --export=ALL,TASK_NAME --job-name='rloclab_${USER}' --mail-user='${USER}@nyu.edu' \
    train.slurm '$@'"
