#!/usr/bin/env bash
mkdir logs
ssh -o StrictHostKeyChecking=accept-new burst \
"cd ~/rloclab && \
sbatch --job-name='rloclab_${USER}' --mail-user='${USER}@nyu.edu' \
tests.slurm '$@'"
