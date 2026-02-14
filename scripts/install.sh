#!/usr/bin/env bash
mkdir logs
ssh -o StrictHostKeyChecking=accept-new burst "cd ~/rloclab && sbatch --job-name='rloclab_${USER}' --mail-user='${USER}@nyu.edu' install.slurm '$@'"
cd /scratch/$USER
git clone https://github.com/isaac-sim/IsaacLab.git
