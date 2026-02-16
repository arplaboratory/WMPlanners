#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

set -euo pipefail

cd /scratch/$USER

python3.11 -m venv isaaclab-5.1.0
source isaaclab-5.1.0/bin/activate
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

rm -rf IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git
