#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --account=rob_gy6323-2025fa
#SBATCH --partition=n2c48m24
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

cp /scratch/ja5009/public/isaac-lab-base.tar /scratch/$USER/

cd /scratch/$USER
tar xf isaac-lab-base.tar
