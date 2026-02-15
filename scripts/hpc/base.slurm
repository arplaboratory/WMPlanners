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

set -euo pipefail

PROJECT_NAME="rloclab"
REMOTE_HOST="greene-dtn"
LOCAL_PROJECT="${HOME}/${PROJECT_NAME}"
REMOTE_PROJECT="${HOME}/${PROJECT_NAME}"

SIF_IMAGE="/scratch/$USER/isaac-lab-base.sif"
RUN_DIR="${LOCAL_PROJECT}"                       # run inside the mirrored project
PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Node-local cache & execution
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CACHE_ROOT="${NODE_TMP}/docker-isaac-sim"
mkdir -p \
  "${CACHE_ROOT}/cache/kit" \
  "${CACHE_ROOT}/cache/ov" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/glcache" \
  "${CACHE_ROOT}/cache/computecache" \
  "${CACHE_ROOT}/cache/wandb" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/documents" \
  "${CACHE_ROOT}/kit-data"

# Prefer node-local tmp for apptainer/singularity scratch
export APPTAINER_TMPDIR="${NODE_TMP}"
export APPTAINER_CACHEDIR="${NODE_TMP}/apptainer-cache"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"
