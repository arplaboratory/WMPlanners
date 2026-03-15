#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --account=
#SBATCH --partition=n2c48m24
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

set -euo pipefail

PROJECT_NAME="base"

SIF_IMAGE="/scratch/$USER/isaac-sim_ds3001.sif"
RUN_DIR="${LOCAL_PROJECT:-${SLURM_SUBMIT_DIR}}"   # run inside the mirrored project
PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Node-local cache & execution
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
# Use persistent scratch-backed cache root so builds/pip do not exhaust node-local /tmp
CACHE_ROOT="${PERSISTENT_CACHE_DIR}"
# Ensure persistent cache and a scratch tmp exist and are writable
mkdir -p "${PERSISTENT_CACHE_DIR}" "${PERSISTENT_CACHE_DIR}/cache/pip" "${PERSISTENT_CACHE_DIR}/logs" "${PERSISTENT_CACHE_DIR}/data"
SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"
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

# Prefer scratch-backed tmp for apptainer/singularity scratch
export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"
