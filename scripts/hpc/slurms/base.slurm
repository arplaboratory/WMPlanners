#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Template Slurm script — generated / consumed by scripts/hpc/pipeline-py-client
# Placeholder strings are replaced by generate_slurm_script() at generation time.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=YourNameHere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
##SBATCH --account=
##SBATCH --mail-type=END
##SBATCH --mail-user=your.email@nyu.edu
##SBATCH --output=../slurm_%j.out
##SBATCH --error=../slurm_%j.err

set -euo pipefail

# ── Load user env (WANDB_API_KEY etc.) ────────────────────────────────────────
[ -f "${HOME}/.envs" ] && source "${HOME}/.envs"

# ── Paths ─────────────────────────────────────────────────────────────────────
ext3_path=YourEXT3PATH.ext3
SIF_IMAGE=YourSIFPATH.sif
RUN_DIR="${LOCAL_PROJECT:-${SLURM_SUBMIT_DIR}}"

DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# ── Persistent scratch-backed caches (unified across all run/install jobs) ────
CACHE_ROOT="/scratch/$USER/singularity-cache"
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

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

# ── Singularity bind flags ─────────────────────────────────────────────────────
CMD="--nv --containall \
  -B ${CACHE_ROOT}/kit-data:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw \
  -B ${CACHE_ROOT}/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
  -B ${CACHE_ROOT}/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
  -B ${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
  -B ${CACHE_ROOT}/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
  -B ${CACHE_ROOT}/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
  -B ${CACHE_ROOT}/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
  -B ${CACHE_ROOT}/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
  -B ${CACHE_ROOT}/documents:${DOCKER_USER_HOME}/Documents:rw \
  -B ${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}/.cache/wandb:rw \
  -B ${HOME}/.config:${DOCKER_USER_HOME}/.config:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  -B ${RUN_DIR}:/workspace/code:rw \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${SCRATCH_TMP}:/var/tmp:rw \
  --env XDG_CACHE_HOME=${DOCKER_USER_HOME}/.cache \
  --env TMPDIR=${SCRATCH_TMP} \
  --env PIP_CACHE_DIR=${DOCKER_USER_HOME}/.cache/pip"

# ── Run ───────────────────────────────────────────────────────────────────────
singularity exec \
  $CMD \
  "${SIF_IMAGE}" /bin/bash -c "
source /ext3/env.sh
export PYTHONPATH=\"/workspace/code/src:\${PYTHONPATH:-}\"
REPLACE THIS WITH YOUR COMMANDS
"
