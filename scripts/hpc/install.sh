#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="${HPC_HOST:-burst}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ds_ga_3001}"
REPO_NAME="$(basename -- "${PROJECT_ROOT}")"
INSTALL_SLURM="${INSTALL_SLURM:-scripts/hpc/install.slurm}"
SCRATCH_DIR="/scratch/lq2146" # this is where we get our sif files from
ISAAC_SIF="isaac-sim_5.1.0.sif"
RCLONE_SIF="rclone.sif"

INSTALL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HPC_HOST="$2"
      shift 2
      ;;
    --remote-project-dir)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --install-slurm)
      INSTALL_SLURM=${PROJECT_ROOT}/"$2"
      shift 2
      ;;
    *)
      INSTALL_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p logs
printf -v INSTALL_ARGS_STR '%q ' "${INSTALL_ARGS[@]}"

# Isaac Sim check
if [ ! -f "$SCRATCH_DIR/$ISAAC_SIF" ]; then
  echo "[INFO] $ISAAC_SIF not found in $SCRATCH_DIR. Exiting ..."
  exit 1
fi

# if HPC_HOST is burst or from a different user, copy files over
if [ "$HPC_HOST" = "burst" ] || [ "$USER" != "lq2146" ]; then
  rclone copy \
    "${SCRATCH_DIR}/${RCLONE_SIF}" \
    ":sftp,host=${HPC_HOST},user=${USER}:${SCRATCH_DIR}/"
  rclone copy \
    "${PROJECT_ROOT}" \
    ":sftp,host=${HPC_HOST},user=${USER}:${PROJECT_ROOT}"
  rclone copy \
    "${SCRATCH_DIR}/${ISAAC_SIF}" \
    ":sftp,host=${HPC_HOST},user=${USER}:${SCRATCH_DIR}/"
fi

# copy the code over if not on hpc (e.g., burst) (TODO)
# rsync -avz --delete ${PROJECT_ROOT} ${USER}@${HPC_HOST}:~/${REPO_NAME}

# ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "${HPC_HOST}" \
#   "cd ~/${REPO_NAME} && sbatch --job-name="install_${USER}" --mail-user="${USER}@nyu.edu" ${INSTALL_SLURM@Q} ${INSTALL_ARGS_STR}"

echo "=== DEBUG INFO ==="
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "INSTALL_SLURM: ${INSTALL_SLURM}"
echo "PWD: $(pwd)"
echo "=================="

cd ${PROJECT_ROOT} && eval "sbatch --job-name=\"install_${USER}\" --mail-user=\"${USER}@nyu.edu\" ${INSTALL_SLURM@Q} ${INSTALL_ARGS_STR}"