#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="${HPC_HOST:-burst}"
HPC_PROJECT_ROOT="${HPC_PROJECT_ROOT:-${HOME}/ds_ga_3001}"
REPO_NAME="$(basename -- "${HPC_PROJECT_ROOT}")"
INSTALL_SLURM="${INSTALL_SLURM:-scripts/hpc/install.slurm}"
SCRATCH_DIR="/scratch/$USER"
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
      HPC_PROJECT_ROOT="$2"
      shift 2
      ;;
    --install-slurm)
      INSTALL_SLURM=${HOME}/${REPO_NAME}/"$2"
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

rclone copy \
  "${SCRATCH_DIR}/${RCLONE_SIF}" \
  ":sftp,host=${HPC_HOST},user=${USER}:${SCRATCH_DIR}/"
rclone copy \
  "${HPC_PROJECT_ROOT}" \
  ":sftp,host=${HPC_HOST},user=${USER}:${HOME}/${REPO_NAME}"
rclone copy \
  "${SCRATCH_DIR}/${ISAAC_SIF}" \
  ":sftp,host=${HPC_HOST},user=${USER}:${SCRATCH_DIR}/"

ssh -o StrictHostKeyChecking=accept-new "${HPC_HOST}" \
  "cd ~/${REPO_NAME} && sbatch --job-name="install_${USER}" --mail-user="${USER}@nyu.edu" ${INSTALL_SLURM@Q} ${INSTALL_ARGS_STR}"