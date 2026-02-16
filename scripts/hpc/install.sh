#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename -- "${REPO_ROOT}")"

HPC_HOST="${HPC_HOST:-burst}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/${REPO_NAME}}"

INSTALL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HPC_HOST="$2"
      shift 2
      ;;
    --remote-project-dir)
      REMOTE_PROJECT_DIR="$2"
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
ssh -o StrictHostKeyChecking=accept-new "${HPC_HOST}" \
  "cd ${REMOTE_PROJECT_DIR@Q} && sbatch --job-name='rloclab_${USER}' --mail-user='${USER}@nyu.edu' scripts/hpc/install.slurm ${INSTALL_ARGS_STR}"