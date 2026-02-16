#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename -- "${REPO_ROOT}")"

HPC_HOST="${HPC_HOST:-burst}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/${REPO_NAME}}"

TASK_NAME="${TASK_NAME:-Template-Rloclab-Go2-Direct-v0}"
TRAIN_ARGS=()

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
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

printf -v TRAIN_ARGS_STR '%q ' "${TRAIN_ARGS[@]}"
ssh -o StrictHostKeyChecking=accept-new "${HPC_HOST}" \
  "cd ${REMOTE_PROJECT_DIR@Q} && \
  export TASK_NAME=\"$TASK_NAME\" && \
  sbatch --export=ALL,TASK_NAME --job-name='rloclab_${USER}' --mail-user='${USER}@nyu.edu' \
  scripts/hpc/train.slurm ${TRAIN_ARGS_STR}"
