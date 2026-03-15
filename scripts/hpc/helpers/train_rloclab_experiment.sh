#!/usr/bin/env bash
set -euo pipefail

# Run from repository root or pass explicit paths.

TASK_NAME="${TASK_NAME:-Template-Rloclab-Go2-Direct-v0}"
AGENT_ENTRY="${AGENT_ENTRY:-rsl_rl_cfg_entry_point}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
NUM_ENVS="${NUM_ENVS:-}"
HEADLESS="${HEADLESS:-1}"

CMD=(python src/rloclab/scripts/rsl_rl/train.py --task "$TASK_NAME" --agent "$AGENT_ENTRY")

if [[ -n "$MAX_ITERATIONS" ]]; then
  CMD+=(--max_iterations "$MAX_ITERATIONS")
fi

if [[ -n "$NUM_ENVS" ]]; then
  CMD+=(--num_envs "$NUM_ENVS")
fi

if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[train_rloclab_experiment] ${CMD[*]}"
"${CMD[@]}"

