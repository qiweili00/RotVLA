#!/bin/bash
# Start the RoboTwin evaluation scheduler.
#   ./run_client.sh [config] [num_client_gpus] [tasks_per_gpu]
# Run from the simulator environment (the one containing sapien).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/../config/robotwin.yaml}"
NUM_CLIENT_GPUS="${2:-}"
TASKS_PER_GPU="${3:-}"

ARGS=("$CONFIG")
if [[ -n "$NUM_CLIENT_GPUS" ]]; then
  ARGS+=(--num-client-gpus "$NUM_CLIENT_GPUS")
fi
if [[ -n "$TASKS_PER_GPU" ]]; then
  ARGS+=(--tasks-per-gpu "$TASKS_PER_GPU")
fi
exec python -u "$SCRIPT_DIR/client.py" "${ARGS[@]}"
