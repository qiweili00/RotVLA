#!/bin/bash
# Start the LIBERO evaluation scheduler (one process per suite).
#   ./run_client.sh [config] [max_parallel]
# Run from the LIBERO environment; set LIBERO_ROOT to your LIBERO checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/../config/libero.yaml}"
MAX_PARALLEL="${2:-}"

ARGS=("$CONFIG")
if [[ -n "$MAX_PARALLEL" ]]; then
  ARGS+=(--max-parallel "$MAX_PARALLEL")
fi
exec python -u "$SCRIPT_DIR/client.py" "${ARGS[@]}"
