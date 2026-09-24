#!/bin/bash
# Start the RotVLA inference server launcher (shared by RoboTwin and LIBERO).
#   ./run_server.sh [config/robotwin.yaml | config/libero.yaml]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config/robotwin.yaml}"
exec python -u "$SCRIPT_DIR/server.py" "$CONFIG"
