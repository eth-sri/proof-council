#!/usr/bin/env bash
# Start the ProofCouncil dev dashboard (foreground).
# Meant to be launched in background mode by the agent, or run directly
# in a terminal (Ctrl-C to stop). Runnable from any working directory.
# Usage: scripts/run_dashboard.sh [PORT]
set -euo pipefail
PORT="${1:-5005}"
cd "$(dirname "$0")/.."
PYTHONPATH=src PROOFCOUNCIL_DEV_PORT="$PORT" exec uv run python app/dev.py --port "$PORT"
