#!/usr/bin/env bash
# Screenshot the dashboard's key UI states for LLM/manual review.
# Requires the dashboard to be running (scripts/run_dashboard.sh [PORT]).
# Usage: scripts/ui_review.sh [PORT] [--probe]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # uv lives here on fresh installs
PORT=5005
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    PORT="$1"
    shift
fi
cd "$(dirname "$0")/.."
exec uv run python scripts/ui_review.py --port "$PORT" "$@"
