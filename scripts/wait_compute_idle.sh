#!/usr/bin/env bash
# Block until no ProofCouncil compute worker (codex exec) is running.
# Usage: scripts/wait_compute_idle.sh [POLL_SECONDS]
set -euo pipefail
POLL="${1:-30}"
while pgrep -f "codex exec -m" >/dev/null 2>&1; do
  sleep "$POLL"
done
echo "no compute worker running"
