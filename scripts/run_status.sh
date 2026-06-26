#!/usr/bin/env bash
# One-shot status snapshot for a run: cache entries, finished flag,
# latest batch dashboard log tail, and live python processes.
# Usage: scripts/run_status.sh RUN [PORT]
#   scripts/run_status.sh claude_only_min___example-3-example
set -uo pipefail
RUN="${1:?usage: run_status.sh RUN [PORT]}"
PORT="${2:-5005}"
cd "$(dirname "$0")/.."
echo "cache entries:"
ls "outputs/$RUN/resume_cache/" 2>/dev/null
echo "--- finished? ---"
curl -s "http://127.0.0.1:$PORT/run/$RUN/human-pending"
echo
echo "--- batch dashboard log ---"
log=$(ls -t outputs/*/dashboard-subprocess.log 2>/dev/null | head -1)
[ -n "${log:-}" ] && tail -8 "$log"
echo "--- python procs ---"
ps -eo pid,command | grep -i python | grep -v grep | head
