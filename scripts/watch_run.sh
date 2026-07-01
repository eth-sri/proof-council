#!/usr/bin/env bash
# Poll a run until its first node is cached or the run reports finished.
# Usage: scripts/watch_run.sh RUN [PORT] [MAX_POLLS]
#   scripts/watch_run.sh claude_subscription_min___example-3-example
set -uo pipefail
RUN="${1:?usage: watch_run.sh RUN [PORT] [MAX_POLLS]}"
PORT="${2:-5005}"
MAX="${3:-60}"
cd "$(dirname "$0")/.."
for i in $(seq 1 "$MAX"); do
  n=$(ls "outputs/$RUN/resume_cache/" 2>/dev/null | wc -l | tr -d ' ')
  fin=$(curl -s "http://127.0.0.1:$PORT/run/$RUN/human-pending" \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('finished'))" 2>/dev/null)
  echo "poll $i: cache=$n finished=$fin"
  [ "${n:-0}" -ge 1 ] && { echo "GOT A CACHED NODE"; break; }
  [ "$fin" = "True" ] && { echo "FINISHED"; break; }
  sleep 3
done
