#!/usr/bin/env bash
# Verify a run's interrupted/resumable state and that the Resume button renders.
# Usage: scripts/verify_resume.sh RUN [PORT]
#   scripts/verify_resume.sh claude_subscription_min___example-4-example
set -uo pipefail
RUN="${1:?usage: verify_resume.sh RUN [PORT]}"
PORT="${2:-5005}"
cd "$(dirname "$0")/.."

echo "--- orphaned CLI workers? ---"
ps -eo pid,command | grep -E "claude -p|codex exec" | grep -v grep | head

echo "--- run finished flag (should be False) ---"
curl -s "http://127.0.0.1:$PORT/run/$RUN/human-pending"
echo

echo "--- resume.json present? ---"
if ls "outputs/$RUN/resume.json" 2>/dev/null; then echo yes; else echo no; fi

echo "--- run-detail status (find_run resolves child + can_resume) ---"
code=$(curl -s -o /tmp/rd.html -w "%{http_code}" "http://127.0.0.1:$PORT/run/$RUN")
echo "$code <- HTTP"
echo "$(grep -c "Resume run" /tmp/rd.html) <- 'Resume run' button occurrences in page"
