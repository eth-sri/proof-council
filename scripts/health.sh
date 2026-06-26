#!/usr/bin/env bash
# Localhost dashboard health check.
# Usage: scripts/health.sh [PORT] [PATH]
#   scripts/health.sh              -> GET http://127.0.0.1:5005/
#   scripts/health.sh 5005 /runs   -> GET http://127.0.0.1:5005/runs
set -euo pipefail
PORT="${1:-5005}"
ENDPOINT="${2:-/}"
code="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}${ENDPOINT}")"
echo "${code} <- http://127.0.0.1:${PORT}${ENDPOINT}"
