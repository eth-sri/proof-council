#!/usr/bin/env bash
# Drive the roster sweep serially: run each attn_roster_* variant (baseline 3
# specialists + extra node(s)) on the known-answer stable_graphs problem
# (Haiku, max_rounds=4), one at a time, then grade them all with the same
# independent judge. Run ids are roster-* to stay separate from the prompt
# sweep's exp-* runs.
# Usage: scripts/run_roster_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

PROBLEM="problems/stable_graphs_M_e.tex"
VARIANTS=(counterex cleandef strategy reduce combo)
RUN_IDS=()

for name in "${VARIANTS[@]}"; do
  echo "=== running roster variant: ${name} ==="
  uv run python scripts/run_workflow.py \
    --workflow "attn_roster_${name}" \
    --problem "${PROBLEM}" \
    --input claude_model=haiku \
    --input max_rounds=4 \
    --run-id "roster-${name}" \
    --run-name "roster ${name}"
  RUN_IDS+=("roster-${name}")
  echo "=== finished roster variant: ${name} ==="
done

echo "=== all roster variants done; judging ==="
uv run python scripts/judge_attn.py "${RUN_IDS[@]}"
echo "=== roster sweep complete ==="
