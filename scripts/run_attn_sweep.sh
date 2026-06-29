#!/usr/bin/env bash
# Drive the attention-harness prompt sweep serially: run each attn_exp_* variant
# on the known-answer stable_graphs problem (Haiku, max_rounds=4), one at a time,
# then grade them all with the independent judge.
#
# Baseline (exp-baseline) is launched separately; this driver runs the rest.
# Usage: scripts/run_attn_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

PROBLEM="problems/stable_graphs_M_e.tex"
VARIANTS=(specialists critic author synth combo)
RUN_IDS=()

for name in "${VARIANTS[@]}"; do
  echo "=== running variant: ${name} ==="
  uv run python scripts/run_workflow.py \
    --workflow "attn_exp_${name}" \
    --problem "${PROBLEM}" \
    --input claude_model=haiku \
    --input max_rounds=4 \
    --run-id "exp-${name}" \
    --run-name "exp ${name}"
  RUN_IDS+=("exp-${name}")
  echo "=== finished variant: ${name} ==="
done

echo "=== all variants done; judging ==="
uv run python scripts/judge_attn.py "${RUN_IDS[@]}"
echo "=== sweep complete ==="
