#!/usr/bin/env bash
set -euo pipefail

# Launch the FirstProof adapter on smoke/input.json *without* Docker.
# Usage: ./smoke/run_local.sh [fast|full]
#
# Outputs land in smoke/output/. Inspect with:
#   jq '.per_problem[] | {id,status,returncode,duration_seconds}' \
#       smoke/output/run_summary.json

VARIANT="${1:-fast}"
case "$VARIANT" in
    fast)
        WORKFLOW="firstproof_smoke_fast"
        VARIANT_BUDGET_USD=5
        ;;
    full)
        WORKFLOW="firstproof_smoke_full"
        VARIANT_BUDGET_USD=15
        ;;
    *)
        echo "Usage: $0 [fast|full]" >&2
        exit 2
        ;;
esac

SMOKE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SMOKE_DIR/.." && pwd)"
SECRETS="$SMOKE_DIR/secrets.env"
INPUT="$SMOKE_DIR/input.json"
OUTPUT="$SMOKE_DIR/output"

if [[ ! -f "$SECRETS" ]]; then
    echo "ERROR: $SECRETS not found." >&2
    echo "Copy secrets.env.example to secrets.env and fill in the keys." >&2
    exit 2
fi

mkdir -p "$OUTPUT"

# Reset any prior smoke output: stale healthcheck.json / .proceed are
# scrubbed by the entrypoint, but the *_tex files and run_summary.json
# linger and can mislead a "did the new run succeed?" inspection.
rm -f "$OUTPUT"/*.tex "$OUTPUT"/solutions.json \
      "$OUTPUT"/run_summary.json "$OUTPUT"/token_usage.jsonl
rm -rf "$OUTPUT"/logs "$OUTPUT"/workflow_runs

# Pull env vars from the secrets file without exporting comments / blanks.
set -a
# shellcheck disable=SC1090
source "$SECRETS"
set +a

export FIRSTPROOF_INPUT_PATH="$INPUT"
export FIRSTPROOF_OUTPUT_DIR="$OUTPUT"
export FIRSTPROOF_WORKFLOW="$WORKFLOW"
export FIRSTPROOF_MAX_PARALLEL="${FIRSTPROOF_MAX_PARALLEL:-6}"
export FIRSTPROOF_HEALTHCHECK="${FIRSTPROOF_HEALTHCHECK:-off}"
export MATHAGENTS_CONFIGS_ROOT="${MATHAGENTS_CONFIGS_ROOT:-$REPO_ROOT/configs}"
export PROOFSTACK_SANDBOX_BACKEND="${PROOFSTACK_SANDBOX_BACKEND:-subprocess}"
# The entrypoint forwards n_rounds + page_limit to every workflow
# subprocess via ``--input KEY=VALUE``, which would otherwise pin the
# production defaults (10 / 12) and ignore the smoke preset's smaller
# values declared in inputs:. Set them explicitly here to the smoke
# values so the run is fast and cheap as documented. Override via the
# environment if you need something different.
export FIRSTPROOF_N_ROUNDS="${FIRSTPROOF_N_ROUNDS:-2}"
export FIRSTPROOF_PAGE_LIMIT="${FIRSTPROOF_PAGE_LIMIT:-8}"
# Same story for the per-question budget: the entrypoint always passes
# ``--budget-usd ${FIRSTPROOF_BUDGET_USD_PER_QUESTION:-1000}`` which
# overrides the preset's ``budget.max_usd``. Pin it here to the smoke
# preset's intended cap so the run actually honours the cheap budget
# the README documents.
export FIRSTPROOF_BUDGET_USD_PER_QUESTION="${FIRSTPROOF_BUDGET_USD_PER_QUESTION:-$VARIANT_BUDGET_USD}"

echo "==> smoke variant:      $VARIANT"
echo "==> workflow preset:    $WORKFLOW"
echo "==> input:              $INPUT"
echo "==> output:             $OUTPUT"
echo "==> max parallel:       $FIRSTPROOF_MAX_PARALLEL"
echo

cd "$REPO_ROOT"
exec uv run python scripts/firstproof_entrypoint.py
