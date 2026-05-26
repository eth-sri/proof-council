#!/usr/bin/env bash
set -euo pipefail

# Build the production Docker image and run the FirstProof adapter on
# smoke/input.json inside it. Mirrors how First Proof's run.sh wires
# /data/input and /data/output on the EC2 instance.
#
# Usage: ./smoke/run_container.sh [fast|full]

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
OUTPUT="$SMOKE_DIR/output_container"
IMAGE_TAG="${IMAGE_TAG:-mathagents-smoke:latest}"

if [[ ! -f "$SECRETS" ]]; then
    echo "ERROR: $SECRETS not found." >&2
    echo "Copy secrets.env.example to secrets.env and fill in the keys." >&2
    exit 2
fi

mkdir -p "$OUTPUT"
rm -f "$OUTPUT"/*.tex "$OUTPUT"/solutions.json \
      "$OUTPUT"/run_summary.json "$OUTPUT"/token_usage.jsonl
rm -rf "$OUTPUT"/logs "$OUTPUT"/workflow_runs

# Build a temporary env-file that combines the user's secrets.env with
# the smoke-variant defaults. Docker's ``--env-file`` is last-wins, so
# blindly appending after the secrets.env copy would silently *shadow*
# any FIRSTPROOF_* override the user explicitly set in secrets.env
# (e.g. ``FIRSTPROOF_BUDGET_USD_PER_QUESTION=2.0``). Instead, only
# append a default when the same key isn't already present in
# secrets.env. Two settings are always-force because they're variant-
# specific or required by the harness: FIRSTPROOF_WORKFLOW (selected
# by the script's positional arg) and PROOFSTACK_SANDBOX_BACKEND
# (must be subprocess inside the container — no docker-in-docker).
TMP_ENV="$(mktemp)"
TMP_ENV_FILTERED="$(mktemp)"
trap 'rm -f "$TMP_ENV" "$TMP_ENV_FILTERED"' EXIT
cp "$SECRETS" "$TMP_ENV"

# Strip any always-force keys the user happened to also set in
# secrets.env so we don't double-define them on the last-wins file.
awk '!/^(FIRSTPROOF_WORKFLOW|PROOFSTACK_SANDBOX_BACKEND)=/' "$TMP_ENV" > "$TMP_ENV_FILTERED"
mv "$TMP_ENV_FILTERED" "$TMP_ENV"

# Append a default only when the key is *not* already present in
# secrets.env. User-set values from the file win.
_append_if_missing() {
    local key="$1"
    local default_value="$2"
    if ! grep -qE "^${key}=" "$TMP_ENV"; then
        echo "${key}=${default_value}"
    fi
}

{
    echo
    # Always-force: variant-specific / required by the harness.
    echo "FIRSTPROOF_WORKFLOW=$WORKFLOW"
    echo "PROOFSTACK_SANDBOX_BACKEND=subprocess"
    # User-overridable defaults. The entrypoint forwards n_rounds +
    # page_limit + budget to every workflow subprocess, so without
    # these set the production defaults (10 / 12 / 1000) would override
    # the smoke preset's cheap values.
    _append_if_missing FIRSTPROOF_HEALTHCHECK "${FIRSTPROOF_HEALTHCHECK:-off}"
    _append_if_missing FIRSTPROOF_MAX_PARALLEL "${FIRSTPROOF_MAX_PARALLEL:-6}"
    _append_if_missing FIRSTPROOF_N_ROUNDS "${FIRSTPROOF_N_ROUNDS:-2}"
    _append_if_missing FIRSTPROOF_PAGE_LIMIT "${FIRSTPROOF_PAGE_LIMIT:-8}"
    _append_if_missing FIRSTPROOF_BUDGET_USD_PER_QUESTION \
        "${FIRSTPROOF_BUDGET_USD_PER_QUESTION:-$VARIANT_BUDGET_USD}"
} >> "$TMP_ENV"

echo "==> building $IMAGE_TAG (CAS layers take ~5-10 min on a fresh cache)"
cd "$REPO_ROOT"
docker build -t "$IMAGE_TAG" .

echo "==> running container with smoke workflow: $WORKFLOW"
docker run --rm \
    -v "$INPUT":/data/input/input.json:ro \
    -v "$OUTPUT":/data/output \
    --env-file "$TMP_ENV" \
    "$IMAGE_TAG"

echo
echo "==> smoke complete. Inspect:"
echo "    jq '.per_problem[] | {id, status, returncode, duration_seconds}' $OUTPUT/run_summary.json"
echo "    jq '.totals' $OUTPUT/run_summary.json"
