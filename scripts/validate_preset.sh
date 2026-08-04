#!/usr/bin/env bash
# Validate a workflow preset YAML; print ok / errors / warnings.
# Usage: scripts/validate_preset.sh PRESET_YAML
#   scripts/validate_preset.sh configs/workflows/claude_subscription_min.yaml
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
PRESET="${1:?usage: validate_preset.sh PRESET_YAML}"
cd "$(dirname "$0")/.."
PYTHONPATH=src uv run python -c '
import sys
from pathlib import Path
from app.dev_data import validate_preset_yaml
r = validate_preset_yaml(Path(sys.argv[1]).read_text())
print("ok:", r["ok"])
print("errors:", r.get("errors"))
print("warnings:", r.get("warnings"))
' "$PRESET"
