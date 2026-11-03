#!/usr/bin/env python3
"""Record independently verified API settlement without resuming paid work."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proofstack.provider_accounting import reconcile_stopped_provider_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("call_dir", type=Path, help="Exact invocation directory under RUN_DIR/agents")
    parser.add_argument("--confirm-workers-stopped", action="store_true")
    parser.add_argument("--confirm-provider-work-terminal", action="store_true",
                        help="Independently retrieve/cancel any saved response IDs first")
    parser.add_argument("--unreported-allowance-usd", type=float, required=True,
                        help="Additional conservative charge for unreported work; never refunds recorded usage")
    args = parser.parse_args()
    saved = reconcile_stopped_provider_call(args.run_dir, args.call_dir,
        workers_stopped=args.confirm_workers_stopped,
        provider_work_terminal=args.confirm_provider_work_terminal,
        unreported_allowance_usd=args.unreported_allowance_usd)
    print(json.dumps({"debit_key": saved["debit_key"], "limits_changed": False, "model_calls_started": 0}))


if __name__ == "__main__":
    main()
