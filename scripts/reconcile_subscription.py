#!/usr/bin/env python3
"""Reconcile a stopped subscription invocation without launching any model work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proofstack.subscription import reconcile_stopped_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("receipt_id")
    parser.add_argument("--confirm-workers-stopped", action="store_true",
                        help="Confirm all run controllers, workers and descendants have stopped")
    parser.add_argument("--unreported-allowance-usd", type=float, required=True,
                        help="Conservative additional nominal charge for unreported work; zero only if none occurred")
    args = parser.parse_args()
    result = reconcile_stopped_call(args.run_dir, args.receipt_id,
                                   workers_stopped=args.confirm_workers_stopped,
                                   unreported_allowance_usd=args.unreported_allowance_usd)
    print(json.dumps({"receipt_id": args.receipt_id, "status": result["status"],
                      "limits_changed": False, "model_calls_started": 0}))


if __name__ == "__main__":
    main()
