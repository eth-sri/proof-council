#!/usr/bin/env python3
"""Read-only checkpoint inventory. Never launches work or changes the run clock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathagents.provider_trace import usage_adjustments
from proofstack.agents.ac.ac_workflow import _problem_hash, _sum_logged_model_cost
from proofstack.agents.firstproof_batch3 import _Schedule, published_document_path
from proofstack.latex_contract import normalize_submission_latex


def inspect_run(root: Path, *, page_limit: int = 16, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    inputs_files = sorted((root / "agents").glob("FirstProofBatch3*Workflow-*/input.json"), key=lambda p: p.stat().st_mtime_ns)
    if not inputs_files:
        raise ValueError("No saved FirstProofBatch3Workflow input")
    inputs = json.loads(inputs_files[-1].read_text())
    schedule = _Schedule.model_validate_json((root / "batch3-schedule.json").read_text())
    if schedule.problem_hash != _problem_hash(inputs["problem"]):
        raise ValueError("Saved problem does not match the schedule hash")
    states = list((root / "ac_workspaces").glob("*/.ac/resume-state.json"))
    matches = [json.loads(path.read_text()) for path in states]
    matches = [state for state in matches if state.get("problem_hash") == schedule.problem_hash
               and state.get("problem_id") == inputs["problem_id"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching AC checkpoint")
    state = matches[0]
    caches = list((root / "resume_cache").glob("*.json"))
    for path in caches:
        json.loads(path.read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    output_file = root / "batch3-output.json"
    output = json.loads(output_file.read_text()) if output_file.exists() else {}
    approved = bool(output.get("submission_approved"))
    if approved:
        path = published_document_path(root, inputs["problem_id"], partial=False,
                                       digest=output.get("submission_sha256"), version=output.get("publication_version"))
        if path is None or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid approved document path")
        text = path.read_text(encoding="utf-8")
        if (not output.get("compiled") or not 0 < output.get("pages", 0) <= page_limit
                or hashlib.sha256(text.encode()).hexdigest() != output["submission_sha256"]
                or normalize_submission_latex(text) != text):
            raise ValueError("Approved export failed hash, page-limit, or document validation")
    terminal = state.get("terminal_outputs") or {}
    action = "preserve_approved_no_calls" if approved else (
        "candidate_cleanup" if terminal.get("early_stopped") and terminal.get("last_critic_accepted") else
        "resume_missing_review" if state.get("awaiting_review_kind") else "resume_research"
    )
    unresolved = {event.get("call_id") for event in events
                  if event.get("kind") == "model.call.cancelled" and event.get("payload", {}).get("usage_unavailable")}
    unresolved.update(event["call_id"] for event in usage_adjustments(root, events)
                      if event["payload"].get("usage_unavailable"))
    return {
        "run_id": root.name, "problem_id": inputs["problem_id"], "checkpoint_readable": True,
        "proposed_boundary": action, "last_round_index": state.get("last_round_run"),
        "next_round_index": state.get("next_round"), "parsed_cache_files": len(caches),
        "cumulative_known_cost_usd": _sum_logged_model_cost(root / "events.jsonl"),
        "original_budget_usd": schedule.initial_usd, "unresolved_cancelled_or_attempt_calls": len(unresolved),
        "research_deadline_unix_s": schedule.research_deadline_unix_s,
        "run_deadline_unix_s": schedule.run_deadline_unix_s,
        "clock_expired": now >= schedule.run_deadline_unix_s,
        "launch_authorized": False,
        "required_operator_checks": [
            "Reconcile legacy missing charges and unknown provider liability before setting remaining spend.",
            "Restore the original /data/output paths, or validate every relocated cache artifact reference.",
            "Do not restore stale process ownership, slot leases, or filesystem locks as live state.",
            "Keep the original clock unless an explicit rehearsal-only extension is approved.",
            "Validate changed code/config cache compatibility; JSON readability alone is not replay validation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_runs", type=Path)
    parser.add_argument("--page-limit", type=int, default=16,
                        help="Explicit rehearsal limit; does not change the competition preset")
    args = parser.parse_args()
    if args.page_limit < 1:
        parser.error("page limit must be positive")
    roots = sorted(path for path in args.workflow_runs.iterdir() if path.is_dir() and not path.name.startswith("."))
    reports = []
    for root in roots:
        try:
            reports.append(inspect_run(root, page_limit=args.page_limit))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reports.append({"run_id": root.name, "checkpoint_readable": False, "error": str(exc)})
    print(json.dumps({"read_only": True, "runs": reports}, indent=2))
    return 0 if reports and all(row["checkpoint_readable"] for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
