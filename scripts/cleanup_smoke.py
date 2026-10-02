"""Bounded live cleanup integration test; never launches the research workflow."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from proofstack.agents.cleanup_session import CleanupSession, cleanup_accounting_unresolved
from proofstack.agents.writeup_loop import _compile_raw
from proofstack.atomic import write_text_atomic
from proofstack.budget import BudgetSpec
from proofstack.cleanup_runtime import check_cleanup_runtime
from proofstack.context import RunContext


TEX = r"""\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{Sum of the first odd integers}
For each integer $n\geq 1$, $\sum_{k=1}^n(2k-1)=n^2$.
This is true for $n=1$. If it holds for $n$, adding $2n+1$ gives
$n^2+2n+1=(n+1)^2$, completing the induction.
\end{document}
"""
PROTOCOL = """This is a small infrastructure test, not a research task. Keep the
manuscript under two pages. Make a useful, minimal editorial improvement, preserving
the exact result and induction proof. Use exactly one correctness-reviewer and one
exposition-reviewer native subagent call, and exactly one managed Codex review. All
three must read/check the current manuscript and baseline and return short reports.
Wait for their results. Include the findings in feedback.md, compile the final
document, and write completion.json. Do not request extra reviews or web research.
"""


def evidence(root, *, paths=None):
    calls, results = {}, {}
    native_completions = set()
    exposed_tools = []
    for path in root.rglob("cli_stdout.log") if paths is None else paths:
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                exposed_tools.append(event.get("tools", []))
            if (event.get("type") == "system" and event.get("subtype") == "task_notification"
                    and event.get("status") == "completed" and str(event.get("summary") or "").strip()):
                native_completions.add(event.get("tool_use_id"))
            if not isinstance(event.get("message"), dict):
                continue
            content = event["message"].get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    calls[block["id"]] = block
                elif block.get("type") == "tool_result":
                    results[block["tool_use_id"]] = block
    completed = [call for key, call in calls.items()
                 if key in results and not results[key].get("is_error", False)]
    return {
        "exposed_tools": exposed_tools,
        "native_tool_calls": sum(call.get("name") in {"Task", "Agent"} for call in calls.values()),
        "completed_native_reviewers": sorted({call.get("input", {}).get("subagent_type", "unknown")
            for call in completed if call.get("name") in {"Task", "Agent"}
            and call["id"] in native_completions}),
        "completed_mcp_tools": sorted({call["name"] for call in completed
                                       if call.get("name", "").startswith("mcp__cleanup__")}),
        "failed_tool_results": sum(bool(result.get("is_error")) for result in results.values()),
        "codex_reports": len(list((root / "cleanup_sessions/smoke/workspace/reviews").glob("*.md"))),
    }


async def main(args):
    runtime = await asyncio.to_thread(check_cleanup_runtime,
                                     claude_version="2.1.251", codex_version="0.154.0")
    if not args.live:
        print(json.dumps({"runtime": runtime, "paid_calls": 0}, indent=2))
        return 0
    if args.budget_usd is None or not math.isfinite(args.budget_usd) or not 0 < args.budget_usd <= 25:
        raise SystemExit("Live smoke requires an authorized --budget-usd in (0, 25]")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive marker prevents an accidental duplicate paid run, even after a crash.
    with (root / "launch.json").open("x") as marker:
        json.dump({"started_at": time.time(), "budget_usd": args.budget_usd}, marker)
    report = {"ok": False, "runtime": runtime, "source_commit": os.environ.get("PROOFCOUNCIL_SOURCE_COMMIT"),
              "source_archive_sha256": os.environ.get("PROOFCOUNCIL_SOURCE_SHA256"),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "budget_usd": args.budget_usd, "wallclock_limit_s": 1800, "started_at": time.time(),
              "scope": "one edit, two native reviewers, one Codex review, one finishing-only resumed repair",
              "budget_note": "Recorded cap, not an invoice guarantee", "errors": []}
    ctx = RunContext.create(run_id="cleanup-smoke", root_workdir=root / "workflow", flat=True,
        run_budget=BudgetSpec(max_usd=args.budget_usd, max_wallclock_s=1800),
        component_configs={"CleanupSession": {"cleanup": {"max_invocations": 1}}})

    def save():
        report["budget_counters"] = asdict(ctx.budgets.root().counters)
        write_text_atomic(root / "smoke-report.json", json.dumps(report, indent=2, default=str))

    save()
    current = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, current.cancel)
    try:
        agent = CleanupSession(ctx)
        async with asyncio.timeout(1800):
            common = dict(problem="Prove that the sum of the first n odd positive integers is n squared.",
                          baseline=TEX, page_limit=2, session_key="smoke")
            first = await agent(document=TEX, constraints=PROTOCOL, **common)
            write_text_atomic(root / "initial-edit.json", first.model_dump_json(indent=2))
            if first.status != "ready":
                raise RuntimeError("initial editorial pass did not report ready")
            report["first_pass_evidence"] = evidence(ctx.root_workdir)
            save()
            previous_logs = set(ctx.root_workdir.rglob("cli_stdout.log"))
            second = await agent(document=first.answer_tex,
                finishing_only=True,
                findings="Targeted repair: explicitly mention the n=0 empty-sum convention as well.",
                constraints="Infrastructure test continuation: make only this small repair, preserve the proof. "
                            "Do not call reviewers again. Compile, update feedback.md and completion.json.", **common)
            write_text_atomic(root / "resumed-edit.json", second.model_dump_json(indent=2))
            if second.status != "ready" or first.session_id != second.session_id:
                raise RuntimeError("targeted repair did not complete in the same session")
            report["finishing_evidence"] = evidence(ctx.root_workdir, paths=(
                set(ctx.root_workdir.rglob("cli_stdout.log")) - previous_logs))
            finishing = report["finishing_evidence"]
            if (not finishing["exposed_tools"] or finishing["native_tool_calls"]
                    or any({"Task", "Agent"} & set(names) for names in finishing["exposed_tools"])
                    or "mcp__cleanup__compile" not in finishing["completed_mcp_tools"]):
                raise RuntimeError('finishing-only --tools "" did not retain MCP compilation without native delegation')
            report["session_id"] = second.session_id
            ok, pages, detail = await asyncio.to_thread(_compile_raw, second.answer_tex, None, secure=True)
            report["final_compile"] = {"compiled": ok, "pages": pages, "diagnostics": detail}
            if not ok or not 0 < pages <= 2:
                raise RuntimeError("final manuscript fails compilation/page checks")
            observed = report["first_pass_evidence"]
            if not {"correctness-reviewer", "exposition-reviewer"} <= set(observed["completed_native_reviewers"]):
                raise RuntimeError("missing completed native reviewer evidence")
            if not {"mcp__cleanup__compile", "mcp__cleanup__read", "mcp__cleanup__review"} <= set(observed["completed_mcp_tools"]):
                raise RuntimeError("missing completed MCP tool evidence")
            if observed["codex_reports"] != 1:
                raise RuntimeError("expected exactly one retained Codex report")
            if observed["failed_tool_results"]:
                raise RuntimeError("one or more tool calls failed; inspect transcripts")
            ctx.budgets.root().check()
            if cleanup_accounting_unresolved(ctx.root_workdir):
                raise RuntimeError("cleanup accounting remains unresolved")
            report["ok"] = True
    except (Exception, asyncio.CancelledError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        report["finished_at"] = time.time()
        report["evidence"] = evidence(ctx.root_workdir)
        report["accounting_unresolved"] = cleanup_accounting_unresolved(ctx.root_workdir)
        save()
        print(json.dumps(report, indent=2, default=str), flush=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize paid calls")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--output", type=Path, default=Path("/data/output"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
