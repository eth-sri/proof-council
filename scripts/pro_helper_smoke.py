"""One live MultiAuthor turn; isolated from all research runs and checkpoints."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(os.environ.get("SMOKE_OUTPUT_DIR", "/data/output"))
# RequestLogger is a module singleton: set its destination before importing agents.
os.environ["MATHAGENTS_REQUEST_LOG_DIR"] = str(ROOT / "request-logs")

from proofstack.agents.ac.author import Author
from proofstack.agents.ac.delegation_context import ContextView
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.agents.ac.helper_smoke_check import assess_recovery
from proofstack.agents.ac.helper_sandbox import HelperSandbox
from proofstack.atomic import write_text_atomic
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.registry import load_preset
from mathagents.config_loader import load_solver_config

COMMIT = os.environ.get("PROOFCOUNCIL_SOURCE_COMMIT", "unrecorded")
LOCK = threading.Lock()
REPORT = {"ok": False, "source_commit": COMMIT, "started_at": time.time(),
          "scope": "one Author turn, one wave, four Pro helpers; no critic/cleanup/Compute/council",
          "errors": [], "checks": {}}
CTX = None

HELPER_PROTOCOL = """Infrastructure test, not open-ended research. Work only on your assigned n.
Read manifest.json and round/critic.md with read_context; retain the critic_marker.
Using code_interpreter, create a short Python script that directly evaluates
sum(comb(n,k)**2 for k in range(n+1)), independently computes comb(2*n,n), and asserts
equality. Execute it. Generate one random UUID hex nonce in the sandbox and save it
to /mnt/data/checkpoints/smoke-marker.txt; save your script at
/mnt/data/checkpoints/smoke-verify.py. Create the directory first.
Write /mnt/data/checkpoints/checkpoint.json containing integer n, integer sum,
integer closed_form, string nonce, string critic_marker, and string script (its
full runnable source). Publish this actual file using publish_sandbox_artifact.
Confirm status=published and sandbox_checkpoint.status=passed. Never transcribe a
computed file with publish_artifact. Do not regenerate the nonce later.
Read the returned published path with read_context. In the next code_interpreter
call, BEFORE restoring anything, check whether the marker file still exists and
matches the original nonce. Restore any missing script/marker from that published
checkpoint: copy the tool reply's read-only attachment_path files to their listed
sandbox_path locations. Execute the script again and verify the restored nonce
matches. Keep the recovered marker in /mnt/data/checkpoints/smoke-marker.txt.
Write /mnt/data/checkpoints/result.json with n, sum, closed_form, nonce, critic_marker, restored_nonce,
marker_survived (boolean), script_sha256 (hex SHA-256 of checkpoint script), and
recomputed (true only if you really re-executed it successfully). Publish it using
publish_sandbox_artifact. Check publication and checkpoint transfer
success. Give a short proof or verification and list both artifact paths in your
final report, but do NOT repeat nonce or critic_marker in your textual report:
the lead must read the published files. No browsing, installations or large files.
Do not invent missing results; report protocol failures honestly and finish.
"""

PROBLEM = """Prove the binomial identity sum_{k=0}^n binom(n,k)^2 = binom(2n,n)
for nonnegative integers n and verify it computationally at n=12,13,14,15.

This is a bounded infrastructure smoke test. Make exactly ONE delegate call
containing FOUR fresh helpers in a single parallel wave, all include_workspace=true:
explorer for n=12, prover for n=13, checker for n=14, prover for n=15. Pass the helper
protocol below verbatim in their shared briefing; assign only the distinct n and
role-specific short proof/check in each task. Do not open a second wave or try to
solve another problem. Before delegation write a small comment containing
LEAD_PREWAVE_MARKER into canonical research_notes.tex, to exercise lead carry-over.
Delegation is asynchronous: use helper_status and wait_helpers until all four
helpers have finished before ending your turn. After completion read each helper's published result.json AND checkpoint.json using
read_context. Do not rely only on the summary. Write one short compiling answer.tex
with the general identity/proof and four verified numeric cases. In research_notes.tex
include each helper's original nonce exactly as read from its artifacts, with n,
critic_marker, and whether its marker survived the local call. Retain
LEAD_PREWAVE_MARKER, restoring carried files if necessary. Write references.bib
(empty is fine). Keep final files short. No council or local Compute requests,
no web search, no further research, no follow-up critic, no rewrite. Finish promptly.

Required helper protocol:
""" + HELPER_PROTOCOL


def save():
    if CTX is not None:
        REPORT["budget_counters"] = asdict(CTX.budgets.root("run").counters)
    write_text_atomic(ROOT / "smoke-report.json", json.dumps(REPORT, indent=2, default=str))


def record(kind, **payload):
    with LOCK:
        with (ROOT / "tool-audit.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": time.time(), "kind": kind, **payload}) + "\n")


def instrument_context():
    # Observation only: preserve original tool behavior and exceptions.
    original_read, original_publish = ContextView.read, ContextView._publish
    original_checkpoint = HelperSandbox.checkpoint

    @wraps(original_read)
    def read(self, path="manifest.json", offset=0, max_chars=12000, revision=None):
        result = original_read(self, path, offset, max_chars, revision)
        data = json.loads(result)
        record("read_context", actor=self.publisher or "lead", path=path,
               offset=offset, next_offset=data.get("next_offset"), revision=data.get("revision"),
               ok="error" not in data)
        return result

    @wraps(original_publish)
    def publish(self, name, content, *, provenance=None):
        result = original_publish(self, name, content, provenance=provenance)
        data = json.loads(result)
        record("publish_artifact", actor=self.publisher, name=name,
               path=data.get("path"), sha256=data.get("sha256"),
               ok=data.get("status") == "published")
        return result

    @wraps(original_checkpoint)
    def checkpoint(self, messages, call_deadline_monotonic_s=None):
        result = original_checkpoint(self, messages, call_deadline_monotonic_s)
        record("sandbox_checkpoint", actor=self.view.publisher, checkpoint=result,
               ok=result.get("status") != "failed")
        return result

    ContextView.read, ContextView._publish = read, publish
    HelperSandbox.checkpoint = checkpoint


def component_config():
    config = deepcopy(load_preset("firstproof_batch3_multiauthor").component_configs["Author"])
    config["delegation"].update(enabled=True, asynchronous=True, max_threads=4, max_tasks_per_turn=4)
    assert config["delegation"]["helper_timeout_s"] is None
    assert config["delegation"]["synthesis_reserve_s"] == 1500
    for ref in (config["model"], config["delegation"]["subagent_model"]):
        model = load_solver_config(ref)
        assert model["reasoning"]["mode"] == "pro"
        assert model["model"] == "gpt-6-astra--max"
    return config


def offline_check():
    from mathagents.request_logger import request_logger
    assert Path(request_logger.log_dir) == ROOT / "request-logs"
    config = component_config()
    ctx = RunContext.create(run_id="offline", root_workdir=ROOT / "offline", flat=True,
                            component_configs={"Author": config})
    seat = SubAuthorSeat(ctx)
    messages = seat.render_messages(seat.Inputs(role="prover", task="check", problem="P"))
    assert "designated checkpoints are automatically" in messages[-1]["content"]
    for n in (12, 13, 14, 15):
        assert sum(math.comb(n, k)**2 for k in range(n+1)) == math.comb(2*n, n)
    return {"ok": True, "paid_calls": 0, "source_commit": COMMIT,
            "helper_model": config["delegation"]["subagent_model"],
            "helper_timeout_s": None, "max_tasks_per_turn": 4, "asynchronous": True, "helpers": 4}


def verify(author, out, marker):
    session = getattr(author, "_async_session", None)
    recs = list(session.records.values()) if session is not None else author._delegation_log
    assert len(recs) == 4, "expected exactly four helpers"
    if session is not None:
        assert all(r["status"] == "completed" for r in recs), "asynchronous helpers did not all finish"
    else:
        assert author._waves_done == 1, "expected one legacy wave"
    assert all(not r["error"] for r in recs), "one or more helpers failed"
    audit = [json.loads(line) for line in (ROOT / "tool-audit.jsonl").read_text().splitlines()]
    def read_by(actor, path):
        return any(e["kind"] == "read_context" and e["actor"] == actor
                   and e["path"] == path and e["ok"] and e.get("offset") == 0
                   and e.get("next_offset") is None for e in audit)
    cases = []
    for rec in recs:
        publisher = "helpers/" + ("" if session is not None else "wave1-") + rec["agent_id"]
        context = session.stores[rec["agent_id"]] if session is not None else author._context
        context_files = ({p: session.artifact_text(context, p) for p in session.artifacts(rec["agent_id"])}
                         if session is not None else context.files)
        cp_path, result_path = publisher + "/checkpoint.json", publisher + "/result.json"
        checkpoint = json.loads(context_files[cp_path])
        result = json.loads(context_files[result_path])
        n = checkpoint["n"]
        assert type(n) is int and n in (12, 13, 14, 15)
        expected = sum(math.comb(n, k)**2 for k in range(n+1))
        assert checkpoint["sum"] == checkpoint["closed_form"] == expected == math.comb(2*n, n)
        assert result["n"] == n and result["sum"] == result["closed_form"] == expected
        nonce = checkpoint["nonce"]
        assert re.fullmatch("[a-f0-9]{32}", nonce)
        assert result["nonce"] == result["restored_nonce"] == nonce
        assert result["critic_marker"] == checkpoint["critic_marker"] == marker
        assert result["recomputed"] is True and type(result["marker_survived"]) is bool
        compile(checkpoint["script"], "helper-script", "exec")  # Never execute helper code on the host.
        assert result["script_sha256"] == hashlib.sha256(checkpoint["script"].encode()).hexdigest()
        report = context_files[rec["report_path"]] if session is not None else rec["report"]
        assert nonce not in report, "artifact-only marker leaked through summary"
        assert nonce in out.research_notes_tex, "lead did not retain artifact-only marker"
        assert read_by(publisher, "round/critic.md") and read_by(publisher, cp_path)
        assert read_by("lead", cp_path) and read_by("lead", result_path)
        recovery = assess_recovery(context_files, context.metadata,
                                   publisher, cp_path, result_path, capture_receipts=[
                                       {"publisher": e["actor"], "checkpoint": e["checkpoint"]}
                                       for e in audit if e["kind"] == "sandbox_checkpoint"
                                   ])
        if rec.get("checkpoint_errors"):
            recovery = {"status": "failed", "errors": recovery["errors"] + rec["checkpoint_errors"]}
        cases.append({"agent_id": rec["agent_id"], "n": n, "expected": expected,
                      "marker_survived_reported": result["marker_survived"], "usd": rec.get("cost_usd", rec.get("usd")),
                      "recovery": recovery})
    assert sorted(c["n"] for c in cases) == [12, 13, 14, 15]
    assert out.via == "container_files" and out.artifact_status != "failed"
    assert "LEAD_PREWAVE_MARKER" in out.research_notes_tex
    assert not out.compute_instructions and not out.council_question
    REPORT["checks"]["helpers_and_handoff"] = cases
    REPORT["checks"]["recovery"] = {
        "status": "failed" if any(c["recovery"]["status"] == "failed" for c in cases)
        else "unverified" if any(c["recovery"]["status"] == "unverified" for c in cases) else "passed",
    }
    REPORT["checks"]["lead"] = {"via": out.via, "artifact_status": out.artifact_status,
                                "files_changed": out.files_changed, "container_id": out.container_id}
    REPORT["note"] = "Recovery requires independent downloaded original/recovered markers, not matching model-authored fields. This validates file continuity, not arbitrary mathematical claims."


async def main(args):
    global CTX
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.offline:
        print(json.dumps(offline_check()), flush=True)
        return 0
    if args.budget_usd not in (25, 50):
        raise SystemExit("Explicit --budget-usd 25 or 50 is required")
    if (ROOT / "smoke-report.json").exists():
        raise SystemExit("Refusing to restart a smoke test or incur duplicate spend")
    REPORT["model_budget_usd"] = args.budget_usd
    REPORT["wallclock_limit_s"] = 4800
    REPORT["budget_note"] = "Cooperative metered cap, not a guaranteed provider invoice ceiling; concurrent or failed calls may add uncertainty."
    CTX = RunContext.create(run_id="pro-helper-smoke", root_workdir=ROOT / "workflow", flat=True,
        run_budget=BudgetSpec(max_usd=args.budget_usd, max_wallclock_s=4800),
        component_configs={"Author": component_config()})
    save()
    instrument_context()
    marker = "CONTEXT-" + uuid.uuid4().hex
    try:
        author = MultiAuthor(CTX, name="Author")
        print(json.dumps({"phase": "started", "commit": COMMIT, "budget_usd": args.budget_usd,
                          "helpers": 4, "helper_model": "Astra Pro/max"}), flush=True)
        async with asyncio.timeout(4800):
            out = await author(problem=PROBLEM, round=1, n_rounds=2, page_limit=16,
                budget_max_usd=args.budget_usd,
                answer_tex="\\documentclass[12pt]{article}\n\\begin{document}Pending smoke.\\end{document}\n",
                research_notes_tex="\\documentclass[12pt]{article}\n\\begin{document}Pending smoke.\\end{document}\n",
                references_bib="", prev_critique="Smoke context check: critic_marker=" + marker,
                workflow_feedback="One turn only. Exactly four helpers in one wave. Finish after synthesis.")
        write_text_atomic(ROOT / "author-output.json", out.model_dump_json(indent=2))
        verify(author, out, marker)
        tex = ROOT / "manuscript"
        tex.mkdir()
        for name in ("answer.tex", "research_notes.tex", "references.bib"):
            write_text_atomic(tex / name, getattr(out, name.replace(".", "_")))
        compiled = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "answer.tex"],
                                  cwd=tex, text=True, capture_output=True, timeout=60)
        write_text_atomic(ROOT / "compile.log", compiled.stdout + compiled.stderr)
        assert compiled.returncode == 0 and (tex / "answer.pdf").is_file(), "lead manuscript does not compile"
        REPORT["checks"]["compile"] = True
        assert CTX.budgets.root("run").counters.usd <= args.budget_usd, "metered budget exceeded"
        REPORT["ok"] = REPORT["checks"]["recovery"]["status"] == "passed"
        if not REPORT["ok"]:
            REPORT["errors"].append("Sandbox recovery failed or remains unverified; see per-helper evidence checks.")
    except Exception as exc:
        REPORT["errors"].append(f"{type(exc).__name__}: {exc}")
        print(json.dumps({"phase": "failed", "error": REPORT["errors"][-1]}), flush=True)
    finally:
        REPORT["finished_at"] = time.time()
        save()
        sys.path.insert(0, "/app/scripts")
        from firstproof_entrypoint import _finalize_output_permissions
        for warning in _finalize_output_permissions(ROOT):
            print(warning, file=sys.stderr, flush=True)
        print(json.dumps(REPORT, default=str), flush=True)
    return 0 if REPORT["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--budget-usd", type=int)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
