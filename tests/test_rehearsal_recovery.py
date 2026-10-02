"""Offline regressions for interrupted-batch recovery failure modes."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mathagents.api_client import APIClient, _ProviderWallclockTimeout, _is_terminal_api_error
from mathagents.provider_trace import ProviderTrace, active_trace, latest_attempts, usage_adjustments
from proofstack.context import RunContext, ResumeCache
from proofstack.agents.ac.compute import Compute, _require_codex_execution_host
from proofstack.agents.ac.council import CouncilMember
from proofstack.agents.ac.author import Author
from proofstack.sandbox.subprocess import _make_preexec


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setattr("mathagents.api_client.request_logger.log_request", lambda **kw: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_response", lambda **kw: None)
    return APIClient(model="claude-test", api="anthropic", max_tokens=128,
                     timeout=2, max_wallclock_per_call_s=2, max_retries=1,
                     max_retries_inner=0, sleep_after_request=0, sleep_on_error=0,
                     read_cost=10, write_cost=50, cache_write_cost=12.5)


def _message(content, *, stop_reason="end_turn", usage=None):
    usage = usage or {"input_tokens": 4, "output_tokens": 12}
    raw = {"id": "msg_test", "content": content, "usage": usage, "stop_reason": stop_reason}
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage=SimpleNamespace(**usage), model_dump=lambda: raw)


def test_thinking_tokens_are_not_lost(api):
    assert api._extract_reasoning_tokens({"output_tokens_details": {"thinking_tokens": 126174}}) == 126174


def test_invalid_request_is_terminal():
    assert _is_terminal_api_error(ValueError("Error code: 400 missing code_execution_tool_result"))
    assert not _is_terminal_api_error(ValueError("503 overloaded"))


def test_claude_unpaired_server_tool_gets_tools_disabled_salvage(api, monkeypatch):
    api.stream_anthropic_messages = False
    replies = iter([
        _message([{"type": "text", "text": "partial"},
                  {"type": "server_tool_use", "name": "code_execution", "id": "unfinished", "input": {}}], stop_reason="max_tokens"),
        _message([{"type": "text", "text": "final qualified answer"}]),
    ])
    payloads = []

    def create(**payload):
        payloads.append(json.loads(json.dumps(payload, default=str)))
        return next(replies)

    monkeypatch.setattr("mathagents.api_client.anthropic.Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    result = api._run_query_with_retry(0, [{"role": "user", "content": "question"}])
    assert len(payloads) == 2
    assert "tools" not in payloads[1]
    assert "unfinished" not in json.dumps(payloads[1])
    assert result.output_tokens == 24
    assert any("final qualified answer" in message.get("content", "") for message in result.conversation)


def test_known_cost_survives_later_failure(api, tmp_path, monkeypatch):
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl", call_id="call")
    token = active_trace.set(trace)

    def fail(idx, query, **kw):
        api._log_provider_request(ts="first", batch_idx=idx, request={"model": api.model})
        api._log_provider_response(ts="first", batch_idx=idx, response={"usage": {
            "input_tokens": 4, "cache_creation_input_tokens": 53585, "output_tokens": 128000,
            "output_tokens_details": {"thinking_tokens": 126174},
        }})
        api._log_provider_request(ts="second", batch_idx=idx, request={"model": api.model})
        raise ValueError("Error code: 400 invalid_request_error")

    monkeypatch.setattr(api, "_run_query", fail)
    try:
        with pytest.raises(ValueError) as error:
            api._run_query_with_retry(0, [{"role": "user", "content": "question"}])
    finally:
        active_trace.reset(token)
    assert error.value.partial_usage["cost"] == pytest.approx(7.0698525)
    assert error.value.partial_usage["reasoning_tokens"] == 126174
    assert error.value.partial_usage["usage_unavailable"]
    assert len(latest_attempts(trace.path)) == 2


def test_receipts_are_idempotent_and_reconcile_late_usage(api, tmp_path):
    path = tmp_path / "agents" / "Council-c1" / "provider-attempts.jsonl"
    trace = ProviderTrace(path, call_id="call")
    payload = {"usage": {"input_tokens": 100, "output_tokens": 200}}
    trace.response(api, "request", 0, payload)
    trace.response(api, "request", 0, payload)
    cost = trace.totals()["cost"]
    events = [{"kind": "model.call", "call_id": "call", "payload": {"cost_usd": cost / 2}}]
    adjustments = usage_adjustments(tmp_path, events)
    assert len(adjustments) == 1
    assert adjustments[0]["payload"]["cost_usd"] == pytest.approx(cost / 2)
    assert len(latest_attempts(path)) == 1


def test_blocked_stream_deadline_records_partial_usage_and_closes(api, tmp_path):
    api.max_wallclock_per_call_s = 0.1
    api.stream_anthropic_messages = True
    closed = threading.Event()

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def __iter__(self):
            yield SimpleNamespace(type="message_start", message=SimpleNamespace(
                id="msg_blocked", usage=SimpleNamespace(input_tokens=20, output_tokens=1)))
            closed.wait(2)

        def close(self):
            closed.set()

        def get_final_message(self):
            raise RuntimeError("stream interrupted")

    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)
    trace.request(api, "request", 0, {"model": api.model})
    start = time.monotonic()
    try:
        with pytest.raises(_ProviderWallclockTimeout):
            api._create_anthropic_message(SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream())), {}, inner_start=time.time())
    finally:
        active_trace.reset(token)
    assert time.monotonic() - start < 1
    assert closed.wait(1)
    assert trace.totals()["input_tokens"] == 20
    assert trace.totals()["usage_unavailable"]


def test_gemini_exhaustion_uses_one_tools_disabled_recovery(api, monkeypatch):
    api.api = "google"
    api.base_url = "https://offline.invalid"
    api.tool_descriptions = [{"type": "code_interpreter"}]
    payloads = []
    replies = iter([
        {"candidates": [{"finishReason": "TOO_MANY_TOOL_CALLS", "content": {"role": "model", "parts": [
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "checked 7 examples"}},
        ]}}], "usageMetadata": {"promptTokenCount": 10, "toolUsePromptTokenCount": 100, "thoughtsTokenCount": 20}},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "Evidence only; theorem remains open."}]}}],
         "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5}},
    ])

    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        response = next(replies)
        return SimpleNamespace(status_code=200, json=lambda: response)

    monkeypatch.setattr("mathagents.api_client.requests.post", post)
    result = api._run_query_with_retry(0, [{"role": "user", "content": "question"}])
    assert len(payloads) == 2
    assert "tools" in payloads[0] and "tools" not in payloads[1]
    assert "checked 7 examples" in json.dumps(payloads[1])
    assert result.input_tokens == 125 and result.output_tokens == 25
    assert result.conversation[-1]["content"].startswith("Evidence only")


def test_virtual_limit_opt_out_keeps_other_limits(monkeypatch):
    import resource
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))
    _make_preexec(8, 4, 30, limit_address_space=False)()
    assert resource.RLIMIT_AS not in [kind for kind, limits in calls]
    assert (resource.RLIMIT_CPU, (120, 120)) in calls
    assert (resource.RLIMIT_CORE, (0, 0)) in calls


@pytest.mark.parametrize("success", [False, True])
def test_native_preflight_fails_before_paid_work(success):
    sandbox = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(
        returncode=0 if success else -5,
        stdout="code-mode preflight passed" if success else "",
        stderr="" if success else "Failed to reserve the virtual address space for the V8 sandbox")))
    if success:
        asyncio.run(_require_codex_execution_host(sandbox))
    else:
        with pytest.raises(RuntimeError, match="before any model call"):
            asyncio.run(_require_codex_execution_host(sandbox))
    assert sandbox.run_command.call_args.args[0][:2] == ["python3", "-c"]


@pytest.mark.parametrize("hoisted", [False, True])
def test_native_host_discovery_follows_node_architecture(tmp_path, monkeypatch, hoisted):
    from proofstack.sandbox import codex_preflight as probe
    package = tmp_path / "node_modules/@openai/codex"
    cli = package / "bin/codex.js"
    cli.parent.mkdir(parents=True)
    cli.touch()
    platform_package = package.parent if hoisted else package / "node_modules/@openai"
    host = platform_package / "codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex-code-mode-host"
    host.parent.mkdir(parents=True)
    host.touch(mode=0o700)
    monkeypatch.setattr(probe.shutil, "which", lambda name: str(cli))
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: b'{"platform":"darwin","arch":"arm64"}')
    assert probe.find_host() == str(host)


def test_failed_worker_and_empty_council_cache_not_replayed(tmp_path):
    ctx = RunContext.create(run_id="test", root_workdir=tmp_path, flat=True)
    compute = Compute(ctx)
    assert not compute.cache_output_is_reusable(Compute.Outputs(
        status="partial", response_md="code-mode host closed its stdout", workspace=tmp_path))
    assert compute.cache_output_is_reusable(Compute.Outputs(
        status="partial", response_md="Verified a finite example; proof still open.", workspace=tmp_path))
    seat = CouncilMember(ctx)
    assert not seat.cache_output_is_reusable(seat.Outputs(text="  "))
    assert seat.cache_output_is_reusable(seat.Outputs(text="A substantive reply"))


def test_failed_edit_cannot_mark_unchanged_manuscript_ready(tmp_path):
    ctx = RunContext.create(run_id="test", root_workdir=tmp_path, flat=True)
    author = Author(ctx)
    inp = Author.Inputs(problem="P", problem_id="p", round=1, n_rounds=5, answer_tex="old manuscript")
    out = author._build_outputs_from_container(inp, "<ready>true</ready>", {}, "container", execution_failed=True)
    assert out.answer_tex == inp.answer_tex
    assert out.artifact_status == "failed" and not out.ready
    assert author._build_outputs_from_container(inp, "No edits needed", {}, "container").artifact_status == "unchanged"


def test_interrupted_cache_replace_preserves_previous_checkpoint(tmp_path, monkeypatch):
    cache = ResumeCache(tmp_path)
    cache.put("key", {"answer": "previous"})
    monkeypatch.setattr("proofstack.atomic.os.replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        cache.put("key", {"answer": "new"})
    assert cache.get("key") == {"answer": "previous"}


def test_abandoned_operation_cleans_up_late_response_in_trace_context(api, tmp_path):
    release = threading.Event()
    cleaned = threading.Event()
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)

    def operation():
        release.wait(2)
        return "response-late"

    def cleanup(response):
        assert active_trace.get() is trace
        trace.update(("late", 0), response_id=response, cancellation_acknowledged=True)
        cleaned.set()

    try:
        with pytest.raises(_ProviderWallclockTimeout):
            api._bounded_provider_operation(operation, timeout=0.05, on_abandoned_result=cleanup)
    finally:
        active_trace.reset(token)
        release.set()
    assert cleaned.wait(1)
    assert latest_attempts(trace.path)[0]["cancellation_acknowledged"]


def test_background_cancel_persists_ack_and_usage(api, tmp_path):
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)
    client = SimpleNamespace(responses=SimpleNamespace(cancel=lambda response_id, **kw: SimpleNamespace(model_dump=lambda: {
        "id": response_id, "status": "cancelled", "usage": {"input_tokens": 20, "output_tokens": 3},
    })))
    try:
        api._cancel_background_response(client, "response-id", "request", 0)
    finally:
        active_trace.reset(token)
    receipt = latest_attempts(trace.path)[0]
    assert receipt["cancellation_requested"] and receipt["cancellation_acknowledged"]
    assert receipt["response_id"] == "response-id" and receipt["input_tokens"] == 20


def test_gemini_failed_wrapup_does_not_restart_tool_loop(api, monkeypatch):
    api.api = "google"
    api.base_url = "https://offline.invalid"
    api.max_retries = 3
    payloads = []

    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        if len(payloads) > 1:
            raise RuntimeError("503 on wrap-up")
        return SimpleNamespace(status_code=200, json=lambda: {
            "candidates": [{"finishReason": "TOO_MANY_TOOL_CALLS"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5},
        })

    monkeypatch.setattr("mathagents.api_client.requests.post", post)
    with pytest.raises(RuntimeError, match="tools-disabled recovery failed") as error:
        api._run_query_with_retry(0, [{"role": "user", "content": "question"}])
    assert len(payloads) == 2
    assert error.value.partial_usage["input_tokens"] == 100


def test_failed_query_charges_known_usage_once(api, tmp_path):
    from proofstack.agents.ac.ac_workflow import _sum_logged_model_cost
    ctx = RunContext.create(run_id="run", root_workdir=tmp_path, flat=True)
    agent = CouncilMember(ctx)

    def query(client, messages):
        client._log_provider_request(ts="request", batch_idx=0, request={"model": client.model})
        client._log_provider_response(ts="request", batch_idx=0, response={"usage": {"input_tokens": 100, "output_tokens": 50}})
        raise RuntimeError("provider transport failed afterwards")

    with pytest.raises(RuntimeError):
        asyncio.run(agent._query(api, [], query, call_id="one-call"))
    expected = api._get_cost(100, 50)
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(expected)
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "model.call"]) == 1
    assert usage_adjustments(ctx.root_workdir, events) == []


def test_standalone_compute_guard_enforces_rss_without_address_limit(monkeypatch):
    from proofstack.sandbox import subprocess as module
    from proofstack.sandbox.base import WorkerStopState

    async def exercise():
        stream = object.__new__(module._StreamingProcess)
        stream._process_group_stop_state = WorkerStopState.SURVIVING
        stream._memory_lease = None
        stream._resident_limit_bytes = 8 * 1024**3
        stream._process_marker = SimpleNamespace(token="worker", created_at=0)
        stream._emit_memory = AsyncMock()
        stream.terminate = AsyncMock()
        monkeypatch.setattr(module, "markers_rss", lambda markers: {"worker": (9 * 1024**3, [])})
        await stream._watch_memory()
        stream.terminate.assert_awaited_once()
        assert stream.memory_failure == "worker_memory_limit"

    asyncio.run(exercise())


def test_sigterm_enters_async_cleanup():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_workflow.py"
    code = '''
import asyncio, os, runpy, signal, sys
wrapper = runpy.run_path(sys.argv[1])["_run_with_stop_signals"]
async def job():
    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
    try:
        await asyncio.sleep(10)
    finally:
        await asyncio.sleep(0)
        print("async-cleanup-completed", flush=True)
raise SystemExit(asyncio.run(wrapper(job)))
'''
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 143, result.stderr
    assert "async-cleanup-completed" in result.stdout


def test_recovery_inventory_validates_export_without_mutation(tmp_path, monkeypatch):
    import hashlib
    import runpy
    from proofstack.agents.ac.ac_workflow import _problem_hash
    from proofstack.latex_contract import normalize_submission_latex
    inspect_run = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_batch3_recovery.py"))["inspect_run"]
    root = tmp_path / "run"
    (root / "agents/FirstProofBatch3Workflow-c0").mkdir(parents=True)
    (root / "agents/FirstProofBatch3Workflow-c0/input.json").write_text(json.dumps({"problem": "P", "problem_id": "p"}))
    (root / "batch3-schedule.json").write_text(json.dumps({
        "problem_hash": _problem_hash("P"), "initial_usd": 600, "partial_reserve_usd": 90,
        "research_deadline_unix_s": 100, "run_deadline_unix_s": 200,
    }))
    state_dir = root / "ac_workspaces/p/.ac"
    state_dir.mkdir(parents=True)
    (state_dir / "resume-state.json").write_text(json.dumps({"problem_hash": _problem_hash("P"), "problem_id": "p", "last_round_run": 2}))
    (root / "events.jsonl").write_text(json.dumps({"kind": "model.call", "payload": {"cost_usd": 12}}) + "\n")
    (root / "submissions").mkdir()
    text = normalize_submission_latex("\\documentclass{article}\n\\begin{document}Proof.\\end{document}")
    (root / "submissions/p.tex").write_text(text)
    (root / "batch3-output.json").write_text(json.dumps({
        "submission_approved": True, "compiled": True, "pages": 1,
        "submission_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(APIClient, "run_queries", lambda *args, **kw: pytest.fail("unexpected model call"))
    report = inspect_run(root, now=201)
    assert report["proposed_boundary"] == "preserve_approved_no_calls"
    assert report["clock_expired"] and report["original_budget_usd"] == 600
    assert report["cumulative_known_cost_usd"] == 12 and not report["launch_authorized"]
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    (root / "submissions/p.tex").write_text("tampered")
    with pytest.raises(ValueError, match="Approved export failed"):
        inspect_run(root)


def test_audit_bounds_tool_output_and_redacts_before_truncation(monkeypatch):
    from mathagents.request_logger import _audit_data
    secret = "sk-offline-private-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    original = {"type": "code_interpreter_call", "outputs": [{"logs": "x" * 16380 + secret + "x" * 100}]}
    logged = _audit_data(original)
    assert secret in original["outputs"][0]["logs"]
    assert secret not in json.dumps(logged)
    assert len(logged["outputs"][0]["logs"]) < 16500
    assert "truncated" in logged["outputs"][0]["logs"]
