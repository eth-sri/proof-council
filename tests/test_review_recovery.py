"""Offline reproductions for PR 14's second recovery review."""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.provider_trace import ProviderTrace, active_trace
from proofstack import subscription as sub
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.agents.final_stateless_review import FinalStatelessCritic, FinalReviewBlocked
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.budget import BudgetTracker
from proofstack.context import RunContext
from proofstack.provider_accounting import (
    provider_attempts_settled, reconcile_provider_call, reconcile_stopped_provider_call,
)
from test_firstproof_batch3 import TEX, pipeline
from test_proof_simplification import simplification
from test_subscription_workflow import credentials, configured, fake_cli


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.parametrize("with_cost", [False, True])
def test_exhausted_failed_creates_settle_without_response_id(tmp_path, provider, with_cost):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    error = RuntimeError("synthetic provider 503")
    if with_cost:
        error.cost = {"cost": 0}

    def query(*_):
        trace = active_trace.get()
        for i in range(2):
            trace.update((str(i), 0), provider=provider, outcome="started")
            trace.failure((str(i), 0), error_type="ServiceUnavailable", retry_decision="stop" if i else "retry")
        raise error

    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(agent._query(SimpleNamespace(model="fake"), [], query))
    assert error.provider_usage_resolved
    assert json.loads((agent.workdir / "provider-settlement.json").read_text())["settled"]
    assert agent.tracker.counters.usd == 0
    assert asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir))


@pytest.mark.parametrize("open_first", [False, True])
def test_retry_success_does_not_hide_an_open_earlier_response(tmp_path, open_first):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))

    def query(*_):
        trace = active_trace.get()
        trace.update(("first", 0), provider="openai", outcome="started",
                     **({"response_id": "still-open", "status": "in_progress"} if open_first else {}))
        trace.failure(("first", 0), error_type="RateLimitError", retry_decision="retry")
        trace.update(("second", 0), status="completed", response_id="done", cost=2, usage_unavailable=False)
        return "Paid report"

    async def run():
        assert await agent._query(SimpleNamespace(model="fake"), [], query) == "Paid report"
        await agent._record_model_usage({"cost_usd": 2, "usage_unavailable": True}, call_id="paid")
        assert await reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir) is (not open_first)

    asyncio.run(run())
    assert json.loads((agent.workdir / "provider-settlement.json").read_text())["settled"] is (not open_first)


def test_unknown_started_attempt_is_not_a_zero_cost_failure():
    assert not provider_attempts_settled([{"provider": "openai", "outcome": "started"}])
    assert not provider_attempts_settled([{"error_type": "LostConnection", "retry_decision": "stop", "cost": 2}])


@pytest.mark.parametrize("provider", ["anthropic", "google"])
@pytest.mark.parametrize("transport_failure", [False, True])
def test_real_provider_failure_paths_record_settlement(tmp_path, monkeypatch, provider, transport_failure):
    from mathagents import api_client as api
    from mathagents.provider_trace import latest_attempts
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(api.APIClient, "_sleep", lambda *args: None)
    monkeypatch.setattr(api.request_logger, "log_request", lambda **kw: None)
    monkeypatch.setattr(api.request_logger, "log_response", lambda **kw: None)
    def create(**kwargs):
        if transport_failure:
            raise TimeoutError("synthetic transport timeout")
        error = RuntimeError("synthetic provider 503")
        error.status_code = 503
        raise error
    def post(*args, **kwargs):
        if transport_failure:
            raise TimeoutError("synthetic transport timeout")
        return SimpleNamespace(status_code=503, text="Unavailable", json=lambda: {"error": {"code": 503}})
    monkeypatch.setattr(api.anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(api.requests, "post", post)
    monkeypatch.setattr(api, "OpenAI", lambda **kw: pytest.fail("Unexpected OpenAI transport"))
    client = api.APIClient(api=provider, model="synthetic-model",
                           max_tokens=10, max_retries_inner=0, use_gdm_tools=True,
                           tools=[(None, {"type": "web_search_preview"})])
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    with pytest.raises(Exception):
        asyncio.run(agent._query(client, [], lambda *_: client._run_query(0, [{"role": "user", "content": "Synthetic"}])))
    rows = latest_attempts(agent.workdir / "provider-attempts.jsonl")
    assert rows and all(row.get("error_type") for row in rows)
    assert provider_attempts_settled(rows) is (not transport_failure)
    assert asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir)) is (not transport_failure)


@pytest.mark.parametrize("field", ["reasoning_tokens", "cached_input_tokens", "cached_write_tokens"])
def test_failed_create_with_partial_usage_cannot_be_discarded(field):
    assert not provider_attempts_settled([{"error_type": "ServiceUnavailable", "retry_decision": "stop", field: 3}])


@pytest.mark.parametrize("status", ["in_progress", "cancelled"])
def test_timeout_with_response_id_uses_actual_terminal_evidence(status):
    from mathagents.api_client import APIClient
    trace = ProviderTrace(None)
    trace.update(("call", 0), provider="openai", response_id="known", status=status, usage_unavailable=False)
    token = active_trace.set(trace)
    try:
        APIClient._report_attempt_failure(None, "call", 0, TimeoutError("synthetic"), retry_decision="stop")
    finally:
        active_trace.reset(token)
    assert provider_attempts_settled(trace.attempts.values()) is (status == "cancelled")


def test_malformed_response_without_id_is_not_a_zero_cost_rejection():
    from mathagents.api_client import APIClient
    trace = ProviderTrace(None)
    trace.update(("call", 0), provider="google", outcome="started")
    token = active_trace.set(trace)
    try:
        APIClient._report_attempt_failure(None, "call", 0, json.JSONDecodeError("Malformed reply", "", 0),
                                          retry_decision="stop")
    finally:
        active_trace.reset(token)
    assert not provider_attempts_settled(trace.attempts.values())


@pytest.mark.parametrize("cancel", [False, True])
def test_supplementary_settlement_disk_failure_preserves_reply_or_cancel(tmp_path, monkeypatch, cancel):
    import proofstack.atomic as atomic
    write = atomic.write_text_atomic
    def disk_full(path, text):
        if path.name == "provider-settlement.json":
            raise OSError("synthetic disk full")
        return write(path, text)
    monkeypatch.setattr(atomic, "write_text_atomic", disk_full)
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    entered, stopped = threading.Event(), threading.Event()
    def query(*_):
        active_trace.get().update(("paid", 0), response_id="done", status="completed", cost=2, usage_unavailable=False)
        entered.set()
        if cancel:
            assert stopped.wait(5)
        return "Paid report"
    async def run():
        task = asyncio.create_task(agent._query(SimpleNamespace(model="fake", terminate=stopped.set), [], query, call_id="paid"))
        assert await asyncio.to_thread(entered.wait, 5)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == "Paid report"
            await agent._record_model_usage({"cost_usd": 2}, call_id="paid")
        assert await reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir)
    asyncio.run(run())
    assert agent.tracker.counters.usd == 2


def test_api_operator_settlement_is_bound_to_evidence_and_idempotent(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    agent = ACCritic(ctx)
    trace = ProviderTrace(agent.workdir / "provider-attempts.jsonl", call_id="open")
    trace.update(("open", 0), response_id="provider-open", status="in_progress", cost=2, usage_unavailable=False)
    assert not asyncio.run(reconcile_provider_call(ctx, agent.tracker, agent.workdir))
    for confirmations in ({}, {"workers_stopped": True}, {"provider_work_terminal": True}):
        with pytest.raises(ValueError, match="Confirm"):
            reconcile_stopped_provider_call(tmp_path, agent.workdir, unreported_allowance_usd=5, **confirmations)
    for _ in range(2):
        reconcile_stopped_provider_call(tmp_path, agent.workdir, workers_stopped=True,
                                       provider_work_terminal=True, unreported_allowance_usd=5)
        assert asyncio.run(reconcile_provider_call(ctx, agent.tracker, agent.workdir))
        assert ctx.budgets.root().counters.usd == 7
    with pytest.raises(ValueError, match="different"):
        reconcile_stopped_provider_call(tmp_path, agent.workdir, workers_stopped=True,
                                       provider_work_terminal=True, unreported_allowance_usd=4)
    trace.update(("new", 0), outcome="started")
    assert not asyncio.run(reconcile_provider_call(ctx, agent.tracker, agent.workdir))
    assert len(json.loads((tmp_path / "recovery-debits.json").read_text())) == 1
    (tmp_path / "execution-backend.json").write_text(json.dumps({"backend": "subscription"}))
    with pytest.raises(ValueError, match="reconcile_subscription"):
        reconcile_stopped_provider_call(tmp_path, agent.workdir, workers_stopped=True,
                                       provider_work_terminal=True, unreported_allowance_usd=0)


@pytest.mark.parametrize("preflight", [False, True])
def test_failed_simplification_query_continues_to_cleanup_and_final_gate(simplification, monkeypatch, preflight):
    execute, calls, _, _, ctx, _ = simplification
    async def failed_author(self, inp):
        calls["simplify_author"] += 1
        if preflight:
            raise ValueError("synthetic context preflight; no request")
        def query(*_):
            trace = active_trace.get()
            trace.update(("create", 0), outcome="started", provider="openai")
            trace.failure(("create", 0), error_type="ServiceUnavailable", retry_decision="stop")
            raise RuntimeError("synthetic 503")
        return await self._query(SimpleNamespace(model="fake"), [], query)
    monkeypatch.setattr(Author, "run", failed_author)
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.submission_approved and not out.accepted_baseline_pending
    assert calls["simplify_author"] == calls["rewrite"] == calls["final_critic"] == 1


def test_cleanup_reconciliation_keeps_baseline_and_previous_partial(simplification):
    from scripts import firstproof_entrypoint as entry
    execute, calls, _, options, ctx, _ = simplification
    options["rewrite_error"] = sub.SubscriptionBlocked("Synthetic unresolved cleanup")
    out = asyncio.run(execute(simplification_rounds=1))
    assert out.accepted_baseline_pending and out.partial_ready
    assert not out.submission_approved and not out.error_retryable and not calls["final_critic"]
    partial, digest = out.answer_tex.read_bytes(), out.partial_sha256
    # Exercise the official adapter's publication decision, not just model fields.
    settings = SimpleNamespace(output_dir=ctx.root_workdir.parent.parent, page_limit=200,
                               stop_requested=threading.Event(), deadline_at=None, batch3=True)
    problem = SimpleNamespace(run_id=ctx.root_workdir.name, safe_id="p")
    assert entry._published_solution(problem, settings)[1].encode() == partial
    options.pop("rewrite_error")
    # An unresolved cleanup ledger on resume still retains the previous publication.
    (ctx.root_workdir / "cleanup-accounting-uncertain.json").write_text("{}")
    resumed = asyncio.run(execute(resume_run=True))
    assert resumed.accepted_baseline_pending and resumed.partial_sha256 == digest
    assert resumed.answer_tex.read_bytes() == partial


@pytest.mark.parametrize("open_attempt", [False, True])
def test_paid_final_verdict_recovers_before_exhausted_allowance(simplification, monkeypatch, open_attempt):
    execute, calls, seen, _, ctx, agent = simplification
    asyncio.run(execute(final_stateless_review=True))
    [state_path] = (ctx.root_workdir / "final_stateless_review").glob("*.json")
    state = json.loads(state_path.read_text())
    critic = FinalStatelessCritic(ctx, name="final_stateless_critic")
    inp = critic.Inputs.model_validate(state["review_input"])
    checkpoint = critic._recovery_checkpoint(inp)
    folder = ctx.root_workdir / "agents" / "paid-final"
    folder.mkdir()
    trace = ProviderTrace(folder / "provider-attempts.jsonl", call_id="saved-final")
    trace.update(("last", 0), response_id="paid-final", status="in_progress" if open_attempt else "completed",
                 cost=11, usage_unavailable=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": "completed", "output": state["review"],
                                      "attempt_workdir": str(folder.relative_to(ctx.root_workdir))}))
    state.update(status="pending", review=None, max_usd=1, initial_spend=0, deadline_unix_s=time.time() - 1)
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(agent, "_phase", AsyncMock(side_effect=AssertionError("no new call")))
    parent_input = agent.Inputs(problem=inp.problem, problem_id="p", final_stateless_review=True)
    if open_attempt:
        with pytest.raises(FinalReviewBlocked, match="unresolved"):
            asyncio.run(agent._review_final(parent_input, inp.answer_tex, deadline=time.monotonic() - 1))
    else:
        result = asyncio.run(agent._review_final(parent_input, inp.answer_tex, deadline=time.monotonic() - 1))
        assert result.answer_ready
        assert json.loads(state_path.read_text())["status"] == "completed"
    assert calls["final_critic"] == 1


def test_operator_api_settlement_unblocks_workflow_without_changing_limits(simplification, monkeypatch):
    execute, calls, _, _, ctx, _ = simplification
    async def failed_author(self, inp):
        calls["simplify_author"] += 1
        def query(*_):
            active_trace.get().update(("open", 0), response_id="unresolved", status="in_progress",
                                      cost=2, usage_unavailable=False)
            raise RuntimeError("synthetic lost polling connection")
        return await self._query(SimpleNamespace(model="fake"), [], query)
    monkeypatch.setattr(Author, "run", failed_author)
    out = asyncio.run(execute(simplification_rounds=1))
    assert out.accepted_baseline_pending and out.partial_ready
    [state] = (ctx.root_workdir / "proof_simplification").glob("*/state.json")
    name = json.loads(state.read_text())["call_agent"]
    [folder] = (ctx.root_workdir / "agents").glob(f"{name}-c*")
    schedule = (ctx.root_workdir / "batch3-schedule.json").read_bytes()
    reconcile_stopped_provider_call(ctx.root_workdir, folder, workers_stopped=True,
                                   provider_work_terminal=True, unreported_allowance_usd=3)
    result = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert result.submission_approved
    assert calls["simplify_author"] == calls["rewrite"] == calls["final_critic"] == 1
    assert (ctx.root_workdir / "batch3-schedule.json").read_bytes() == schedule
    assert ctx.budgets.root().spec.max_usd == 100


def test_subscription_final_receipt_recovers_with_no_new_call_allowance(configured, fake_cli, monkeypatch):
    from proofstack.budget import BudgetSpec
    workflow, _ = configured
    parent = FirstProofBatch3Workflow(workflow.ctx)
    inp = parent.Inputs(problem="Synthetic question", problem_id="synthetic", final_stateless_review=True)
    _, review_input = parent._final_review_identity(inp, TEX)
    critic = FinalStatelessCritic(workflow.ctx, name="final_stateless_critic")
    result = asyncio.run(critic(**review_input.model_dump()))
    assert result.answer_ready
    workflow.ctx.budgets.root().spec = BudgetSpec(max_usd=0)
    monkeypatch.setattr(parent, "_phase", AsyncMock(side_effect=AssertionError("no new call")))
    recovered = asyncio.run(parent._review_final(inp, TEX, deadline=time.monotonic() + 30))
    assert recovered.answer_ready and len(fake_cli) == 1


def test_nonzero_cli_exit_settles_usage_and_allows_one_retry(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    command = ConfigurableCLIAgent._command_for
    def fail(self, inp):
        cmd = command(self, inp)
        return [*cmd[:-1], cmd[-1] + "\nsys.exit(1)\n"]
    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", fail)
    for _ in range(2):
        with pytest.raises(sub.SubscriptionCallFailed):
            asyncio.run(sub.run_packet(workflow, "same failed packet"))
        sub.check_pending_calls(workflow.ctx.root_workdir)
    with pytest.raises(sub.SubscriptionCallFailed, match="bounded retry"):
        asyncio.run(sub.run_packet(workflow, "same failed packet"))
    assert len(fake_cli) == 2 and workflow.tracker.planning_usd > 0
    assert workflow.tracker.counters.usd == 0


def test_cancelled_cli_debits_call_allowance_but_preserves_cleanup_reserve(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    parent = FirstProofBatch3Workflow(workflow.ctx)
    phase = parent._phase_context("research", usd=10, seconds=30)
    researcher = ACCritic(phase)
    started = workflow.ctx.root_workdir / "worker-ready"
    command = ConfigurableCLIAgent._command_for
    def hang(self, inp):
        cmd = command(self, inp)
        return [*cmd[:-1], cmd[-1] + f"\nPath({str(started)!r}).touch()\nimport time\ntime.sleep(60)\n"]
    async def run():
        with monkeypatch.context() as patch:
            patch.setattr(ConfigurableCLIAgent, "_command_for", hang)
            task = asyncio.create_task(sub.run_packet(researcher, "interrupted research"))
            async with asyncio.timeout(10):
                while not started.exists():
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        sub.check_pending_calls(workflow.ctx.root_workdir)
        assert workflow.ctx.budgets.root().planning_usd == pytest.approx(10)
        # A different packet can use the retained parent budget immediately.
        assert (await sub.run_packet(workflow, "cleanup"))["status"] == "done"
    asyncio.run(run())
    tracker = BudgetTracker("restored", subscription_mode=True)
    sub.restore_usage(workflow.ctx, tracker)
    assert tracker.planning_usd == pytest.approx(workflow.ctx.budgets.root().planning_usd)
    assert 10 < tracker.planning_usd < 11 and tracker.counters.usd == 0


def test_queued_packet_rechecks_unresolved_sibling(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    entered, release = asyncio.Event(), asyncio.Event()
    async def fail(self, inp):
        entered.set()
        await release.wait()
        raise RuntimeError("unknown stop")
    monkeypatch.setattr(ConfigurableCLIAgent, "run", fail)
    async def run():
        workflow.ctx.subscription_limiter = asyncio.Semaphore(1)
        first = asyncio.create_task(sub.run_packet(workflow, "first"))
        await entered.wait()
        second = asyncio.create_task(sub.run_packet(workflow, "second"))
        await asyncio.sleep(0)
        release.set()
        for task in (first, second):
            with pytest.raises(sub.SubscriptionBlocked):
                await task
    asyncio.run(run())
    assert len(list((workflow.ctx.root_workdir / "subscription_calls").glob("*.json"))) == 1
    assert not fake_cli


def test_manual_reconciliation_recovers_completed_output_without_extra_debit(configured, fake_cli):
    workflow, _ = configured
    output = asyncio.run(sub.run_packet(workflow, "completed review"))
    root = workflow.ctx.root_workdir
    [receipt] = (root / "subscription_calls").glob("*.json")
    saved = json.loads(receipt.read_text())
    saved.update(status="started")
    saved.pop("output")
    receipt.write_text(json.dumps(saved))
    before = set((root / "subscription_usage").glob("*.json"))
    result = sub.reconcile_stopped_call(root, receipt.stem, workers_stopped=True, unreported_allowance_usd=100)
    assert result["status"] == "completed" and result["output"] == output
    assert set((root / "subscription_usage").glob("*.json")) == before


def test_manual_reconciliation_requires_override_for_negative_stop_evidence(configured):
    workflow, _ = configured
    root = workflow.ctx.root_workdir
    receipt = root / "subscription_calls" / ("a" * 64 + ".json")
    receipt.parent.mkdir()
    receipt.write_text('{"status":"started","attempts":1}')
    sub._settlement_path(receipt).parent.mkdir()
    sub._settlement_path(receipt).write_text('{"settled":false,"process_stopped":false}')
    with pytest.raises(sub.SubscriptionBlocked, match="override"):
        sub.reconcile_stopped_call(root, receipt.stem, workers_stopped=True, unreported_allowance_usd=5)
    sub.reconcile_stopped_call(root, receipt.stem, workers_stopped=True, unreported_allowance_usd=5,
                               override_stop_evidence=True)
    sub.check_pending_calls(root)


@pytest.mark.parametrize("crash_point", ["before_unlink", "after_unlink"])
def test_retry_setup_crash_is_reconcilable_and_cannot_reuse_old_settlement(configured, fake_cli, monkeypatch, crash_point):
    workflow, _ = configured
    command = ConfigurableCLIAgent._command_for
    def fail(self, inp):
        cmd = command(self, inp)
        return [*cmd[:-1], cmd[-1] + "\nsys.exit(1)\n"]
    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", fail)
    with pytest.raises(sub.SubscriptionCallFailed):
        asyncio.run(sub.run_packet(workflow, "retry packet"))
    root = workflow.ctx.root_workdir
    [receipt] = (root / "subscription_calls").glob("*.json")
    unlink = Path.unlink
    def crash(path, *args, **kwargs):
        if path == sub._settlement_path(receipt):
            if crash_point == "after_unlink":
                unlink(path, *args, **kwargs)
            raise OSError("synthetic retry setup crash")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", crash)
        with pytest.raises(sub.SubscriptionBlocked):
            asyncio.run(sub.run_packet(workflow, "retry packet"))
    assert json.loads(receipt.read_text())["attempts"] == 2
    assert not sub._settled(receipt)
    sub.reconcile_stopped_call(root, receipt.stem, workers_stopped=True, unreported_allowance_usd=0)
    sub.check_pending_calls(root)
    assert len(fake_cli) == 1


def test_settled_failed_receipt_is_not_rewritten(configured, monkeypatch):
    workflow, _ = configured
    root = workflow.ctx.root_workdir
    receipt = root / "subscription_calls" / ("b" * 64 + ".json")
    receipt.parent.mkdir()
    receipt.write_text('{"status":"failed","attempts":1}')
    sub._settlement_path(receipt).parent.mkdir()
    sub._settlement_path(receipt).write_text('{"settled":true,"workdir":"agents/stopped"}')
    monkeypatch.setattr(sub, "write_text_atomic", lambda *args: pytest.fail("read-only check rewrote receipt"))
    for _ in range(2):
        sub.check_pending_calls(root)


@pytest.mark.parametrize("usage", [False, True])
def test_finished_native_compute_usage_is_durable_or_estimated(configured, fake_cli, monkeypatch, usage):
    from proofstack.agents.ac.compute import Compute
    workflow, inp = configured
    inp = inp.model_copy(update={"compute_sandbox_backend": "subprocess", "compute_max_parallel_workers": 0,
        "compute_filesystem_min_free_bytes": 0, "compute_filesystem_min_free_inodes": 0,
        "compute_filesystem_reservation_bytes": 0})
    script = """
import os, sys, json, subprocess
from pathlib import Path
sys.stdin.read()
assert not any(k.endswith('API_KEY') for k in os.environ)
Path('responses').mkdir(exist_ok=True)
Path('responses/response_round_1.md').write_text('Synthetic computation: 1+1=2')
"""
    if usage:
        script += "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}), flush=True)\n"
    script += "subprocess.run([os.environ['FINISH_BIN'], json.dumps({'status':'done','summary':'Synthetic computation'})], check=True)\n"
    setup = Compute.setup
    async def synthetic_setup(self, sandbox, inp):
        async def auth_status(cmd, **kwargs):
            assert cmd == ["codex", "login", "status"]
            return SimpleNamespace(returncode=0, stdout="Synthetic subscription login", stderr="")
        with monkeypatch.context() as patch:
            patch.setattr(sandbox, "run_command", auth_status)
            await setup(self, sandbox, inp)
        self.CLI_CMD = [sys.executable, "-c", script]
    monkeypatch.setattr(Compute, "setup", synthetic_setup)
    monkeypatch.setattr(Compute, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr("proofstack.agents.ac.compute._require_codex_cli_version", AsyncMock())
    async def run():
        result = await sub.run_with_nominal_allowance(workflow.compute, 4, lambda:
            workflow._safe_compute(inp=inp, round=1, instructions="Count", answer_tex="",
            research_notes_tex="", references_bib="", compute_workspace=workflow.ctx.root_workdir / "compute"))
        assert result.status == "done" and "1+1=2" in result.response_md, result.model_dump()
    asyncio.run(run())
    sub.check_pending_calls(workflow.ctx.root_workdir)
    if usage:
        assert 0 < workflow.ctx.budgets.root().planning_usd < .01
    else:
        assert workflow.ctx.budgets.root().planning_usd == 4
    assert list((workflow.ctx.root_workdir / "subscription_calls").glob("*.json"))
    assert not fake_cli


def test_unfinished_subscription_research_runs_partial_cleanup_after_cutoff(configured, fake_cli, monkeypatch):
    from contextlib import asynccontextmanager
    from proofstack.agents import cleanup_session as cleanup
    from proofstack.agents.ac import ac_workflow as ac
    workflow, _ = configured
    monkeypatch.setattr(ac, "_simple_compile_latex", lambda tex, *args, **kwargs:
                        ac._CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1))
    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", AsyncMock(return_value=(True, 1, "")))
    monkeypatch.setattr(cleanup.CleanupSession, "_preflight", AsyncMock())
    sandbox = cleanup.CleanupSession._sandbox
    monkeypatch.setattr(cleanup.CleanupSession, "_sandbox", lambda *args: {**sandbox(*args), "memory_policy": None})
    monkeypatch.setattr(cleanup.CleanupReviews, "attribution_retrieved", lambda self: True)
    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
    monkeypatch.setattr(cleanup, "cleanup_tools", tools)
    command = ConfigurableCLIAgent._command_for
    started = workflow.ctx.root_workdir / "critic-started"
    def slow_critic(self, inp):
        cmd = command(self, inp)
        if "<answer_ready>" in inp.packet:
            # All research critics have this verdict contract; the first one
            # hangs until the parent delivers the research-cutoff cancellation.
            return [*cmd[:-1], cmd[-1] + f"\nPath({str(started)!r}).touch()\nimport time\ntime.sleep(60)\n"]
        return cmd
    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", slow_critic)
    agent = FirstProofBatch3Workflow(workflow.ctx)
    phase = agent._phase
    async def cutoff(name, **kwargs):
        if name != "research":
            return await phase(name, **kwargs)
        task = asyncio.create_task(phase(name, **kwargs))
        try:
            async with asyncio.timeout(10):
                while not started.exists():
                    if task.done():
                        return await task
                    await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                raise TimeoutError("Synthetic research cutoff") from None
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    monkeypatch.setattr(agent, "_phase", cutoff)
    out = asyncio.run(agent(problem="Synthetic question", problem_id="synthetic", execution_backend="subscription",
        n_rounds=2, enable_compute=False, enable_council=False, research_seconds=60,
        partial_cleanup_seconds=20, partial_cleanup_reserve_fraction=0.15))
    assert started.exists(), "The research critic must actually be interrupted"
    assert out.partial_ready and not out.submission_approved
    assert out.error is None and out.error_retryable
    assert any(cmd[0] == "claude" for cmd, _, _ in fake_cli), "Partial editor never ran"
    sub.check_pending_calls(workflow.ctx.root_workdir)


def test_parallel_subscription_shutdown_shares_one_phase_allowance(configured, fake_cli, monkeypatch):
    from proofstack.agents.ac.ac_workflow import ACWorkflow
    from proofstack.agents.ac.compute import Compute
    workflow, inp = configured
    parent = FirstProofBatch3Workflow(workflow.ctx)
    phase = parent._phase_context("research", usd=12, seconds=30)
    phase.subscription_limiter = asyncio.Semaphore(4)
    researcher = ACWorkflow(phase)
    started = workflow.ctx.root_workdir / "started-workers"
    started.mkdir()
    command = ConfigurableCLIAgent._command_for
    def hang(self, inp):
        cmd = command(self, inp)
        return [*cmd[:-1], cmd[-1] + f"\nPath({str(started / self.name)!r}).touch()\nimport time\ntime.sleep(60)\n"]
    # Native Compute's usage guard has a separate subprocess test above. Here
    # use the packet transport to exercise all four concurrent budget branches.
    async def compute(self, inp):
        await sub.run_packet(self, "Synthetic compute")
    monkeypatch.setattr(Compute, "run", compute)
    workspace = workflow.ctx.root_workdir / "synthetic-workspace"
    workspace.mkdir()
    for name, text in (("answer.tex", TEX), ("research_notes.tex", ""), ("references.bib", "")):
        (workspace / name).write_text(text)
    async def run():
        with monkeypatch.context() as patch:
            patch.setattr(ConfigurableCLIAgent, "_command_for", hang)
            task = asyncio.create_task(researcher._gather_critic_council(inp=inp, workspace=workspace,
                author_k=Author.Outputs(answer_tex=TEX, council_question="Synthetic question", compute_instructions="Count"),
                critic_conversation=[], mode="fresh", round=1, run_council=True, run_compute=True))
            try:
                async with asyncio.timeout(10):
                    while len(list(started.iterdir())) < 4:
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        sub.check_pending_calls(workflow.ctx.root_workdir)
        assert phase.budgets.root().planning_usd == pytest.approx(12)
        assert (await sub.run_packet(workflow, "cleanup after fanout"))["status"] == "done"
    asyncio.run(run())
    assert len(fake_cli) == 5
    assert 12 < workflow.ctx.budgets.root().planning_usd < 13
