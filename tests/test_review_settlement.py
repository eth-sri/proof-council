"""Synthetic provider/CLI regressions for the third PR 14 recovery review."""
import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace, active_trace, latest_attempts
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.compute import Compute
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.agents.final_stateless_review import FinalStatelessCritic, FinalReviewBlocked
from proofstack.budget import BudgetSpec, BudgetTracker, BudgetExhausted
from proofstack.context import RunContext
from proofstack.kinds.api_call import APICallAgent
from proofstack.provider_accounting import (
    provider_attempts_settled, reconcile_provider_call, reconcile_stopped_provider_call,
)
from proofstack import subscription as sub
from test_research_critic_context import packet
from test_subscription_workflow import credentials, configured, fake_cli
from test_firstproof_batch3 import pipeline
from test_proof_simplification import simplification


@pytest.mark.parametrize("failure", ["transport", "legacy", "rejected_create"])
def test_transport_failure_does_not_wedge_ordinary_critic(tmp_path, monkeypatch, failure):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda *_: pytest.fail("Unexpected provider client"))
    calls = []

    async def reply(self, inp):
        calls.append(self.workdir)
        def query(client, messages):
            trace = active_trace.get()
            trace.request(client, "outage", 0, {})
            error = TimeoutError("Synthetic outage")
            if failure == "rejected_create":
                error.status_code = 503
            APIClient._report_attempt_failure(None, "outage", 0, error, retry_decision="stop")
            if failure == "legacy":
                trace.update(("outage", 0), transport_uncertain=False, reconciliation_pending=True)
            if len(calls) == 1:
                raise error
            trace.update(("retry", 0), provider="openai", response_id="paid", status="completed",
                         cost=2, usage_unavailable=False)
            return "Paid review\n<answer_ready>true</answer_ready>"
        raw = await self._query(SimpleNamespace(api="openai", model="synthetic"), [], query)
        return self.parse_output(raw, inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    inp = packet(research_notes_tex="").model_dump()
    async def run():
        with pytest.raises(TimeoutError):
            await ACCritic(ctx)(**inp)
        assert (await ACCritic(ctx)(**inp)).answer_ready
        assert (await ACCritic(ctx)(**inp)).answer_ready
        assert await reconcile_provider_call(ctx, ctx.budgets.root(), calls[-1]) is (failure == "rejected_create")
        if failure != "rejected_create":
            reconcile_stopped_provider_call(tmp_path, calls[-1], workers_stopped=True,
                provider_work_terminal=True, unreported_allowance_usd=5)
        assert (await ACCritic(ctx)(**inp)).answer_ready
        assert await reconcile_provider_call(ctx, ctx.budgets.root(), calls[-1])
        assert ctx.budgets.root().counters.usd == (2 if failure == "rejected_create" else 7)
    asyncio.run(run())
    assert len(calls) == 2


@pytest.mark.parametrize("status", [408, 504, 524])
def test_http_timeout_is_not_a_zero_cost_rejection(status):
    trace = ProviderTrace(None)
    trace.update(("call", 0), provider="openai", outcome="error")
    error = RuntimeError("Synthetic HTTP timeout")
    error.status_code = status
    token = active_trace.set(trace)
    try:
        APIClient._report_attempt_failure(None, "call", 0, error, retry_decision="stop")
    finally:
        active_trace.reset(token)
    assert not provider_attempts_settled(trace.attempts.values())


@pytest.mark.parametrize("terminal", [False, True])
def test_late_response_resolves_transport_uncertainty_only_when_terminal(terminal):
    row = {"provider": "openai", "transport_uncertain": True, "error_type": "TimeoutError",
           "retry_decision": "stop", "response_id": "late-response",
           "status": "completed" if terminal else "in_progress", "cost": 2, "usage_unavailable": False}
    assert provider_attempts_settled([row]) is terminal


def test_shutdown_share_is_not_a_live_spending_limit():
    root = BudgetTracker("run", BudgetSpec(max_usd=1000), subscription_mode=True)
    child = BudgetTracker("critic", parent=root)
    agent = SimpleNamespace(tracker=child)
    async def review():
        assert sub.call_allowance(child) == 1
        child.add_api_equivalent_usd(2)
        child.check()
        return "Paid review"
    assert asyncio.run(sub.run_with_nominal_allowance(agent, 1, review)) == "Paid review"
    assert child.spec is None and sub.call_allowance(child) == 998


@pytest.mark.parametrize("operator", [False, True])
def test_raw_final_report_recovers_before_budget_gate(tmp_path, operator):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_usd=1),
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda *_: pytest.fail("Recovery must not construct a provider client"))
    critic = FinalStatelessCritic(ctx)
    inp = packet(mode="fresh", prior_messages=[], research_notes_tex="")
    folder = tmp_path / "agents" / "paid-final"
    trace = ProviderTrace(folder / "provider-attempts.jsonl", call_id="paid")
    if operator:
        trace.update(("uncertain", 0), error_type="TimeoutError", retry_decision="stop", transport_uncertain=True)
    trace.update(("request", 0), provider="openai", response_id="paid-response", status="completed",
                 cost=2, usage_unavailable=False)
    (folder / f"provider-completed-{trace.id}.json").write_text(json.dumps({
        "invocation_id": trace.id, "report": "Paid approval\n<answer_ready>true</answer_ready>", "tool_evidence": []}))
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": str(folder.relative_to(tmp_path)), "review_input": inp.model_dump(mode="json")}))
    if operator:
        with pytest.raises(FinalReviewBlocked, match="unresolved"):
            asyncio.run(critic.recover_completed(inp))
        reconcile_stopped_provider_call(tmp_path, folder, workers_stopped=True,
            provider_work_terminal=True, unreported_allowance_usd=3)
    for _ in range(2):
        assert asyncio.run(critic.recover_completed(inp)).answer_ready
    assert ctx.budgets.root().counters.usd == (5 if operator else 2)
    assert json.loads(checkpoint.read_text())["status"] == "completed"
    critic.verdict_attempt = 1
    assert asyncio.run(critic.recover_completed(inp)) is None


@pytest.mark.parametrize("tail", ["finish_then_drain", "clean_exit_without_finish", "finish_no_usage",
                                   "error", "blocked", "timeout"])
def test_finished_compute_uses_metered_cost(configured, fake_cli, monkeypatch, tail):
    workflow, inp = configured
    inp = inp.model_copy(update={"compute_sandbox_backend": "subprocess", "compute_max_parallel_workers": 0,
        "compute_filesystem_min_free_bytes": 0, "compute_filesystem_min_free_inodes": 0,
        "compute_filesystem_reservation_bytes": 0})
    script = """
import os, sys, json, subprocess, time
from pathlib import Path
sys.stdin.read()
assert not any(k.endswith('API_KEY') for k in os.environ)
Path('responses').mkdir(exist_ok=True)
Path('responses/response_round_1.md').write_text('Synthetic computation: 1+1=2')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}), flush=True)
"""
    if tail == "finish_no_usage":
        script = script[:script.index("print(json.dumps")]
    if tail != "clean_exit_without_finish":
        status = tail if tail in {"error", "blocked", "timeout"} else "done"
        script += f"subprocess.run([os.environ['FINISH_BIN'], json.dumps({{'status':{status!r},'summary':'Done'}})], check=True)\ntime.sleep(60)\n"
    setup = Compute.setup
    async def synthetic_setup(self, sandbox, inp):
        async def auth_status(cmd, **kwargs):
            assert cmd == ["codex", "login", "status"]
            return SimpleNamespace(returncode=0, stdout="Synthetic login", stderr="")
        with monkeypatch.context() as patch:
            patch.setattr(sandbox, "run_command", auth_status)
            await setup(self, sandbox, inp)
        self.CLI_CMD = [sys.executable, "-c", script]
    monkeypatch.setattr(Compute, "setup", synthetic_setup)
    monkeypatch.setattr(Compute, "POLL_INTERVAL_S", .01)
    monkeypatch.setattr(Compute, "DONE_DRAIN_GRACE_S", .05)
    monkeypatch.setattr("proofstack.agents.ac.compute._require_codex_cli_version", AsyncMock())
    result = asyncio.run(sub.run_with_nominal_allowance(workflow.compute, 4, lambda:
        workflow._safe_compute(inp=inp, round=1, instructions="Count", answer_tex="",
        research_notes_tex="", references_bib="", compute_workspace=workflow.ctx.root_workdir / "compute")))
    assert "1+1=2" in result.response_md
    if tail == "finish_no_usage":
        assert workflow.ctx.budgets.root().planning_usd == 4
        assert len(list((workflow.ctx.root_workdir / "subscription_usage").glob("interrupted-*.json"))) == 1
    else:
        assert 0 < workflow.ctx.budgets.root().planning_usd < .01
        assert not list((workflow.ctx.root_workdir / "subscription_usage").glob("interrupted-*.json"))
    sub.check_pending_calls(workflow.ctx.root_workdir)
    assert not fake_cli


@pytest.mark.parametrize("role,over_budget", [("author", False), ("critic", False), ("author", True), ("critic", True)])
def test_successful_simplification_blocks_on_earlier_open_attempt(simplification, monkeypatch, role, over_budget):
    from proofstack.agents.proof_simplification import SimplificationCritic
    execute, calls, _, _, ctx, _ = simplification
    cls = Author if role == "author" else SimplificationCritic
    original = cls.run
    async def reply(self, inp):
        def query(*_):
            trace = active_trace.get()
            trace.update(("first", 0), provider="openai", response_id="still-running", status="in_progress")
            trace.update(("retry", 0), provider="openai", response_id="finished", status="completed",
                         cost=1, usage_unavailable=False)
            return "Synthetic completed reply"
        await self._query(SimpleNamespace(model="synthetic"), [], query)
        result = await original(self, inp)
        if over_budget:
            error = BudgetExhausted("synthetic", "usd", 1, 2)
            error.completed_output = result
            raise error
        return result
    monkeypatch.setattr(cls, "run", reply)
    out = asyncio.run(execute(simplification_rounds=1))
    assert not out.submission_approved and calls["rewrite"] == calls["final_critic"] == 0
    assert out.accepted_baseline_pending and out.accepted_baseline_path.exists()
    assert out.partial_ready and out.output_kind == "partial_unreviewed"
    state = json.loads(next((ctx.root_workdir / "proof_simplification").glob("*/state.json")).read_text())
    assert state["call_pending"] and state["candidate"]


def test_no_spawn_budget_failure_settles_without_charge(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    original = ConfigurableCLIAgent._run_once
    async def exhaust(self, inp, **kw):
        self.tracker.spec = BudgetSpec(max_tool_calls=0)
        return await original(self, inp, **kw)
    monkeypatch.setattr(ConfigurableCLIAgent, "_run_once", exhaust)
    with pytest.raises(BudgetExhausted):
        asyncio.run(sub.run_packet(workflow, "Synthetic no-spawn budget failure"))
    sub.check_pending_calls(workflow.ctx.root_workdir)
    receipt = next((workflow.ctx.root_workdir / "subscription_calls").glob("*.json"))
    settlement = json.loads(sub._settlement_path(receipt).read_text())
    assert settlement["settled"] and not settlement["spawn_attempted"]
    assert workflow.ctx.budgets.root().planning_usd == 0


def test_preflight_without_marker_is_settled_but_global_crash_is_not(tmp_path):
    folder = tmp_path / "agents" / "review"
    folder.mkdir(parents=True)
    assert sub.local_call_settlement(tmp_path, folder) == (True, None)
    receipts = tmp_path / "subscription_calls"
    receipts.mkdir()
    (receipts / ("a" * 64 + ".json")).write_text(json.dumps({"status": "started", "attempts": 1}))
    with pytest.raises(sub.SubscriptionBlocked):
        sub.local_call_settlement(tmp_path, folder)


@pytest.mark.parametrize("error", ["context", "programming"])
def test_settled_cleanup_error_routes_baseline_through_final_gate(simplification, monkeypatch, error):
    from proofstack.agents.batch3_critic import CleanupContextTooLarge
    execute, calls, _, options, ctx, _ = simplification
    options["rewrite_error"] = CleanupContextTooLarge("Synthetic overflow") if error == "context" else TypeError("Synthetic bug")
    out = asyncio.run(execute(simplification_rounds=0, final_stateless_review=True))
    assert out.output_kind == "accepted_solution" and out.submission_approved
    assert not out.accepted_baseline_pending
    assert calls["final_critic"] == 1 and calls["rewrite"] == 1
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "batch3.recovery_blocked" and e["payload"]["operator_action_required"] for e in events)


@pytest.mark.parametrize("status", [200, 502, 504])
def test_google_html_error_preserves_http_status(tmp_path, monkeypatch, status):
    from mathagents import api_client as api
    import requests
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(api.APIClient, "_sleep", lambda *args: None)
    monkeypatch.setattr(api.request_logger, "log_request", lambda **kw: None)
    monkeypatch.setattr(api.request_logger, "log_response", lambda **kw: None)
    response = requests.Response()
    response.status_code, response._content = status, b"<html>synthetic gateway reply</html>"
    monkeypatch.setattr(api.requests, "post", lambda *args, **kw: response)
    client = APIClient(api="google", model="synthetic", max_tokens=10, max_retries_inner=0,
                       use_gdm_tools=True, tools=[(None, {"type": "web_search_preview"})])
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    with pytest.raises(Exception):
        asyncio.run(agent._query(client, [], lambda *_: client._run_query(0, [{"role": "user", "content": "Synthetic"}])))
    rows = latest_attempts(agent.workdir / "provider-attempts.jsonl")
    assert provider_attempts_settled(rows) is (status == 502)
    assert rows[-1]["error_type"] == ("JSONDecodeError" if status == 200 else "HTTPError")
    assert rows[-1]["outcome"] == "error"


@pytest.mark.parametrize("error_code,retry,failures,salvage", [
    ("rate_limit_exceeded", True, 3, False), ("server_error", True, 1, False),
    ("server_error", False, 1, False), ("rate_limit_exceeded", False, 1, True),
    ("server_error", True, 78, False),
])
def test_terminal_background_failure_is_priced_once(tmp_path, monkeypatch, error_code, retry, failures, salvage):
    from mathagents import api_client as api
    monkeypatch.setattr("mathagents.provider_trace.time.time", lambda: 100.0)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(APIClient, "_sleep", lambda *args: None)
    monkeypatch.setattr(api.time, "sleep", lambda *args: None)
    monkeypatch.setattr(api.request_logger, "log_request", lambda **kw: None)
    monkeypatch.setattr(api.request_logger, "log_response", lambda **kw: None)
    client = APIClient(api="openai", model="gpt-6-astra", use_openai_responses_api=True,
                       background=True, max_retries_inner=failures if retry else 0, max_tokens=128000,
                       read_cost=10, write_cost=50, long_context_threshold_tokens=272000,
                       long_context_input_multiplier=2, long_context_output_multiplier=1.5)
    monkeypatch.setattr(client, "_is_background_server_kill", lambda code, elapsed: code == "server_error")
    calls = []
    def sdk_object(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: sdk_object(item) for key, item in value.items()})
        if isinstance(value, list):
            return [sdk_object(item) for item in value]
        return value
    def response(status, identity, **kw):
        data = {"id": identity, "object": "response", "created_at": 0,
                "model": "gpt-6-astra", "status": status, "output": [], "usage": None, "error": None,
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], **kw}
        result = sdk_object(data)
        result.model_dump = lambda: data
        return result
    def create(**kw):
        calls.append(kw)
        return response("queued", f"response-{len(calls)}")
    def retrieve(identity, **kw):
        if int(identity.split("-")[-1]) <= failures:
            output = [{"type": "message", "id": "salvaged", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "y" * 60, "annotations": []}]}] if salvage else []
            return response("failed", identity, output=output,
                            error={"code": error_code, "message": "Synthetic provider failure"})
        return response("completed", identity, output=[{"type": "message", "id": "message", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "Paid review", "annotations": []}]}],
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                   "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 5}})
    transport = SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=retrieve))
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    question = "x" * 300000 if failures == 78 else "Synthetic"
    def query(client, messages):
        result = client._openai_query_responses_api(transport, 0, [{"role": "user", "content": question}])
        return 0, result.conversation, {"cost": result.cost_usd, "input_tokens": result.input_tokens,
                                        "output_tokens": result.output_tokens}
    async def run():
        if retry or salvage:
            _, _, cost = await agent._query(client, [], query, call_id="paid")
            await agent._record_model_usage({"cost_usd": cost["cost"], "in_tokens": cost["input_tokens"],
                                             "out_tokens": cost["output_tokens"]}, call_id="paid")
        else:
            with pytest.raises(ValueError):
                await agent._query(client, [], query, call_id="paid")
        for _ in range(2):
            assert await reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir)
    asyncio.run(run())
    rows = latest_attempts(agent.workdir / "provider-attempts.jsonl")
    assert len(rows) == (failures + 1 if retry else 1)
    assert rows[0]["cost_estimated"] and rows[0]["usage_unavailable"]
    assert not any(row.get("admission_rejected") for row in rows)
    assert rows[0]["estimate_reason"] == ("salvaged_response_text" if salvage else "terminal_response_without_usage")
    failed_cost = failures * ((len(question) // 3) * 10 + (20 if salvage else 0) * 50) / 1_000_000
    assert rows[0]["estimate_output_tokens"] == (20 if salvage else 0)
    assert agent.tracker.counters.usd == pytest.approx(failed_cost + (.0011 if retry else 0))
    assert agent.tracker.counters.tokens == (30 if retry else 23 if salvage else 0)
    assert provider_attempts_settled(rows)


def test_terminal_estimate_is_recovered_locally_after_crash(tmp_path):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    trace = ProviderTrace(agent.workdir / "provider-attempts.jsonl", call_id="crash")
    client = SimpleNamespace(api="openai", model="synthetic", _get_cost=lambda *args: 3)
    trace.request(client, "request", 0, {})
    trace.response(client, "request", 0, {"id": "saved", "status": "failed"})
    assert not provider_attempts_settled(latest_attempts(trace.path))
    for _ in range(2):
        assert asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir))
    assert agent.tracker.counters.usd == 3


def test_known_partial_cost_is_charged_without_claiming_settlement(tmp_path):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    trace = ProviderTrace(agent.workdir / "provider-attempts.jsonl", call_id="partial")
    trace.update(("open", 0), provider="openai", response_id="still-open", status="in_progress",
                 usage_unavailable=True, cost=2)
    for _ in range(2):
        assert not asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir))
    assert agent.tracker.counters.usd == 2
