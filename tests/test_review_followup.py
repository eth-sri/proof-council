"""Offline regressions for PR 14's fourth review; no live provider calls."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace, active_trace, latest_attempts
from proofstack import subscription as sub
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.compute import Compute
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.visual_blocks import (
    ACInitBlock, ACAuthorBlock, ACCouncilBlock, ACComputeBlock,
    ACStatefulCriticBlock, ACFreshCriticBlock,
)
from proofstack.agents.cleanup_session import CleanupSession
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.budget import BudgetSpec, BudgetExhausted
from proofstack.context import RunContext
from proofstack.provider_accounting import provider_attempts_settled, reconcile_provider_call
from test_subscription_workflow import credentials, configured, fake_cli, TEX
from test_firstproof_batch3 import pipeline
from test_proof_simplification import simplification


@pytest.mark.parametrize("status,body,decision,code", [
    (400, "Input token count exceeds the maximum", "stop", "context_length_exceeded"),
    (403, '{"error":{"status":"PERMISSION_DENIED"}}', "stop", None),
    (404, "Model does not exist", "stop", None),
    (502, "<html>temporary gateway failure</html>", "retry", None),
    (504, "<html>temporary gateway timeout</html>", "retry", None),
])
def test_google_error_classification(tmp_path, monkeypatch, status, body, decision, code):
    import requests
    from mathagents import api_client as api
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic")
    response = requests.Response()
    response.status_code, response._content = status, body.encode()
    monkeypatch.setattr(api.requests, "post", lambda *a, **kw: response)
    monkeypatch.setattr(api.request_logger, "log_request", lambda **kw: None)
    client = APIClient(api="google", model="synthetic", max_tokens=10)
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)
    try:
        with pytest.raises(requests.HTTPError) as caught:
            client._google_query_with_internal_tools(0, [{"role": "user", "content": "Synthetic"}])
    finally:
        active_trace.reset(token)
    assert body in str(caught.value) and caught.value.status_code == status
    [row] = latest_attempts(trace.path)
    assert row["retry_decision"] == decision and row["error_code"] == code
    assert provider_attempts_settled([row]) is (status != 504)


@pytest.mark.parametrize("entry", ["safe_wrapper", "dag"])
def test_real_council_reconciliation_error_propagates(configured, fake_cli, monkeypatch, entry):
    workflow, inp = configured
    command = ConfigurableCLIAgent._command_for
    def no_usage(self, packet):
        cmd = command(self, packet)
        cmd[-1] = "import sys; sys.stdin.read(); sys.exit(1)"
        return cmd
    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", no_usage)
    async def run():
        if entry == "safe_wrapper":
            return await workflow._safe_council(round=1, author_question="Synthetic question",
                answer_tex=TEX, research_notes_tex="", references_bib="", member_models=inp.council_models[:1])
        block = ACCouncilBlock(workflow.ctx)
        return await block(state={"inputs": inp.model_dump(), "run_council": True,
            "current_author": Author.Outputs(answer_tex=TEX, council_question="Synthetic question",
                                             council_to=inp.council_models[:1]).model_dump()})
    with pytest.raises(sub.SubscriptionBlocked):
        asyncio.run(run())
    with pytest.raises(sub.SubscriptionBlocked):
        sub.check_pending_calls(workflow.ctx.root_workdir)


def test_cleanup_blocks_pending_subscription_before_preflight(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    receipts = workflow.ctx.root_workdir / "subscription_calls"
    receipts.mkdir()
    (receipts / ("a" * 64 + ".json")).write_text('{"status":"started","attempts":1}')
    preflight = AsyncMock(side_effect=AssertionError("Must not reach cleanup preflight"))
    monkeypatch.setattr(CleanupSession, "_preflight", preflight)
    with pytest.raises(sub.SubscriptionBlocked):
        asyncio.run(CleanupSession(workflow.ctx).run(SimpleNamespace()))
    preflight.assert_not_awaited()
    assert not fake_cli


@pytest.mark.parametrize("mode", ["stateful", "fresh"])
def test_dag_shares_shutdown_allowance_with_both_helpers(configured, fake_cli, monkeypatch, mode):
    workflow, inp = configured
    workflow.ctx.budgets.root().spec = BudgetSpec(max_usd=12, max_wallclock_s=120)
    seen = {}
    async def author(self, packet):
        return Author.Outputs(answer_tex=TEX, ready=True, council_question="Suggest a lemma",
                              compute_instructions="Check 1+1", artifact_status="ok")
    async def critic(self, packet):
        seen["critic"] = sub.call_allowance(self.tracker)
        return ACCritic.Outputs(answer_ready=False, review_md="Synthetic", mode=packet.mode)
    async def compute(self, packet):
        seen["compute"] = sub.call_allowance(self.tracker)
        return Compute.Outputs(status="done", response_md="2")
    command = ConfigurableCLIAgent._command_for
    def member(self, packet):
        seen.setdefault("members", []).append(self.component_config["subscription_call_allowance_usd"])
        return command(self, packet)
    run_role = sub.run_model_role
    async def model_role(agent, packet):
        if isinstance(agent, Author):
            return await author(agent, packet)
        if isinstance(agent, ACCritic):
            return await critic(agent, packet)
        return await run_role(agent, packet)
    monkeypatch.setattr(sub, "run_model_role", model_role)
    monkeypatch.setattr(Compute, "run", compute)
    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", member)
    async def run():
        state = (await ACInitBlock(workflow.ctx)(**inp.model_dump())).state
        state = (await ACAuthorBlock(workflow.ctx)(state=state)).state
        assert state["run_compute"] and state["run_council"]
        assert state["review_nominal_allowance"] == 12
        state["critic_mode"] = mode
        critic_cls = ACStatefulCriticBlock if mode == "stateful" else ACFreshCriticBlock
        await asyncio.gather(critic_cls(workflow.ctx)(state=state),
                            ACCouncilBlock(workflow.ctx)(state=state), ACComputeBlock(workflow.ctx)(state=state))
    asyncio.run(run())
    assert seen["critic"] == seen["compute"] == 4
    assert seen["members"] == [2, 2]
    assert workflow.tracker.counters.usd == 0


@pytest.mark.parametrize("over_budget", [False, True])
def test_simplification_rejection_handed_off_once(simplification, over_budget):
    execute, _, seen, options, ctx, _ = simplification
    options.update(simplify_verdicts=["research"], stop_accepting_after_baseline_flaw=True)
    if over_budget:
        options["simplify_review_error"] = BudgetExhausted("simplification", "usd", 1, 2)
    asyncio.run(execute(simplification_rounds=1))
    state = seen["resumed_states"][0]
    rejections = [r for r in state["review_history"] if "Prior acceptance is revoked" in r["review_md"]]
    assert len(rejections) == 1
    saved = json.loads((ctx.root_workdir / "batch3-accepted-baseline.json").read_text())
    assert saved["revoked"] and saved["source_sha256"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete", "completed"])
def test_terminal_text_estimate_and_local_replay(tmp_path, monkeypatch, status):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    client = APIClient(api="openai", model="gpt-6-astra", max_tokens=128000,
        read_cost=10, write_cost=50, long_context_threshold_tokens=272000,
        long_context_input_multiplier=2, long_context_output_multiplier=1.5)
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    trace = ProviderTrace(agent.workdir / "provider-attempts.jsonl", call_id="paid")
    trace.request(client, "synthetic", 0, {"input": [{"role": "user", "content": "x" * 300}]})
    trace.response(client, "synthetic", 0, {"id": "paid", "status": status, "usage": None,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "y" * 60}]}]})
    for _ in range(2):
        assert asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir))
    assert agent.tracker.counters.usd == pytest.approx(.002)
    assert agent.tracker.counters.tokens == 0  # estimates are not reported usage
    [row] = latest_attempts(trace.path)
    assert row["usage_unavailable"] and row["cost_estimated"]


@pytest.mark.parametrize("finish,usage", [("stop", True), ("length", True), ("tool_calls", True),
                                          (None, True), ("stop", False)])
@pytest.mark.parametrize("provider", ["openai", "openrouter", "together"])
def test_chat_completion_settlement(tmp_path, monkeypatch, finish, usage, provider):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "synthetic")
    client = APIClient(api=provider, model="synthetic", use_openai_responses_api=False)
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    trace.request(client, "chat", 0, {})
    trace.response(client, "chat", 0, {"id": "chatcmpl-synthetic", "choices": [
        {"finish_reason": finish, "message": {"role": "assistant", "content": "Review"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20} if usage else None})
    assert provider_attempts_settled(latest_attempts(trace.path)) is (finish is not None and usage)


def test_no_usage_legacy_capacity_estimate_is_not_applied_by_default(tmp_path):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    trace = ProviderTrace(agent.workdir / "provider-attempts.jsonl", call_id="legacy")
    trace.update(("call", 0), provider="openai", response_id="legacy", status="failed",
                 terminal_usage_estimate_usd=30.6, estimate_input_tokens=1_050_000,
                 estimate_output_tokens=128000)
    assert not asyncio.run(reconcile_provider_call(agent.ctx, agent.tracker, agent.workdir))
    assert agent.tracker.counters.usd == 0


def test_empty_rate_limit_does_not_erase_previously_reported_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    client = APIClient(api="openai", model="synthetic", read_cost=10, write_cost=50)
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    trace.request(client, "call", 0, {"input": [{"content": "Synthetic"}]})
    trace.response(client, "call", 0, {"id": "paid", "status": "in_progress", "usage_partial": True,
        "usage": {"input_tokens": 10, "output_tokens": 20}})
    trace.response(client, "call", 0, {"id": "paid", "status": "failed", "usage": None, "output": [],
        "error": {"code": "rate_limit_exceeded"}})
    trace.estimate_terminal_usage()
    [row] = latest_attempts(trace.path)
    assert row["cost"] == pytest.approx(.0011)
    assert not row.get("admission_rejected") and row["cost_estimated"]


@pytest.mark.parametrize("streaming", [False, True])
def test_real_chat_client_records_terminal_receipt(tmp_path, monkeypatch, streaming):
    from mathagents import api_client as api
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setattr(api.request_logger, "log_request", lambda **kw: None)
    monkeypatch.setattr(api.request_logger, "log_response", lambda **kw: None)
    client = APIClient(api="openai", model="synthetic", max_tokens=100,
                       use_openai_responses_api=False, stream_openai_chat_completions=streaming)
    usage_data = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    usage = SimpleNamespace(**usage_data, model_dump=lambda: usage_data)
    message_data = {"role": "assistant", "content": "Review", "tool_calls": None}
    message = SimpleNamespace(**message_data, model_dump=lambda: message_data)
    response = SimpleNamespace(id="chatcmpl-synthetic", usage=usage,
        choices=[SimpleNamespace(message=message, finish_reason="stop")], model_dump=lambda: {
            "id": "chatcmpl-synthetic", "usage": usage_data,
            "choices": [{"message": message_data, "finish_reason": "stop"}]})
    if streaming:
        response = iter([SimpleNamespace(id="chatcmpl-synthetic", usage=None, choices=[
            SimpleNamespace(index=0, delta=SimpleNamespace(content="Review"), finish_reason="stop")]),
            SimpleNamespace(id="chatcmpl-synthetic", usage=usage, choices=[])])
    transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)
    try:
        result = client._openai_query_chat_completions_api(transport, 0, [{"role": "user", "content": "Question"}])
    finally:
        active_trace.reset(token)
    assert result.input_tokens == 10 and result.output_tokens == 20
    assert provider_attempts_settled(latest_attempts(trace.path))


def test_cleanup_failure_keeps_pending_proof_when_final_gate_unavailable(simplification):
    from proofstack.agents.batch3_critic import CleanupContextTooLarge
    execute, calls, _, options, ctx, _ = simplification
    options.update(rewrite_error=CleanupContextTooLarge("Synthetic overflow"),
                   final_error=TimeoutError("Synthetic final gate timeout"))
    out = asyncio.run(execute(simplification_rounds=0, final_stateless_review=True))
    assert out.accepted_baseline_pending and not out.submission_approved
    assert out.accepted_baseline_path.exists() and calls["final_critic"] == 1
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "batch3.recovery_blocked" for e in events)
