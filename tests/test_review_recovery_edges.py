"""Offline regressions for PR 14's fifth review; no paid provider calls."""
import asyncio
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.api_client import _estimate_salvaged_usage
from mathagents.provider_trace import ProviderTrace, latest_attempts
from proofstack import subscription as sub
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.council import CouncilMember
from proofstack.agents.final_stateless_review import FinalStatelessCritic, FinalReviewBlocked
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.agents.dag_workflow import DAGWorkflow
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.provider_accounting import provider_attempts_settled
from test_firstproof_batch3 import TEX, pipeline
from test_proof_simplification import simplification
from test_research_critic_context import packet
from test_subscription_workflow import credentials, configured, fake_cli


def test_failed_response_salvage_with_real_sdk_objects():
    # Some legacy tests replace SDK modules globally. A separate interpreter
    # exercises the installed Pydantic serializers rather than those stubs.
    script = textwrap.dedent('''
        import socket
        from types import SimpleNamespace
        from unittest.mock import patch
        from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText
        from openai.types.responses.response_error import ResponseError
        from mathagents.api_client import APIClient
        from mathagents.provider_trace import ProviderTrace, active_trace

        def no_network(*args, **kwargs):
            raise AssertionError("Unexpected network access")
        socket.socket.connect = no_network
        text = ResponseOutputText.model_construct(type="output_text", text="A paid proof.", annotations=[])
        message = ResponseOutputMessage.model_construct(id="message", type="message", status="completed",
                    role="assistant", content=[text])
        failed = Response.model_construct(id="saved", status="failed", output=[message], usage=None,
                    error=ResponseError.model_construct(code="rate_limit_exceeded", message="Synthetic limit"))
        queued = Response.model_construct(id="saved", status="queued", output=[], usage=None)
        creates = []
        def create(**payload):
            creates.append(payload)
            return queued
        transport = SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=lambda *a, **k: failed))
        client = APIClient(api="custom", model="synthetic-pro", use_openai_responses_api=True,
                    background=True, max_tokens=100, read_cost=10, write_cost=50)
        trace = ProviderTrace()
        token = active_trace.set(trace)
        try:
            with patch("mathagents.api_client.request_logger.log_request"), \\
                 patch("mathagents.api_client.request_logger.log_response"), \\
                 patch("mathagents.api_client.time.sleep"):
                result = client._openai_query_responses_api(transport, 0, [{"role":"user", "content":"Prove this."}])
            assert len(creates) == 1, creates
            assert result.conversation[-1]["content"].strip() == "A paid proof."
            row, = trace.attempts.values()
            assert row["cost_estimated"] and row["cost"] > 0
            assert row["input_tokens"] > 0 and row["output_tokens"] > 0
        finally:
            active_trace.reset(token)
    ''')
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("history", ["queued", "in_progress", "legacy"])
def test_zero_cost_rejection_requires_never_running_evidence(tmp_path, history):
    client = SimpleNamespace(api="openai", model="synthetic", _get_cost=lambda i, o, *_: (i + o) / 100)
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(client, "call", 0, {"input": "x" * 300, "max_output_tokens": 200})
    trace.response(client, "call", 0, {"id": "saved", "status": "queued"})
    if history == "in_progress":
        trace.response(client, "call", 0, {"id": "saved", "status": "in_progress"})
        trace.response(client, "call", 0, {"id": "saved", "status": "queued"})
    if history == "legacy":
        trace.attempts[("call", 0)].pop("ever_in_progress")
        trace.update(("call", 0))
    trace = ProviderTrace.restore(trace.path, trace.id)
    key, = trace.attempts
    trace.response(client, *key, {"id": "saved", "status": "failed", "output": [],
                                "error": {"code": "rate_limit_exceeded"}})
    trace.estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    if history == "queued":
        assert row["cost"] == 0 and row["admission_rejected"]
    else:
        assert not row.get("admission_rejected")
        assert row["cost"] == 3 and row["cost_estimated"] and row["usage_unavailable"]
    assert provider_attempts_settled([row])


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_missing_terminal_usage_reserves_unseen_output_after_crash(tmp_path, terminal):
    client = SimpleNamespace(api="openai", model="synthetic", _get_cost=lambda i, o, *_: i + o)
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(client, "call", 0, {"input": "x" * 300, "max_output_tokens": 200})
    trace.response(client, "call", 0, {"id": "saved", "status": "in_progress"})
    trace.response(client, "call", 0, {"id": "saved", "status": terminal, "output": []})
    for _ in range(2):
        ProviderTrace.restore(trace.path, trace.id).estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert row["cost"] == 300 and row["estimate_output_allowance_tokens"] == 200
    assert row["usage_unavailable"] and row["cost_estimated"]
    assert not row.get("reasoning_tokens"), "An allowance is not measured reasoning usage"


def test_request_estimate_includes_replayed_tool_content():
    payload = {"instructions": "a" * 30, "input": [
        {"role": "user", "content": "b" * 30},
        {"type": "function_call_output", "output": "c" * 30},
        {"type": "function_call", "arguments": "d" * 30},
        {"type": "code_interpreter_call", "code": "e" * 30,
         "outputs": [{"type": "logs", "logs": "f" * 30}]},
        {"type": "reasoning", "summary": [{"text": "g" * 30}]},
    ]}
    assert _estimate_salvaged_usage(payload, []) == (70, 0)
    assert _estimate_salvaged_usage(None, []) == (0, 0)


@pytest.mark.parametrize("terminal", ["completed", "cancelled", "in_progress", "no_id"])
def test_final_recovery_polls_known_id_without_new_inference(tmp_path, monkeypatch, terminal):
    operations = []
    class Client:
        model = "synthetic"
        def reconcile_background_response(self, response_id, *, timeout, on_response):
            operations.append(response_id)
            assert 0 < timeout <= 15
            response = {"id": response_id, "status": terminal, "output": []}
            if terminal != "in_progress":
                response["usage"] = {"input_tokens": 100, "output_tokens": 20}
            if terminal == "completed":
                response["output"] = [{"type": "message", "content": [{"type": "output_text",
                                      "text": "Saved review\n<answer_ready>true</answer_ready>"}]}]
            on_response(response)
        def _extract_usage_tokens(self, usage):
            return usage["input_tokens"], usage["output_tokens"], 0, 0
        def _extract_reasoning_tokens(self, usage):
            return 0
        def _get_cost(self, *args):
            return 2
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_usd=1, max_wallclock_s=60),
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda _: Client())
    critic = FinalStatelessCritic(ctx)
    inp = packet(mode="fresh", prior_messages=[], research_notes_tex="")
    folder = tmp_path / "agents" / "old-critic"
    trace = ProviderTrace(folder / "provider-attempts.jsonl", call_id="paid-review")
    if terminal == "no_id":
        trace.update(("call", 0), provider="openai", outcome="error", error_type="TimeoutError",
                     retry_decision="stop", transport_uncertain=True)
    else:
        trace.update(("call", 0), provider="openai", response_id="saved", status="in_progress")
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": "agents/old-critic", "review_input": inp.model_dump(mode="json")}))
    monkeypatch.setattr(critic, "_call_review", AsyncMock(side_effect=AssertionError("No new inference")))
    if terminal in {"in_progress", "no_id"}:
        with pytest.raises(FinalReviewBlocked, match="unresolved"):
            asyncio.run(critic.recover_completed(inp))
    else:
        for _ in range(2):
            result = asyncio.run(critic.recover_completed(inp))
            assert (result is not None and result.answer_ready) is (terminal == "completed")
        assert ctx.budgets.root().counters.usd == 2
    assert operations == ([] if terminal == "no_id" else ["saved"])


def test_council_reconciliation_survives_real_yaml_dag(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    calls = []
    async def author(self, **kwargs):
        calls.append("author")
        return self.Outputs(answer_tex=TEX, ready=True, council_question="Check a lemma")
    async def critic(self, **kwargs):
        return self.parse_output("Synthetic review\n<answer_ready>false</answer_ready>", self.Inputs(**kwargs))
    async def council(self, **kwargs):
        calls.append("council")
        raise sub.SubscriptionBlocked("Synthetic council requires reconciliation")
    monkeypatch.setattr(Author, "__call__", author)
    monkeypatch.setattr(ACCritic, "__call__", critic)
    monkeypatch.setattr(CouncilMember, "__call__", council)
    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", AsyncMock(return_value=(True, 1, "")))
    parent = FirstProofBatch3Workflow(workflow.ctx)
    out = asyncio.run(parent(problem="Synthetic question", problem_id="synthetic",
        execution_backend="subscription", n_rounds=1, enable_compute=False, enable_council=True,
        research_seconds=60, partial_cleanup_seconds=20, max_recovery_attempts=2))
    assert calls.count("author") == 1 and "council" in calls
    assert "requires reconciliation" in out.error and not out.error_retryable
    events = [json.loads(line) for line in (workflow.ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "batch3.recovery_blocked" for e in events)
    assert not any(e["kind"] in {"batch3.recovery", "ac.last_gasp", "batch3.partial_cleanup"} for e in events)
    assert not fake_cli


@pytest.mark.parametrize("kind", ["agent", "map_chain"])
def test_dag_cannot_retry_or_soft_fail_reconciliation(tmp_path, monkeypatch, kind):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        component_configs={"DAGWorkflow": {"dag": {"nodes": [], "outputs": {}}}})
    workflow = DAGWorkflow(ctx)
    caller = AsyncMock(side_effect=sub.SubscriptionBlocked("Unresolved worker"))
    caller.Inputs = ACCritic.Inputs
    caller.component_config = {}
    monkeypatch.setattr(workflow, "_agent_for", lambda *a, **k: caller)
    state = {"input": {"problem": "Synthetic", "problem_id": "p"}}
    node = {"id": "test", "retries": 2, "soft_fail": True, "on_error": "skip_item", "default": {}}
    async def run():
        if kind == "agent":
            await workflow._run_agent_node(node, state)
        else:
            await workflow._run_chain_item({"id": "map", "steps": [node]}, state, {}, 0,
                                           semaphore=asyncio.Semaphore(1))
    with pytest.raises(sub.SubscriptionBlocked, match="Unresolved worker"):
        asyncio.run(run())
    assert caller.await_count == 1


def test_disabled_simplification_keeps_current_document_and_notes(simplification):
    execute, _, _, _, _, agent = simplification
    asyncio.run(execute())
    current = TEX.replace("Proof.", "New accepted proof.")
    inp = agent.Inputs(problem="Synthetic", problem_id="p", simplification_rounds=0)
    result = asyncio.run(agent._simplify_accepted(inp, current, "Current notes", "Current report",
                        agent.Outputs(problem_id="p"), deadline=time.monotonic() + 30, reserve=0))
    assert result == (current, "Current notes", None)


@pytest.mark.parametrize("damage", ["malformed", "mismatched"])
def test_invalid_final_checkpoint_during_recovery_preserves_baseline(simplification, monkeypatch, damage):
    execute, calls, _, _, ctx, agent = simplification
    async def broken_cleanup(ctx, inp, document, notes, **kwargs):
        key, review_input = agent._final_review_identity(inp, document)
        path = agent._final_review_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if damage == "malformed":
            path.write_text("{broken")
        else:
            from proofstack.agents.final_stateless_review import FinalReviewState
            state = FinalReviewState(key="f" * 64, review_input=review_input, max_usd=10,
                                     initial_spend=0, deadline_unix_s=time.time() + 30)
            path.write_text(state.model_dump_json())
        raise TypeError("Synthetic editor failure")
    monkeypatch.setattr(agent, "_candidate_loop", broken_cleanup)
    out = asyncio.run(execute(final_stateless_review=True))
    assert out.accepted_baseline_pending and out.accepted_baseline_path.read_text() == TEX
    assert not out.submission_approved and not out.error_retryable
    assert "Synthetic editor failure" in out.error and "Retained manuscript recovery" in out.error
    assert calls["final_critic"] == 0
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "batch3.recovery_blocked" for e in events)


def test_resume_after_cutoff_restores_ready_simplification(simplification, monkeypatch):
    execute, calls, seen, _, ctx, agent = simplification
    class Crash(BaseException):
        pass
    retain = agent._retain_simplified
    def crash(state):
        if state.completed == 1 and state.stage == "ready":
            raise Crash()
        retain(state)
    with monkeypatch.context() as patch:
        patch.setattr(agent, "_retain_simplified", crash)
        with pytest.raises(Crash):
            asyncio.run(execute(simplification_rounds=3))
    assert agent._accepted_baseline.document == TEX
    schedule = ctx.root_workdir / "batch3-schedule.json"
    saved = json.loads(schedule.read_text())
    saved["research_deadline_unix_s"] = time.time() - 1
    schedule.write_text(json.dumps(saved))
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert calls["research"] == calls["simplify_author"] == calls["simplify_critic"] == 1
    assert calls["rewrite"] == 0 and calls["final_critic"] == 1
    assert out.submission_approved and "Simpler proof 1" in out.answer_tex.read_text()
    assert "Simpler proof 1" in seen["final_inputs"][0].answer_tex
