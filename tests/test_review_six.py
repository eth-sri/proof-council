"""Offline regression coverage for the sixth PR review."""
import asyncio
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace, latest_attempts
from proofstack.agents.ac.ac_workflow import ACWorkflow, _CompileResult
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.cleanup_session import CleanupSettings
from proofstack.agents.dag_workflow import DAGWorkflow
from proofstack.agents.final_stateless_review import FinalStatelessCritic, FinalReviewBlocked, FinalReviewState
from proofstack.agents.proof_simplification import SimplificationCritic
from proofstack.budget import BudgetSpec, BudgetExhausted
from proofstack.context import RunContext
from proofstack.provider_accounting import provider_attempts_settled
from proofstack.registry import load_preset
from proofstack.subscription import SubscriptionBlocked
from test_cleanup_session import editor, TEX as CLEANUP_TEX
from test_firstproof_batch3 import TEX, ORIGINAL, pipeline
from test_proof_simplification import simplification
from test_research_critic_context import packet


@pytest.fixture
def pricing_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    return APIClient(api="openai", model="synthetic", use_openai_responses_api=True,
                     max_tokens=128000, read_cost=10, write_cost=50,
                     long_context_threshold_tokens=272000,
                     long_context_input_multiplier=2, long_context_output_multiplier=1.5)


@pytest.mark.parametrize("code,elapsed,characters,expected_output,expected_cost", [
    ("server_error", 0, 300000, 0, 1),
    ("server_error", 1, 300000, 50, 1.0025),
    ("rate_limit_exceeded", 1, 300000, 50, 1.0025),
    ("context_length_exceeded", 30, 3300000, 0, 0),
    ("invalid_file", 30, 300000, 0, 0),
    ("file_not_found", 30, 300000, 0, 0),
    ("array_above_max_length", 30, 300000, 0, 0),
    ("server_error", 3300, 300000, 128000, 7.4),
])
def test_terminal_estimates_use_elapsed_time_not_the_full_allowance(
        tmp_path, monkeypatch, pricing_client, code, elapsed, characters, expected_output, expected_cost):
    clock = [100.0]
    monkeypatch.setattr("mathagents.provider_trace.time.time", lambda: clock[0])
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(pricing_client, "call", 0, {"input": "x" * characters, "max_output_tokens": 128000})
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "queued"})
    clock[0] += elapsed
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "failed", "output": [],
                                            "error": {"code": code}})
    for _ in range(2):
        clock[0] += 3600
        ProviderTrace.restore(trace.path, trace.id).estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert row["cost"] == pytest.approx(expected_cost)
    assert row["estimate_output_tokens"] == expected_output
    assert row["estimate_elapsed_s"] == elapsed
    assert row["cost_estimated"] and row["usage_unavailable"]
    assert not row.get("output_tokens"), "Estimates must not masquerade as reported token usage"


@pytest.mark.parametrize("restore", [False, True])
def test_execution_between_polls_or_across_crash_is_not_assumed_free(tmp_path, monkeypatch, pricing_client, restore):
    clock = [100.0]
    monkeypatch.setattr("mathagents.provider_trace.time.time", lambda: clock[0])
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(pricing_client, "call", 0, {"input": "x" * 300000, "max_output_tokens": 128000})
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "queued"})
    if restore:
        trace = ProviderTrace.restore(trace.path, trace.id)
    clock[0] += 60
    key, = trace.attempts
    trace.response(pricing_client, *key, {"id": "saved", "status": "failed", "error": {"code": "rate_limit_exceeded"}})
    trace.estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert row["cost"] == pytest.approx(1.15)
    assert row["usage_unavailable"] and not row.get("admission_rejected")
    assert provider_attempts_settled([row])


def test_legacy_terminal_receipt_without_timing_remains_unresolved(tmp_path, pricing_client):
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.update(("old", 0), provider="openai", response_id="saved", status="queued",
                 ever_in_progress=False, terminal_estimate_source="request_text",
                 terminal_usage_estimate_usd=7.4, estimate_input_tokens=100000,
                 estimate_output_allowance_tokens=128000)
    trace.response(pricing_client, "old", 0, {"id": "saved", "status": "failed", "error": {"code": "rate_limit_exceeded"}})
    trace.estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert not provider_attempts_settled([row])
    assert not row.get("cost_estimated") and not row.get("admission_rejected")


@pytest.mark.parametrize("later", ["cancelled", "accept", "reject"])
def test_recovery_keeps_earlier_completed_rejection(tmp_path, pricing_client, monkeypatch, later):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        run_budget=BudgetSpec(max_usd=100, max_wallclock_s=10),
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda _: pricing_client)
    critic, inp = FinalStatelessCritic(ctx), packet(mode="fresh", prior_messages=[], research_notes_tex="")
    trace = ProviderTrace(tmp_path / "agents/saved/provider-attempts.jsonl", call_id="paid-review")
    for number in range(2):
        trace.request(pricing_client, str(number), 0, {"input": "Question", "max_output_tokens": 128000})
        trace.response(pricing_client, str(number), 0, {"id": f"saved-{number}", "status": "in_progress"})
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": "agents/saved", "review_input": inp.model_dump(mode="json")}))
    calls = []
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        assert not cancel_pending
        calls.append(response_id)
        verdict = "false" if response_id == "saved-0" or later == "reject" else "true"
        on_response({"id": response_id, "status": "cancelled" if response_id == "saved-1" and later == "cancelled" else "completed",
                     "usage": {"input_tokens": 10, "output_tokens": 20},
                     "output": [{"type": "message", "content": [{"type": "output_text",
                         "text": f"Review {response_id}\n<answer_ready>{verdict}</answer_ready>"}]}]})
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    monkeypatch.setattr(critic, "_call_review", AsyncMock(side_effect=AssertionError("No new opinion")))
    for _ in range(2):
        result = asyncio.run(critic.recover_completed(inp))
        assert not result.answer_ready and "saved-0" in result.review_md
    assert calls == ["saved-0", "saved-1"]
    assert ctx.budgets.root().counters.usd == pytest.approx(.0022)


@pytest.mark.parametrize("outcome", ["completed", "expired", "pending"])
def test_final_recovery_preserves_live_response_until_saved_deadline(tmp_path, pricing_client, monkeypatch, outcome):
    operations = []
    class SDK:
        def __init__(self, **kwargs):
            self.responses = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def retrieve(self, response_id, **kwargs):
            operations.append(("retrieve", response_id))
            done = len(operations) == 2 and outcome == "completed"
            return SimpleNamespace(model_dump=lambda: {"id": response_id,
                "status": "completed" if done else "in_progress",
                "usage": {"input_tokens": 10, "output_tokens": 20} if done else None,
                "output": [{"type": "message", "content": [{"type": "output_text",
                    "text": "Saved approval\n<answer_ready>true</answer_ready>"}]}] if done else []})
        def cancel(self, *args, **kwargs):
            pytest.fail("A recover-only poll must not cancel live work")
        def create(self, *args, **kwargs):
            pytest.fail("Recovery must not start another model call")
    monkeypatch.setattr("mathagents.api_client.OpenAI", SDK)
    sleep = asyncio.sleep
    async def fast_sleep(seconds):
        await sleep(0)
    if outcome == "completed":
        monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.sleep", fast_sleep)
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        run_budget=BudgetSpec(max_usd=100, max_wallclock_s=10),
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda _: pricing_client)
    critic, inp = FinalStatelessCritic(ctx), packet(mode="fresh", prior_messages=[], research_notes_tex="")
    trace = ProviderTrace(tmp_path / "agents/saved/provider-attempts.jsonl", call_id="paid")
    trace.request(pricing_client, "call", 0, {"input": "Question", "max_output_tokens": 128000})
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "in_progress"})
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": "agents/saved", "review_input": inp.model_dump(mode="json")}))
    if outcome != "completed":
        deadline = time.monotonic() + (-1 if outcome == "expired" else .1)
        with pytest.raises(FinalReviewBlocked, match="unresolved"):
            asyncio.run(critic.recover_completed(inp, deadline=deadline))
        assert operations == [("retrieve", "saved")]
        assert latest_attempts(trace.path)[0]["status"] == "in_progress"
        assert ctx.budgets.root().counters.usd == 0
    else:
        result = asyncio.run(critic.recover_completed(inp, deadline=time.monotonic() + 2))
        assert result.answer_ready
        assert operations == [("retrieve", "saved"), ("retrieve", "saved")]
        assert ctx.budgets.root().counters.usd == pytest.approx(.0011)


def test_mathematical_rejection_precedes_an_unverified_acceptance(tmp_path, pricing_client, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        api_client_factory=lambda _: pytest.fail("Completed reports need no new call"))
    critic, inp = FinalStatelessCritic(ctx), packet(mode="fresh", prior_messages=[])
    trace = ProviderTrace(tmp_path / "agents/saved/provider-attempts.jsonl", call_id="paid")
    for number, verdict in enumerate(("true", "false")):
        trace.request(pricing_client, str(number), 0, {"input": "Synthetic"})
        trace.response(pricing_client, str(number), 0, {"id": f"saved-{number}", "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "output": [{"type": "message", "content": [{"type": "output_text",
                "text": f"Review {number}\n<answer_ready>{verdict}</answer_ready>"}]}]})
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": "agents/saved", "review_input": inp.model_dump(mode="json")}))
    monkeypatch.setattr(critic, "_reconcile_recovery", AsyncMock())
    result = asyncio.run(critic.recover_completed(inp))
    assert not result.answer_ready and "Review 1" in result.review_md


@pytest.mark.parametrize("critic_class,parent", [(FinalStatelessCritic, ACCritic), (SimplificationCritic, Batch3CleanupCritic)])
@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("wildcard", [False, True])
def test_specialized_critic_inherits_effective_parent_model(tmp_path, critic_class, parent, specialized, wildcard):
    overrides = {f"{parent.__module__}.{parent.__qualname__}": "parent-model"}
    if wildcard:
        overrides["*"] = "wildcard-model"
    if specialized:
        overrides[f"{critic_class.__module__}.{critic_class.__qualname__}"] = "specialized-model"
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, model_overrides=overrides)
    agent = critic_class(ctx)
    assert agent.ctx.model_for(agent, agent.MODEL) == ("specialized-model" if specialized else "parent-model")


@pytest.mark.parametrize("critic_class,parent", [(FinalStatelessCritic, ACCritic), (SimplificationCritic, Batch3CleanupCritic)])
@pytest.mark.parametrize("specialized", [False, True])
def test_component_model_configuration_is_not_shadowed_by_inherited_default(tmp_path, critic_class, parent, specialized):
    config = {parent.__name__: {"model": "parent-model"}}
    if specialized:
        config[critic_class.__name__] = {"model": "specialized-model"}
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs=config)
    agent = critic_class(ctx)
    assert agent.ctx.model_for(agent, agent.MODEL) == ("specialized-model" if specialized else "parent-model")


@pytest.mark.parametrize("damage", ["malformed", "mismatched"])
def test_tail_checkpoint_error_preserves_accepted_baseline(simplification, damage):
    execute, calls, _, options, ctx, agent = simplification
    options["final_error"] = RuntimeError("Synthetic interruption")
    asyncio.run(execute(final_stateless_review=True))
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", final_stateless_review=True)
    baseline = agent._accepted_baseline.document
    key, review_input = agent._final_review_identity(inp, baseline)
    path = agent._final_review_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    if damage == "malformed":
        path.write_text("{broken")
    else:
        path.write_text(FinalReviewState(key="f" * 64, review_input=review_input, max_usd=10,
                       initial_spend=0, deadline_unix_s=time.time() + 30).model_dump_json())
    agent._final_review_pointer().write_text("{}")
    schedule_path = ctx.root_workdir / "batch3-schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["research_deadline_unix_s"] = time.time() - 1
    schedule_path.write_text(json.dumps(schedule))
    before = dict(calls)
    out = asyncio.run(execute(resume_run=True))
    assert out.accepted_baseline_pending and out.accepted_baseline_path.read_text() == baseline
    assert not out.submission_approved and not out.error_retryable
    assert "Retained manuscript recovery" in out.error
    assert calls == before
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] != "batch3.partial_cleanup"
    assert any(e["kind"] == "batch3.recovery_blocked" for e in events)


@pytest.mark.parametrize("changed", [False, True])
def test_legacy_api_cleanup_identity_only_allows_missing_backend(editor, changed):
    agent, ctx, seen = editor
    first = asyncio.run(agent(problem="P", document=CLEANUP_TEX, baseline=CLEANUP_TEX))
    path = first.workspace.parent / "session.json"
    state = json.loads(path.read_text())
    settings = CleanupSettings(review_retry_reserve_fraction=0).model_dump()
    settings.pop("execution_backend")
    if changed:
        settings["effort"] = "medium"
    identity = {"problem": "P", "baseline": CLEANUP_TEX, "partial": False, "page_limit": 16, "settings": settings}
    state["identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(state))
    if changed:
        with pytest.raises(ValueError, match="different inputs/settings"):
            asyncio.run(agent(problem="P", document=CLEANUP_TEX, baseline=CLEANUP_TEX, findings="Resume"))
        assert len(seen["commands"]) == 1
    else:
        out = asyncio.run(agent(problem="P", document=CLEANUP_TEX, baseline=CLEANUP_TEX, findings="Resume"))
        assert out.session_id == first.session_id and "--resume" in seen["commands"][-1]
        assert ctx.budgets.root().counters.usd == pytest.approx(.5)


@pytest.mark.parametrize("dag", [False, True])
def test_standalone_final_critic_propagates_reconciliation(tmp_path, monkeypatch, dag):
    preset = load_preset("author_critic")
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs=preset.component_configs,
        api_client_factory=lambda _: pytest.fail("No API calls"))
    async def author(self, inp):
        return self.Outputs(answer_tex=TEX, ready=False)
    async def critic(self, inp):
        if inp.omit_author_thinking and inp.mode == "fresh":
            raise SubscriptionBlocked("Unsettled final review")
        return self.parse_output("No proof\n<answer_ready>false</answer_ready>", inp)
    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(ACCritic, "run", critic)
    monkeypatch.setattr(ACWorkflow, "_deterministic_ready", AsyncMock(return_value=(True, [])))
    fixed = _CompileResult(tex=TEX, tex_path=None, pdf_path=None, compiled=True, pages=1)
    monkeypatch.setattr("proofstack.agents.ac.ac_workflow._simple_compile_latex", lambda *a, **k: fixed)
    monkeypatch.setattr("proofstack.agents.ac.visual_blocks._simple_compile_latex", lambda *a, **k: fixed)
    workflow = preset.workflow_cls(ctx) if dag else ACWorkflow(ctx)
    with pytest.raises(SubscriptionBlocked, match="Unsettled final review"):
        asyncio.run(workflow(problem="Synthetic", problem_id="p", n_rounds=1,
                            enable_compute=False, enable_council=False, enable_final_critic=True))
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert not any(e["kind"] in {"ac.last_gasp", "workflow.last_gasp", "ac.final_critic_failed"} for e in events)


def test_dag_budget_fallback_propagates_reconciliation(tmp_path, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        component_configs={"DAGWorkflow": {"dag": {"nodes": [
            {"id": "fallback", "agent": "proofstack.agents.ac.critic.ACCritic",
             "run_on": "budget_exhausted"}], "outputs": {}}}})
    workflow = DAGWorkflow(ctx)
    monkeypatch.setattr(workflow, "_run_nodes", AsyncMock(side_effect=SubscriptionBlocked("Unresolved worker")))
    with pytest.raises(SubscriptionBlocked, match="Unresolved worker"):
        asyncio.run(workflow._run_budget_fallback(None, {"run": {}, "node": {}}, BudgetExhausted("run", "usd", 1, 2)))


@pytest.mark.parametrize("expired_window", [False, True])
def test_final_recovery_uses_real_sdk_without_cancelling_or_recreating(tmp_path, expired_window):
    # Isolate the real SDK from older tests that install module-level stubs.
    script = textwrap.dedent('''
        import asyncio, json, socket, sys, time
        from pathlib import Path
        from unittest.mock import patch
        import httpx
        from openai import OpenAI
        from mathagents.api_client import APIClient
        from mathagents.provider_trace import ProviderTrace
        from proofstack.agents.final_stateless_review import FinalStatelessCritic
        from proofstack.budget import BudgetSpec
        from proofstack.context import RunContext

        def no_network(*args, **kwargs):
            raise AssertionError("Only mock HTTP transport is permitted")
        socket.socket.connect = no_network
        operations = []
        expired_window = sys.argv[2] == "True"
        def handle(request):
            operations.append((request.method, request.url.path))
            assert request.method == "GET" and request.url.path == "/v1/responses/saved", operations
            done = expired_window or len(operations) >= 2
            return httpx.Response(200, json={"id":"saved", "object":"response", "created_at":100,
                "status":"completed" if done else "in_progress", "model":"synthetic",
                "parallel_tool_calls":True, "tool_choice":"auto", "tools":[],
                "usage":{"input_tokens":10, "output_tokens":20, "total_tokens":30} if done else None,
                "output":[{"id":"message", "type":"message", "role":"assistant", "status":"completed",
                    "content":[{"type":"output_text", "annotations":[],
                                "text":"Saved verdict\\n<answer_ready>true</answer_ready>"}]}] if done else []})
        def sdk(**kwargs):
            return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        sleep = asyncio.sleep
        async def fast_sleep(seconds):
            await sleep(0)
        client = APIClient(api="custom", model="synthetic", use_openai_responses_api=True,
                           read_cost=10, write_cost=50)
        client.api, client.api_key = "openai", "synthetic"
        root = Path(sys.argv[1])
        ctx = RunContext.create(root_workdir=root, flat=True, run_budget=BudgetSpec(max_wallclock_s=10),
            api_client_factory=lambda _:client, component_configs={"ACCritic":{"research_notes_transport":"inline"}})
        critic = FinalStatelessCritic(ctx)
        inp = critic.Inputs(problem="Synthetic", answer_tex="Synthetic proof", mode="fresh")
        trace = ProviderTrace(root / "agents/saved/provider-attempts.jsonl", call_id="paid")
        trace.request(client, "request", 0, {"input":"Synthetic", "max_output_tokens":128000})
        trace.response(client, "request", 0, {"id":"saved", "status":"in_progress"})
        checkpoint = critic._recovery_checkpoint(inp)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text(json.dumps({"status":"recovery_started", "kind":"ordinary",
            "attempt_workdir":"agents/saved", "review_input":inp.model_dump(mode="json")}))
        with patch("mathagents.api_client.OpenAI", sdk), patch("proofstack.agents.ac.critic.asyncio.sleep", fast_sleep):
            for _ in range(2):
                result = asyncio.run(critic.recover_completed(inp,
                    deadline=time.monotonic()+(-3600 if expired_window else 5), cutoff=time.monotonic()+5))
                assert result.answer_ready
        assert operations == [("GET", "/v1/responses/saved")] * (1 if expired_window else 2), operations
        assert abs(ctx.budgets.root().counters.usd - .0011) < 1e-10
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(expired_window)], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
