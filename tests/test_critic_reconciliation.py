"""Synthetic restart receipts and SDK operations; no live provider calls."""
import json
from types import SimpleNamespace

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace, latest_attempts
from proofstack.agents.ac.ac_workflow import ACWorkflow, _sum_logged_model_cost
from proofstack.agents.ac.critic import ACCritic
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext
from proofstack.kinds.api_call import APICallAgent
from test_research_critic_context import packet, runner


@pytest.mark.parametrize("provider,recovered", [("anthropic", False), ("google", False), ("google", True)])
@pytest.mark.parametrize("fault", [None, "unfinished", "missing_usage", "pending"])
def test_inline_provider_receipts_resume_without_openai_reconciliation(tmp_path, monkeypatch, runner, provider, recovered, fault):
    calls = []
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda _: pytest.fail("Completed non-OpenAI work must not create a reconciliation client"))
    client = SimpleNamespace(api=provider, model="synthetic", _get_cost=lambda *args: .25,
        _extract_usage_tokens=lambda u: (u["input_tokens"], u["output_tokens"], 0, 0),
        _extract_reasoning_tokens=lambda u: 0)

    async def reply(self, inp):
        calls.append(inp.mode)
        trace = ProviderTrace(self.workdir / "provider-attempts.jsonl", call_id="paid-call")
        if recovered:
            trace.request(client, "tool-limit", 0, {})
            response = {"candidates": [{"finishReason": "TOO_MANY_TOOL_CALLS"}]}
            if fault != "missing_usage":
                response["usageMetadata"] = {"promptTokenCount": 10, "candidatesTokenCount": 20}
            trace.response(client, "tool-limit", 0, response)
            if fault == "pending":
                trace.update(("tool-limit", 0), reconciliation_pending=True)
        trace.request(client, "request", 0, {})
        if provider == "anthropic":
            response = {"id": "msg-synthetic", "stop_reason": None if fault == "unfinished" else "end_turn"}
            if fault != "missing_usage":
                response["usage"] = {"input_tokens": 10, "output_tokens": 20}
        else:
            response = {"candidates": [{"finishReason": "FINISH_REASON_UNSPECIFIED" if fault == "unfinished" else "STOP"}]}
            if fault != "missing_usage" or recovered:
                response["usageMetadata"] = {"promptTokenCount": 10, "candidatesTokenCount": 20}
        trace.response(client, "request", 0, response)
        if fault == "pending" and not recovered:
            trace.update(("request", 0), reconciliation_pending=True)
        cost = trace.totals()["cost"]
        if cost:
            self.tracker.add_usd(cost)
            await self.events.emit("model.call", {"cost_usd": cost}, call_id="paid-call")
        return self.parse_output("Paid review\n<answer_ready>true</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    inp = packet().model_dump()
    first = runner.run(ACCritic(ctx)(**inp))
    assert first.answer_ready
    for _ in range(2):
        if fault:
            with pytest.raises(RuntimeError, match="unresolved provider work"):
                runner.run(ACCritic(ctx)(**inp))
        else:
            assert runner.run(ACCritic(ctx)(**inp)) == first
            assert ctx.budgets.root().counters.usd == (.5 if recovered else .25)
    assert calls == ["stateful"]


def test_resume_touches_saved_explicit_container_without_uploading(tmp_path, monkeypatch, runner):
    operations = []
    class Client:
        terminated = False
        model = "synthetic"
        def touch_code_interpreter_container(self, container_id, *, timeout):
            operations.append(("touch", container_id))
            assert 0 < timeout <= 5
            return "running"
        def reconcile_background_response(self, response_id, *, timeout, on_response):
            operations.append(("reconcile", response_id))
            on_response({"id": response_id, "status": "completed", "usage": {"input_tokens": 10},
                         "output": [{"type": "message", "content": [{"type": "output_text",
                            "text": "Paid review\n<answer_ready>false</answer_ready>"}]}]})
        def _extract_usage_tokens(self, usage):
            return 10, 20, 0, 0
        def _extract_reasoning_tokens(self, usage):
            return 0
        def _get_cost(self, *args):
            return .25

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda _: Client())
    critic, inp = ACCritic(ctx), packet()
    trace = ProviderTrace(tmp_path / "agents/critic/provider-attempts.jsonl", call_id="paid-call")
    trace.update(("request", 0), provider="openai", response_id="saved-response", status="in_progress",
                 reconciliation_pending=True, usage_unavailable=True)
    (trace.path.parent / "research-notes-attachment.json").write_text(json.dumps({
        "container_mode": "explicit", "container_id": "saved-container"}))
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps({"status": "interrupted", "attempts": 1, "attempt_workdir": "agents/critic"}))
    async def no_call(*args, **kwargs):
        pytest.fail("Must reuse paid report without uploading or calling a model")
    monkeypatch.setattr(ACCritic, "_call_review", no_call)
    out = runner.run(critic(**inp.model_dump()))
    assert not out.answer_ready
    assert operations == [("touch", "saved-container"), ("reconcile", "saved-response")]
    assert ctx.budgets.root().counters.usd == .25


@pytest.mark.parametrize("terminal", ["completed", "cancelled", "failed", "unresolved"])
@pytest.mark.parametrize("saved_report", [False, True])
@pytest.mark.parametrize("missing_usage", [False, True])
def test_resume_reconciles_all_attempts_before_report_or_retry(tmp_path, monkeypatch, runner, terminal, saved_report, missing_usage):
    calls, reconciled = [], []

    class Client:
        model = "synthetic"
        terminated = False

        def reconcile_background_response(self, response_id, *, timeout, on_response):
            reconciled.append(response_id)
            if terminal == "unresolved":
                return {}
            on_response({"id": response_id, "status": terminal,
                         "usage": None if missing_usage else {"input_tokens": 10, "output_tokens": 20},
                         "output": [{"type": "message", "content": [{"type": "output_text",
                                    "text": "Retained report\n<answer_ready>false</answer_ready>"}]}]
                         if terminal == "completed" else []})

        def _extract_usage_tokens(self, usage):
            return usage["input_tokens"], usage["output_tokens"], 0, 0

        def _extract_reasoning_tokens(self, usage):
            return 5

        def _get_cost(self, *args):
            return 1.5

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda cfg: Client(),
                            run_budget=BudgetSpec(max_usd=50, max_wallclock_s=60),
                            component_configs={"ACCritic": {"research_notes_transport": "inline"}})

    async def reply(self, inp):
        calls.append(inp.mode)
        if len(calls) == 1:
            raise ValueError("context_length_exceeded")
        if len(calls) == 2:
            rows = [{"attempt_id": "fresh:old:0", "invocation_id": "fresh", "call_id": "paid-call",
                     "response_id": "saved-response", "status": "in_progress", "outcome": "error",
                     "reconciliation_pending": True, "usage_unavailable": True, "cost": .2}]
            if saved_report:
                rows.append({"attempt_id": "fresh:new:0", "invocation_id": "fresh", "call_id": "paid-call",
                             "response_id": "already-completed", "status": "completed", "outcome": "response",
                             "usage_unavailable": False, "cost": .5})
                (self.workdir / "provider-completed-fresh.json").write_text(json.dumps({
                    "report": "Paid report\n<answer_ready>false</answer_ready>"}))
            (self.workdir / "provider-attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.tracker.add_usd(.2)
            await self.events.emit("model.call", {"cost_usd": .2}, call_id="paid-call")
            raise RuntimeError("simulated crash")
        return self.parse_output("Retry report\n<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(RuntimeError, match="crash"):
        runner.run(ACCritic(ctx)(**packet().model_dump()))
    if saved_report:
        critic = ACCritic(ctx)
        ctx.resume_cache.put(critic._cache_key(packet()), critic.Outputs(review_md="Legacy cached report").model_dump())
    if terminal == "unresolved":
        with pytest.raises(RuntimeError, match="unresolved"):
            runner.run(ACCritic(ctx)(**packet().model_dump()))
        assert calls == ["stateful", "fresh"]
    else:
        result = runner.run(ACCritic(ctx)(**packet().model_dump()))
        assert not result.answer_ready
        assert len(calls) == (3 if terminal in {"cancelled", "failed"} and not saved_report else 2)
        assert ctx.budgets.root().counters.usd == pytest.approx(1.5 + (.5 if saved_report else 0))
        before = ctx.budgets.root().counters.usd
        assert runner.run(ACCritic(ctx)(**packet().model_dump())) == result
        assert ctx.budgets.root().counters.usd == before
        rows = latest_attempts(next(tmp_path.glob("agents/*/provider-attempts.jsonl")))
        assert not rows[0]["reconciliation_pending"]
        assert rows[0]["usage_unavailable"] == missing_usage
        assert rows[0].get("cost_estimated", False) == missing_usage
        if missing_usage:
            assert rows[0]["estimate_input_tokens"] == 1_050_000
            assert rows[0]["estimate_output_tokens"] == 128_000
    assert reconciled == ["saved-response"]


@pytest.mark.parametrize("outer_resume", [False, True])
def test_local_reconciliation_never_accounts_for_a_parallel_sibling(tmp_path, runner, outer_resume):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            api_client_factory=lambda cfg: pytest.fail("already settled"))
    critic = ACCritic(ctx)
    trace = ProviderTrace(tmp_path / "agents/critic/provider-attempts.jsonl", call_id="critic-call")
    trace.update(("request", 0), response_id="saved", status="completed", cost=2,
                 usage_unavailable=False, input_tokens=10, output_tokens=20, reasoning_tokens=5)

    async def exercise():
        if outer_resume:
            workflow = ACWorkflow(ctx)
            await workflow._apply_resume_budget_offset()
            assert ctx.budgets.root().counters.usd == 2
        # Council has logged its call but has not charged the shared tracker yet.
        await critic.events.emit("model.call", {"cost_usd": 4}, call_id="council-call")
        rows = latest_attempts(trace.path)
        await critic._reconcile_recovery(trace.path, rows)
        critic.tracker.add_usd(4)
        assert ctx.budgets.root().counters.usd == 6
        await critic._reconcile_recovery(trace.path, rows)
        assert ctx.budgets.root().counters.usd == 6
        assert _sum_logged_model_cost(tmp_path / "events.jsonl") == 6

    runner.run(exercise())


def test_terminal_missing_usage_estimate_can_exhaust_budget_before_retry(tmp_path, monkeypatch, runner):
    class Client:
        model = "synthetic"

        def reconcile_background_response(self, response_id, *, timeout, on_response):
            on_response({"id": response_id, "status": "cancelled", "usage": None})

        def _get_cost(self, *args):
            return 3.0

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda cfg: Client(),
                            run_budget=BudgetSpec(max_usd=2),
                            component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    critic, inp = ACCritic(ctx), packet()
    trace = ProviderTrace(tmp_path / "agents/critic/provider-attempts.jsonl", call_id="saved-call")
    trace.update(("request", 0), response_id="saved", status="cancelled", usage_unavailable=True)
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps({"status": "interrupted", "attempts": 1,
                                     "attempt_workdir": "agents/critic"}))
    async def no_retry(*args):
        pytest.fail("estimated debit must be checked before another model call")
    monkeypatch.setattr(APICallAgent, "run", no_retry)
    for _ in range(2):
        with pytest.raises(BudgetExhausted):
            runner.run(critic(**inp.model_dump()))
        assert ctx.budgets.root().counters.usd == 3
        assert _sum_logged_model_cost(tmp_path / "events.jsonl") == 3


@pytest.mark.parametrize("evidence", [[], [{"type": "code_interpreter_call", "outputs": None}]])
def test_completed_paid_receipt_retrieves_missing_tool_evidence(tmp_path, runner, evidence):
    inp = packet()
    operations = []

    class Client:
        model = "synthetic"

        def reconcile_background_response(self, response_id, *, timeout, on_response):
            operations.append(response_id)
            metadata = critic._notes_metadata(inp)
            on_response({"id": response_id, "status": "completed", "usage": {"input_tokens": 10},
                         "output": [{"type": "code_interpreter_call", "status": "completed",
                                     "container_id": "container", "code": critic._notes_verification_code(inp),
                                     "outputs": [{"type": "logs", "logs": json.dumps({k: metadata[k] for k in
                                                                                       ("filename", "bytes", "sha256")})}]},
                                    {"type": "message", "content": [{"type": "output_text", "text":
                                     "<research_notes_status>verified</research_notes_status><answer_ready>true</answer_ready>"}]}]})

        def _extract_usage_tokens(self, usage):
            return 10, 20, 0, 0

        def _extract_reasoning_tokens(self, usage):
            return 5

        def _get_cost(self, *args):
            return 2.0

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda cfg: Client())
    critic = ACCritic(ctx)
    trace = ProviderTrace(tmp_path / "agents/critic/provider-attempts.jsonl", call_id="saved-call")
    trace.update(("request", 0), response_id="saved", status="completed", usage_unavailable=False,
                 cost=2, code_interpreter_calls=evidence)
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps({"status": "interrupted", "attempts": 1,
                                     "attempt_workdir": "agents/critic"}))
    out = runner.run(critic(**inp.model_dump()))
    assert out.answer_ready and out.research_notes_execution_verified
    assert runner.run(critic(**inp.model_dump())) == out
    assert operations == ["saved"] and ctx.budgets.root().counters.usd == 2


@pytest.mark.parametrize("completed", [False, True])
def test_sdk_reconciliation_only_retrieves_or_cancels(monkeypatch, completed):
    operations = []

    class SDK:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.responses = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def retrieve(self, response_id, **kwargs):
            operations.append("retrieve")
            status = "completed" if completed else "in_progress"
            return SimpleNamespace(model_dump=lambda: {"id": response_id, "status": status,
                                   "usage": {"input_tokens": 1} if completed else None})

        def cancel(self, response_id, **kwargs):
            operations.append("cancel")
            return SimpleNamespace(model_dump=lambda: {"id": response_id, "status": "cancelled",
                                   "usage": {"input_tokens": 1}})

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setattr("mathagents.api_client.OpenAI", SDK)
    client = APIClient(model="synthetic", use_openai_responses_api=True)
    records = []
    result = client.reconcile_background_response("saved-id", on_response=records.append)
    assert result["status"] == ("completed" if completed else "cancelled")
    assert operations == (["retrieve"] if completed else ["retrieve", "retrieve", "cancel"])
    assert records[-1] == result
