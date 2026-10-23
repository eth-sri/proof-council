from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.execution_policy import subscription_only
from proofstack import subscription as sub
from proofstack.agents.ac.ac_workflow import ACWorkflow
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.council import Council
from proofstack.agents.ac.compute import Compute
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.final_stateless_review import FinalStatelessCritic
from proofstack.agents.proof_simplification import SimplificationCritic
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.agents.cleanup_session import CleanupSession, _MeteredTurn
from proofstack.budget import BudgetSpec, BudgetTracker, BudgetExhausted
from proofstack.context import RunContext
from proofstack.kinds.cli import CLIDoneRecord

TEX = r"\documentclass{article}\begin{document}A synthetic proof.\end{document}"
REPLY = "Independent review.\n<ready>true</ready>\n<answer_ready>true</answer_ready>\n<cleanup_verdict>accept</cleanup_verdict>"


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    codex = tmp_path / "login-codex"
    claude = tmp_path / "login-claude"
    codex.mkdir()
    claude.mkdir()
    (codex / "auth.json").write_text(json.dumps({"tokens": {"access_token": "synthetic-codex-token"}}))
    (claude / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "synthetic-claude-token", "refreshToken": "synthetic-refresh-token", "subscriptionType": "max"}}))
    monkeypatch.setenv("CODEX_HOME", str(codex))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setattr(sub.shutil, "which", lambda name: "/fake/" + name)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(name, "synthetic-paid-secret")
    return codex, claude


@pytest.fixture
def configured(tmp_path, credentials):
    ctx = RunContext.create(run_id="subscription", root_workdir=tmp_path / "run", flat=True,
                            run_budget=BudgetSpec(max_usd=1050, max_wallclock_s=120))
    workflow = ACWorkflow(ctx)
    inputs = sub.configure_workflow(workflow, ACWorkflow.Inputs(problem="Synthetic question", problem_id="synthetic",
                                    execution_backend="subscription"))
    return workflow, inputs


@pytest.fixture
def fake_cli(monkeypatch):
    """Execute real CLI lifecycle, but only spawn a synthetic Python program."""
    seen = []
    command_for = ConfigurableCLIAgent._command_for

    def command(self, inp):
        cmd = command_for(self, inp)
        seen.append((cmd, self.component_config, inp))
        reply = REPLY.replace("<ready>true", "<ready>false") if getattr(inp, "file_0", None) == "" else REPLY
        tex = TEX.replace("synthetic proof", "simplified synthetic proof") if "proof-simplification phase" in getattr(inp, "packet", "") else TEX
        # Assert the *child's* effective credentials, not just the config.
        script = """
import json, os, sys
from pathlib import Path
sys.stdin.read()
assert not any(k.endswith('API_KEY') or k == 'ANTHROPIC_AUTH_TOKEN' for k in os.environ)
"""
        if cmd[0] == "codex":
            script += "assert json.loads((Path(os.environ['CODEX_HOME']) / 'auth.json').read_text())['tokens']['access_token'] == 'synthetic-codex-token'\n"
            usage = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 20}}
        else:
            script += "assert json.loads((Path(os.environ['CLAUDE_CONFIG_DIR']) / '.credentials.json').read_text())['claudeAiOauth']['subscriptionType'] == 'max'\n"
            usage = {"type": "result", "subtype": "success", "is_error": False, "result": REPLY,
                     "usage": {"input_tokens": 100, "output_tokens": 20}, "num_turns": 1,
                     "total_cost_usd": 0.01, "modelUsage": {sub.CLAUDE_MODEL: {
                         "inputTokens": 100, "outputTokens": 20, "cacheReadInputTokens": 0,
                         "cacheCreationInputTokens": 0, "costUSD": 0.01}}}
        script += f"Path('report.md').write_text({reply!r})\n"
        script += f"Path('answer.tex').write_text({tex!r})\n"
        script += "Path('feedback.md').write_text('Synthetic editorial report')\n"
        script += "Path('completion.json').write_text(json.dumps({'status':'ready','summary':'Synthetic completed edit'}))\n"
        script += f"print({json.dumps(usage)!r}, flush=True)\n"
        return [sys.executable, "-c", script]

    monkeypatch.setattr(ConfigurableCLIAgent, "_command_for", command)
    monkeypatch.setattr(ConfigurableCLIAgent, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr("proofstack.sandbox.subprocess._terminate_marked_processes", AsyncMock(return_value=True))
    # No provider object may be instantiated, even if a forgotten fallback fires.
    import mathagents
    monkeypatch.setattr(mathagents, "APIClient", lambda *a, **kw: pytest.fail("paid API attempted"))
    return seen


def test_api_remains_default_without_subscription_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(sub, "preflight", lambda: pytest.fail("subscription preflight in API mode"))
    agent = ACWorkflow(RunContext.create(run_id="api", root_workdir=tmp_path, flat=True))
    inp = ACWorkflow.Inputs(problem="P", problem_id="synthetic")
    assert sub.configure_workflow(agent, inp) is inp
    assert agent.ctx.execution_backend == "api"
    assert not agent.tracker.is_subscription


def test_configures_all_roles_and_preserves_api_ledger(configured):
    agent, inp = configured
    assert inp.council_models == sub.COUNCIL_MODELS
    assert inp.compute_model == sub.CODEX_MODEL
    assert inp.compute_reasoning_effort == "xhigh"
    assert all(child.ctx is agent.ctx for child in (agent.author, agent.critic, agent.council, agent.compute))
    agent.tracker.add_api_equivalent_usd(12)
    assert agent.ctx.budgets.root().planning_usd == 12
    assert agent.ctx.budgets.root().counters.usd == 0
    with pytest.raises(RuntimeError, match="Paid"):
        agent.tracker.add_usd(1)
    with pytest.raises(sub.SubscriptionBlocked):
        agent.ctx.api_client_factory("anything")


@pytest.mark.parametrize("bad", [None, "not json", json.dumps({"OPENAI_API_KEY": "synthetic-paid-secret"})])
def test_missing_or_paid_codex_login_never_uses_host_key(credentials, bad):
    path = credentials[0] / "auth.json"
    path.unlink() if bad is None else path.write_text(bad)
    with pytest.raises(sub.SubscriptionBlocked):
        sub.preflight()


def test_claude_requires_subscription_not_console_auth(credentials):
    (credentials[1] / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": "console"}}))
    with pytest.raises(sub.SubscriptionBlocked):
        sub.preflight()


def test_api_guard_blocks_constructed_clients_and_propagates_to_threads():
    from mathagents.api_client import APIClient
    async def run():
        with subscription_only():
            with pytest.raises(RuntimeError, match="API"):
                APIClient(model="unused")
            existing = object.__new__(APIClient)
            with pytest.raises(RuntimeError, match="API"):
                await asyncio.to_thread(lambda: next(existing.run_queries([])))
    asyncio.run(run())


def test_backend_and_allowance_cannot_change_on_resume(configured):
    agent, inp = configured
    for update in ({"execution_backend": "api"}, {"subscription_planning_usd": 2000}):
        with pytest.raises(sub.SubscriptionBlocked):
            sub.configure_workflow(agent, inp.model_copy(update=update))
    assert sub.configure_workflow(agent, inp).execution_backend == "subscription"


@pytest.mark.parametrize("receipt", ['{"status":"started"}', '{"status":"completed"}', 'truncated'])
def test_resume_blocks_all_unresolved_calls_even_with_changed_input(configured, receipt):
    agent, inp = configured
    root = agent.ctx.root_workdir
    (root / "subscription_calls").mkdir()
    (root / "subscription_calls" / "earlier-packet.json").write_text(receipt)
    resumed = ACWorkflow(RunContext.create(run_id="subscription", root_workdir=root, flat=True))
    with pytest.raises(sub.SubscriptionBlocked, match="Unresolved subscription"):
        sub.configure_workflow(resumed, inp.model_copy(update={"problem": "different input"}))


@pytest.mark.parametrize("payload", [{"subscription": True},
    {"subscription": True, "api_equivalent_usd": -1},
    {"subscription": True, "api_equivalent_usd": 1, "cost_usd": 1}])
def test_invalid_subscription_ledger_cannot_reset_budget(configured, payload):
    agent, _ = configured
    asyncio.run(agent.events.emit("model.call", payload))
    with pytest.raises(sub.SubscriptionBlocked, match="usage ledger"):
        sub.restore_usage(agent.ctx, BudgetTracker("resumed", subscription_mode=True))


@pytest.mark.parametrize("worker_type", [ConfigurableCLIAgent, Compute])
def test_inherited_subscription_policy_rejects_misconfigured_api_worker(tmp_path, worker_type):
    ctx = RunContext.create(run_id="api", root_workdir=tmp_path, flat=True)
    worker = worker_type(ctx)
    async def run():
        with subscription_only():
            with pytest.raises(RuntimeError, match="Paid API"):
                await worker.run(SimpleNamespace())
    asyncio.run(run())


def test_resume_restores_nominal_usage_but_not_reasoning_twice(configured):
    agent, _ = configured
    async def record():
        await agent.events.emit("model.call", {"subscription": True, "api_equivalent_usd": 2,
                                "in_tokens": 100, "out_tokens": 20, "reasoning_out_tokens": 15})
        await agent.events.emit("model.call", {"subscription": True, "api_equivalent_usd": 3,
                                "metered_tokens": 50})
    asyncio.run(record())
    tracker = BudgetTracker("resumed", BudgetSpec(max_usd=5), subscription_mode=True)
    sub.restore_usage(agent.ctx, tracker)
    sub.restore_usage(agent.ctx, tracker)
    assert tracker.counters.usd == 0
    assert tracker.planning_usd == 5
    assert tracker.counters.tokens == 170
    with pytest.raises(BudgetExhausted, match="api_equivalent"):
        tracker.check()


def test_author_and_council_use_cli_and_completed_calls_are_reused(configured, fake_cli):
    workflow, inp = configured
    async def run():
        author = Author(workflow.ctx)
        first = await author(problem="P", round=0, n_rounds=2, research_notes_tex="all notes", references_bib="bib")
        second = await author(problem="P", round=0, n_rounds=2, research_notes_tex="all notes", references_bib="bib")
        council = await Council(workflow.ctx)(author_question="Q", answer_tex=TEX, member_models=inp.council_models)
        return first, second, council
    first, second, council = asyncio.run(run())
    assert first.answer_tex == second.answer_tex == TEX
    assert first.research_notes_tex == "all notes"
    assert first.via == "subscription_cli"
    assert len(council.replies) == 2 and all(reply.text and not reply.error for reply in council.replies)
    assert [cmd[0] for cmd, _, _ in fake_cli].count("codex") == 2
    assert [cmd[0] for cmd, _, _ in fake_cli].count("claude") == 1
    for cmd, cfg, _ in fake_cli:
        assert cfg["sandbox"]["provider_keys"] == []
        if cmd[0] == "codex":
            assert sub.CODEX_MODEL in cmd and 'model_reasoning_effort="xhigh"' in cmd
            assert 'forced_login_method="chatgpt"' in cmd
        else:
            assert sub.CLAUDE_MODEL in cmd and "--bare" not in cmd
    assert workflow.tracker.counters.usd == 0
    assert workflow.ctx.budgets.root().planning_usd > 0
    assert not list(workflow.ctx.root_workdir.rglob("auth.json"))
    assert not list(workflow.ctx.root_workdir.rglob(".credentials.json"))


@pytest.mark.parametrize("critic", [ACCritic, Batch3CleanupCritic, SimplificationCritic, FinalStatelessCritic])
def test_every_critic_uses_codex_with_original_protocol(configured, fake_cli, critic):
    workflow, _ = configured
    async def run():
        seat = critic(workflow.ctx)
        fields = dict(problem="P", round=1, answer_tex=TEX, research_notes_tex="Full research notes",
                      mode="fresh", baseline_tex=TEX, compile_feedback="OK", editor_response="Proposed simplification")
        return await seat(**fields)
    result = asyncio.run(run())
    assert result.answer_ready
    assert not result.parse_failed
    assert fake_cli[0][0][0] == "codex"
    if critic in (ACCritic, FinalStatelessCritic):
        assert result.research_notes_status == "inline"


def test_stateless_review_has_no_history_and_stays_distinct(configured, fake_cli):
    workflow, _ = configured
    async def run():
        inp = dict(problem="P", round=1, answer_tex=TEX, mode="fresh")
        await ACCritic(workflow.ctx)(**inp)
        await FinalStatelessCritic(workflow.ctx)(**inp)
    asyncio.run(run())
    assert len(fake_cli) == 2
    for _, _, inp in fake_cli:
        assert "# assistant" not in inp.packet


def test_interrupted_receipt_blocks_retry(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    async def interrupted(self, inp):
        raise asyncio.CancelledError
    async def run():
        with monkeypatch.context() as m:
            m.setattr(ConfigurableCLIAgent, "run", interrupted)
            with pytest.raises(asyncio.CancelledError):
                await sub.run_packet(workflow, "request")
        with pytest.raises(sub.SubscriptionBlocked, match="Interrupted"):
            await sub.run_packet(workflow, "request")
    asyncio.run(run())
    assert not fake_cli


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_missing_usage_is_not_free_success(configured, provider):
    workflow, _ = configured
    config = sub.cli_config(provider, 10)
    ctx = replace(workflow.ctx, component_configs={"worker": config})
    worker = ConfigurableCLIAgent(ctx, name="worker")
    async def run():
        with pytest.raises((RuntimeError, ValueError)):
            await worker.record_cli_usage("", "", CLIDoneRecord(status="done"))
    asyncio.run(run())
    assert worker.cli_usage_must_succeed()


def test_tools_disabled_reader_gets_prefix_in_prompt(configured, fake_cli):
    workflow, _ = configured
    result = asyncio.run(sub.run_packet(workflow, "Only prefix A, not B", provider="claude", tools=False))
    assert result["text"] == REPLY
    cmd, _, inp = fake_cli[0]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "Only prefix A, not B" in inp.prompt
    assert "Read request.md" not in inp.prompt


def test_concurrent_claude_calls_isolate_and_remove_auth(configured, fake_cli, monkeypatch):
    from proofstack.transient_auth import codex_auth_run_root
    workflow, _ = configured
    ctx = replace(workflow.ctx, component_configs={"worker": sub.cli_config("claude", 10)})
    worker = ConfigurableCLIAgent(ctx, name="worker")
    setup = ConfigurableCLIAgent.setup
    homes = []
    barrier = asyncio.Barrier(2)
    async def concurrent_setup(self, sandbox, inp):
        await setup(self, sandbox, inp)
        home = self._claude_auth_home
        homes.append(home)
        await barrier.wait()
        assert self._claude_auth_home == home
        assert (home / ".credentials.json").is_file()
    monkeypatch.setattr(ConfigurableCLIAgent, "setup", concurrent_setup)
    async def run():
        return await asyncio.gather(worker(prompt="One", packet="One"), worker(prompt="Two", packet="Two"))
    outputs = asyncio.run(run())
    assert all(out.status == "done" for out in outputs)
    assert len(set(homes)) == 2
    assert not any(home.exists() for home in homes)
    assert not codex_auth_run_root(ctx.root_workdir).exists()


def test_cleanup_editor_native_reviewers_and_attribution_are_subscription_only(configured, fake_cli, monkeypatch):
    from proofstack.agents import cleanup_session as cleanup
    workflow, _ = configured
    config = {"CleanupSession": {"cleanup": {"review_retry_reserve_fraction": 0}}}
    ctx = replace(workflow.ctx, component_configs=config)
    agent = CleanupSession(ctx)
    monkeypatch.setattr(agent, "_preflight", AsyncMock())
    sandbox = agent._sandbox
    monkeypatch.setattr(agent, "_sandbox", lambda *args: {**sandbox(*args), "memory_policy": None})
    monkeypatch.setattr(cleanup.CleanupReviews, "attribution_retrieved", lambda self: True)
    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        await callbacks["codex_review"]("Review citations")
    monkeypatch.setattr(cleanup, "cleanup_tools", tools)
    result = asyncio.run(agent(problem="P", document=TEX, baseline=TEX))
    assert result.status == "ready"
    assert ctx.budgets.root().counters.usd == 0
    assert ctx.budgets.root().planning_usd > 0
    assert {cmd[0] for cmd, _, _ in fake_cli} == {"codex", "claude"}
    editor = next(cmd for cmd, _, _ in fake_cli if cmd[0] == "claude")
    assert all(a["model"] == sub.CLAUDE_MODEL for a in json.loads(editor[editor.index("--agents") + 1]).values())
    for cmd, cfg, _ in fake_cli:
        assert "OPENAI_API_KEY" not in cfg["sandbox"]["provider_keys"]
        if cmd[0] == "codex":
            assert 'forced_login_method="api"' not in cmd
        else:
            assert cfg["usage"]["auth_mode"] == "subscription"


def test_compute_missing_login_fails_before_spawn_despite_host_api_key(configured, credentials, monkeypatch):
    workflow, _ = configured
    (credentials[0] / "auth.json").unlink()
    monkeypatch.setattr("proofstack.kinds.cli.CLIAgent.run", AsyncMock(side_effect=AssertionError("spawned")))
    async def run():
        with pytest.raises(RuntimeError, match="paid API fallback"):
            await Compute(workflow.ctx)(compute_workspace=workflow.ctx.root_workdir / "compute", problem="P",
                    problem_id="synthetic", round=1, instructions="Count to two")
    asyncio.run(run())


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_VERTEX", "CODEX_API_KEY"])
def test_rejects_any_injected_paid_child_environment(key):
    with pytest.raises(sub.SubscriptionBlocked):
        sub.check_cli_environment({key: "injected"})


def test_complete_batch3_research_simplification_cleanup_and_final_gate(configured, fake_cli, monkeypatch):
    from proofstack.agents import cleanup_session as cleanup
    from proofstack.agents.ac import ac_workflow as ac
    from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow

    workflow, _ = configured
    monkeypatch.setattr(ac, "_simple_compile_latex", lambda tex, *args, **kwargs:
                        ac._CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1))
    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", AsyncMock(return_value=(True, 1, "")))
    monkeypatch.setattr(CleanupSession, "_preflight", AsyncMock())
    sandbox = CleanupSession._sandbox
    monkeypatch.setattr(CleanupSession, "_sandbox", lambda *args: {**sandbox(*args), "memory_policy": None})
    monkeypatch.setattr(cleanup.CleanupReviews, "attribution_retrieved", lambda self: True)
    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
    monkeypatch.setattr(cleanup, "cleanup_tools", tools)

    agent = FirstProofBatch3Workflow(workflow.ctx)
    result = asyncio.run(agent(problem="Synthetic question", problem_id="synthetic", execution_backend="subscription",
        n_rounds=2, full_critic_interval=1, enable_compute=False, enable_council=False,
        research_seconds=60, partial_cleanup_seconds=0, final_review_seconds=10, final_review_export_seconds=1,
        simplification_rounds=1))
    assert result.error is None, result.model_dump()
    assert result.submission_approved, result.model_dump()
    assert result.final_critic_answer_ready
    assert result.answer_tex.is_file()
    assert any(cmd[0] == "claude" for cmd, _, _ in fake_cli)
    events = [json.loads(line) for line in (workflow.ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    agents = [r["agent"] for r in events if r["kind"] == "agent.start"]
    assert any("simplif" in name.lower() for name in agents)
    assert "final_stateless_critic" in agents
    charges = [r["payload"] for r in events if r["kind"] == "model.call"]
    assert charges and all(row["subscription"] and row["cost_usd"] == 0 for row in charges)
    assert sum(row["api_equivalent_usd"] for row in charges) > 0


def test_completed_over_budget_critic_retains_typed_verdict(configured, fake_cli):
    workflow, _ = configured
    workflow.ctx.budgets.root().spec = BudgetSpec(max_usd=0.000001, max_wallclock_s=120)
    async def run():
        with pytest.raises(BudgetExhausted) as error:
            await ACCritic(workflow.ctx)(problem="P", round=1, answer_tex=TEX, mode="fresh")
        assert isinstance(error.value.completed_output, ACCritic.Outputs)
        assert error.value.completed_output.answer_ready
    asyncio.run(run())
    assert len(fake_cli) == 1


def test_subscription_error_blocks_instead_of_falling_back(configured, fake_cli, monkeypatch):
    workflow, _ = configured
    worker = AsyncMock(side_effect=RuntimeError("quota unavailable"))
    monkeypatch.setattr(ConfigurableCLIAgent, "run", worker)
    async def run():
        with pytest.raises(sub.SubscriptionBlocked, match="no API fallback"):
            await sub.run_packet(workflow, "request")
        with pytest.raises(sub.SubscriptionBlocked, match="Interrupted"):
            await sub.run_packet(workflow, "request")
    asyncio.run(run())
    assert worker.await_count == 1
    assert not fake_cli


def test_concurrent_identical_packets_start_once(configured, fake_cli):
    workflow, _ = configured
    async def run():
        return await asyncio.gather(sub.run_packet(workflow, "same"), sub.run_packet(workflow, "same"), return_exceptions=True)
    results = asyncio.run(run())
    assert len(fake_cli) == 1
    assert any(isinstance(out, dict) and out["text"] == REPLY for out in results)
    assert all(isinstance(out, (dict, sub.SubscriptionBlocked)) for out in results)


def test_completed_receipt_reuse_survives_new_worker_instance(configured, fake_cli):
    workflow, _ = configured
    async def run():
        first = await sub.run_packet(workflow, "same", key="receipt")
        second = await sub.run_packet(SimpleNamespace(ctx=workflow.ctx, tracker=workflow.tracker, name=workflow.name),
                                      "same", key="receipt")
        assert first == second
    asyncio.run(run())
    assert len(fake_cli) == 1


def test_deadline_covers_queued_workers_without_starting_one(configured, fake_cli):
    workflow, _ = configured
    workflow.tracker.spec = BudgetSpec(max_wallclock_s=0.05)
    workflow.ctx.subscription_limiter = asyncio.Semaphore(0)
    async def run():
        with pytest.raises(TimeoutError):
            await sub.run_packet(workflow, "queued")
    asyncio.run(run())
    assert not fake_cli
    assert not list((workflow.ctx.root_workdir / "subscription_calls").glob("*.json"))


@pytest.mark.parametrize("setting", ["monitor", "parallelism", "external_resume"])
def test_rejects_unsupported_paths_before_model_calls(configured, setting):
    workflow, inp = configured
    if setting == "monitor":
        workflow.ctx.monitor = object()
    elif setting == "external_resume":
        workflow.ctx.resume_from = "another-run-with-no-ledger"
    else:
        inp = inp.model_copy(update={"author_parallelism": 3})
    with pytest.raises((sub.SubscriptionBlocked, ValueError)):
        sub.configure_workflow(workflow, inp)


def test_adapter_skips_api_login_bootstrap_for_subscription_preset(tmp_path, credentials, monkeypatch):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    import proofstack.registry as registry
    _clear_firstproof_env(monkeypatch)
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    monkeypatch.setattr(registry, "load_preset", lambda *a: SimpleNamespace(inputs={"execution_backend": "subscription"}))
    bootstrap = AsyncMock(side_effect=AssertionError("API login bootstrap attempted"))
    monkeypatch.setattr(fp, "_bootstrap_codex_auth", bootstrap)
    monkeypatch.setattr(fp, "_run_healthcheck", AsyncMock(return_value=settings))
    assert asyncio.run(fp._startup_checks(settings)) is settings
    bootstrap.assert_not_awaited()


@pytest.mark.parametrize("mode", ["off", "warn", "strict"])
def test_adapter_cannot_choose_api_cleanup_when_subscription_unavailable(tmp_path, monkeypatch, mode):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    import proofstack.registry as registry
    from proofstack import healthcheck
    from proofstack.agents import cleanup_session as cleanup
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    preset = registry.load_preset("firstproof_batch3")
    preset.inputs["execution_backend"] = "subscription"
    monkeypatch.setattr(registry, "load_preset", lambda *a: preset)
    monkeypatch.setattr(healthcheck, "check_compute", AsyncMock())
    monkeypatch.setattr(cleanup, "check_cleanup_launch", AsyncMock(side_effect=cleanup.CleanupUnavailable("test unavailable")))
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    with pytest.raises(cleanup.CleanupUnavailable):
        asyncio.run(fp._run_healthcheck(settings))
    assert settings.cleanup_disabled_reason is None
