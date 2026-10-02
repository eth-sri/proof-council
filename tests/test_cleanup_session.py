from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from proofstack.agents import cleanup_session as mod
from proofstack.agents.ac.ac_workflow import ACWorkflow, _sum_logged_model_cost
from proofstack.agents.cleanup_session import CleanupSession, CleanupSettings, _MeteredTurn, _read_file
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext
from proofstack.registry import load_preset


TEX = r"\documentclass{article}\begin{document}An honest partial result.\end{document}"
_attribution_retrieved = mod.CleanupReviews.attribution_retrieved


@pytest.mark.parametrize("parent_cap,episode_remaining,spent,grant", [
    (1000, 500, 48.55, 75), (200, 500, 48.55, 50),
    (1000, 180, 48.55, 30), (150, 500, 48.55, 0),
    (1000, 150, 48.55, 0), (200, 500, 155, 45),
])
def test_review_topup_preserves_promised_funds_and_all_enclosing_caps(parent_cap, episode_remaining, spent, grant):
    from proofstack.budget import BudgetTracker

    parent = BudgetTracker("problem", BudgetSpec(max_usd=parent_cap))
    invocation = BudgetTracker("invocation", BudgetSpec(max_usd=150), parent)
    helper = BudgetTracker("reviewer", BudgetSpec(max_usd=45), invocation)
    helper.add_usd(spent)
    granted = mod._top_up_review_budget(helper, invocation, parent, episode_remaining=episode_remaining, max_topup=75)
    assert granted == pytest.approx(grant)
    assert helper.spec.max_usd == pytest.approx(45 + grant)
    assert invocation.spec.max_usd == pytest.approx(150 + grant)
    assert parent.counters.usd == spent
    assert parent.spec.max_usd == parent_cap


def _claude_session_tail(cmd, cost, *, result=True):
    """Mimic Claude Code: a resumed process restores the transcript's saved
    cumulative total, and its result reports that cumulative total."""
    resume = "--resume" in cmd
    sid = cmd[cmd.index("--resume" if resume else "--session-id") + 1]
    tail = (
        "import json,os; from pathlib import Path; "
        "cfg = Path(os.environ.get('CLAUDE_CONFIG_DIR', '.claude')); "
        f"ts = sorted(cfg.glob('projects/*/{sid}.jsonl')); t = ts[0] if ts else cfg / 'projects/p/{sid}.jsonl'; "
        f"prev = [json.loads(l) for l in t.read_text().splitlines() if 'cost-state' in l] if {resume} and t.exists() else []; "
        "n = len(prev) + 1; "
        f"total = (prev[-1]['totalCostUSD'] if prev else 0) + {cost}; "
        "mu = {'claude-fable-5-1': {'inputTokens': 100*n, 'outputTokens': 20*n, 'cacheReadInputTokens': 0, "
        "'cacheCreationInputTokens': 0, 'costUSD': total}}; "
    )
    if result:
        tail += ("print(json.dumps({'type':'result','subtype':'success','session_id':'test',"
                 "'usage':{'input_tokens':100,'output_tokens':20},'num_turns':1,'total_cost_usd':total,"
                 "'modelUsage':mu}), flush=True); ")
    return tail + ("t.parent.mkdir(parents=True, exist_ok=True); "
                   "t.open('a').write(json.dumps({'type':'cost-state','totalCostUSD':total,'modelUsage':mu}) + chr(10)); ")


@pytest.fixture
def editor(tmp_path, monkeypatch):
    from proofstack.sandbox import subprocess as sandbox_process

    ctx = RunContext.create(run_id="cleanup", root_workdir=tmp_path, flat=True,
                            run_budget=BudgetSpec(max_usd=10, max_wallclock_s=60),
                            component_configs={"CleanupSession": {"cleanup": {"review_retry_reserve_fraction": 0}}})
    seen = {"commands": [], "envs": [], "tools": [], "prompts": [], "completion": True}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CODEX_API_KEY", "unrelated-host-key")

    async def preflight(self, settings):
        pass

    sandbox = CleanupSession._sandbox

    def unqueued_test_sandbox(self, settings, seconds, provider):
        spec = sandbox(self, settings, seconds, provider)
        spec["memory_policy"] = None
        return spec

    @asynccontextmanager
    async def tools(**callbacks):
        seen["tools"].append(callbacks)
        yield {"mcpServers": {}}

    def command(self, inp):
        original = super(_MeteredTurn, self)._command_for(inp)
        seen["commands"].append(original)
        seen["envs"].append(self.component_config.get("env") or {})
        if original[0] == "codex":
            script = (
                "import sys,json,os; from pathlib import Path; Path('stdin.txt').write_text(sys.stdin.read()); "
                "assert Path('.codex').is_dir(); "
                "assert os.environ['CODEX_API_KEY'] == os.environ['OPENAI_API_KEY'] == 'test-key'; "
                "Path('report.md').write_text('Independent review of the snapshot'); "
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':20}}))"
            )
        else:
            seen["prompts"].append(inp.prompt)
            script = (
                "import sys,json,os; from pathlib import Path; sys.stdin.read(); "
                "assert 'CODEX_API_KEY' not in os.environ and 'OPENAI_API_KEY' not in os.environ; "
                "assert 'An honest partial result' in Path('answer.tex').read_text(); "
                "Path('feedback.md').write_text('Checked exposition; known gap retained.'); "
                + _claude_session_tail(original, 0.25)
            )
            if seen["completion"]:
                script += "Path('completion.json').write_text(json.dumps({'status':'ready','summary':'Edited'})); "
        return [sys.executable, "-c", script]

    monkeypatch.setattr(CleanupSession, "_preflight", preflight)
    monkeypatch.setattr(CleanupSession, "_sandbox", unqueued_test_sandbox)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    # These dummy editors test lifecycle/accounting rather than MCP review
    # retrieval. Dedicated review-job tests below exercise the real gate.
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", lambda self: True)
    monkeypatch.setattr(_MeteredTurn, "_command_for", command)
    monkeypatch.setattr(_MeteredTurn, "POLL_INTERVAL_S", 0.01)
    # These dummy-CLI tests exercise editorial lifecycle/accounting, not the
    # host-wide psutil scan (covered by the sandbox memory tests).
    monkeypatch.setattr(sandbox_process, "markers_rss", lambda markers: {
        marker["token"]: (1024, True) for marker in markers
    })
    return CleanupSession(ctx), ctx, seen


@pytest.mark.parametrize("partial", [False, True])
def test_real_cli_lifecycle_reuses_explicit_session_and_bills_once(editor, partial):
    agent, ctx, seen = editor

    async def run():
        first = await agent(problem="P", document=TEX, baseline=TEX, partial=partial)
        second = await agent(problem="P", document=first.answer_tex, baseline=TEX,
                             findings="Repair the introduction", partial=partial)
        return first, second

    first, second = asyncio.run(run())
    assert first.session_id == second.session_id
    assert first.workspace == second.workspace
    assert first.status == "ready"
    assert ctx.budgets.root().counters.usd == pytest.approx(0.5)
    assert "--session-id" in seen["commands"][0]
    assert "--resume" in seen["commands"][1]
    cmd = seen["commands"][1]
    assert cmd[cmd.index("--resume") + 1] == first.session_id
    assert "feedback" in first.feedback_md.lower() or "exposition" in first.feedback_md
    assert json.loads((first.workspace.parent / "session.json").read_text())["in_flight"] is False
    assert (first.workspace.parent / "baseline.tex").read_text() == TEX
    assert "CURRENT CRITIC REPAIR REQUEST" not in seen["prompts"][0]
    assert ("CURRENT CRITIC REPAIR REQUEST" in seen["prompts"][1]) is (not partial)
    assert "Repair the introduction" in seen["prompts"][1]
    assert "re-read them even if" in seen["prompts"][1]
    assert ("unchanged manuscript" in seen["prompts"][1]) is (not partial)
    if partial:
        assert "not instructions to solve missing mathematics" in seen["prompts"][1]


@pytest.mark.parametrize("retry_fails", [False, True])
def test_failed_mandatory_review_gets_one_bounded_topup_and_retry(editor, monkeypatch, retry_fails):
    agent, ctx, _ = editor
    ctx.budgets.root().spec = BudgetSpec(max_usd=30, max_wallclock_s=60)
    agent.component_config = {"cleanup": {"max_invocation_usd": 10, "max_review_topup_usd": 5, "review_retry_reserve_fraction": 0}}
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)
    run = _MeteredTurn.run
    calls = []

    async def failed_review(self, inp):
        result = await run(self, inp)
        if self.component_config["usage"]["type"] == "codex_jsonl":
            calls.append(self)
            if len(calls) == 1 or retry_fails:
                self.tracker.add_usd(4)
                raise RuntimeError("synthetic stopped, metered reviewer failure")
        return result

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        first = await callbacks["review_start"]("Verify citations", "attribution")
        result = await callbacks["review_status"](first["review_id"], 20)
        assert result["status"] == "failed"
        second = await callbacks["review_start"]("Focused citation retry", "attribution")
        result = await callbacks["review_status"](second["review_id"], 20)
        if retry_fails:
            assert result["status"] == "failed"
            with pytest.raises(BudgetExhausted):
                await callbacks["review_start"]("Third retry must not get another top-up", "attribution")
        else:
            assert result["status"] == "completed"

    monkeypatch.setattr(_MeteredTurn, "run", failed_review)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    if retry_fails:
        with pytest.raises(mod.CleanupIncomplete, match="attribution"):
            asyncio.run(agent(problem="P", document=TEX))
    else:
        assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"
    assert len(calls) == 2
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert state["review_topups"] == [{"usd": 5, "reason": "failed mandatory attribution review"}]
    assert state["recorded_usd"] == pytest.approx(ctx.budgets.root().counters.usd)
    assert (8 if retry_fails else 4) < state["recorded_usd"] < (9 if retry_fails else 5)
    assert not state["in_flight"]


@pytest.mark.parametrize("finishing_only", [False, True])
def test_finishing_gate_blocks_new_optional_work_but_preserves_mandatory_review_and_reports(editor, monkeypatch, finishing_only):
    agent, _, seen = editor
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)

    @asynccontextmanager
    async def tools(**callbacks):
        assert callbacks["progress"]()["stage"] == ("finishing" if finishing_only else "editing")
        yield {"mcpServers": {}}
        if not finishing_only:
            original_clock = time.monotonic
            monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: original_clock()+49, time=time.time))
        assert callbacks["progress"]()["stage"] == "finishing"
        with pytest.raises(RuntimeError, match="Finishing stage"):
            await callbacks["review_start"]("Optional broad review", "general")
        first = await callbacks["review_start"]("Required attribution", "attribution")
        result = await callbacks["review_status"](first["review_id"], 20)
        assert result["status"] == "completed"
        assert (await callbacks["review_start"]("Required attribution", "attribution"))["review_id"] == first["review_id"]
        with pytest.raises(RuntimeError, match="Finishing stage"):
            await callbacks["review_start"]("Another broad attribution review", "attribution")

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    result = asyncio.run(agent(problem="P", document=TEX, finishing_only=finishing_only))
    assert result.status == "ready"
    command = seen["commands"][0]
    assert command[command.index("--tools")+1] == ("" if finishing_only else "Task")
    assert "cleanup_control" in seen["prompts"][0]


@pytest.mark.parametrize("finishing_only", [False, True])
@pytest.mark.parametrize("retry_reserve", [0, .2])
def test_finishing_reuses_paid_attribution_budget_without_releasing_estimates(editor, monkeypatch, finishing_only, retry_reserve):
    agent, ctx, seen = editor
    ctx.budgets.root().spec = BudgetSpec(max_usd=30, max_wallclock_s=60)
    agent.component_config = {"cleanup": {"max_invocation_usd": 10, "review_retry_reserve_fraction": retry_reserve}}
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)
    invocations = []
    review_id = None

    @asynccontextmanager
    async def tools(**callbacks):
        nonlocal review_id
        invocations.append(callbacks)
        yield {"mcpServers": {}}
        if len(invocations) == 1:
            job = await callbacks["review_start"]("Verify citations", "attribution")
            review_id = job["review_id"]
            report = await callbacks["review_status"](review_id, 20)
            assert report["status"] == "completed"
        else:
            report = await callbacks["review_status"](review_id)
            assert report["status"] == "completed" and report["retrieved"]
            if finishing_only:
                with pytest.raises(RuntimeError, match="Finishing stage"):
                    await callbacks["review_start"]("Optional second review", "general")
                with pytest.raises(ValueError, match="remaining USD budget"):
                    await callbacks["codex_review"]("Do not bypass the finishing gate")

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    asyncio.run(agent(problem="P", document=TEX))
    state_path = ctx.root_workdir / "cleanup_sessions/standalone/session.json"
    state = json.loads(state_path.read_text())
    # An existing conservative debit must remain charged, even when the
    # finishing invocation does not need a new attribution allocation.
    ctx.budgets.root().add_usd(6)
    state["recorded_usd"] += 6
    state["estimated_usd"] = 6
    state_path.write_text(json.dumps(state))
    before = ctx.budgets.root().counters.usd
    asyncio.run(agent(problem="P", document=TEX, findings="Finish saved edits", finishing_only=finishing_only))
    commands = [cmd for cmd in seen["commands"] if cmd[0] == "claude"]
    caps = [float(cmd[cmd.index("--max-budget-usd") + 1]) for cmd in commands]
    editing_cap = 6 * (1 - retry_reserve)
    finishing_cap = 10 - (1 - retry_reserve)
    assert caps == pytest.approx([editing_cap, finishing_cap if finishing_only else editing_cap])
    assert sum(cmd[0] == "codex" for cmd in seen["commands"]) == 1
    assert ctx.budgets.root().counters.usd == pytest.approx(before + 0.25)
    _assert_settled(ctx, 6)
    if finishing_only:
        assert "your own saved draft, possibly unfinished" in seen["prompts"][-1]
        assert "former budget share is now available" in seen["prompts"][-1]
        assert "Finishing stage is active now" in seen["prompts"][-1]
        assert "Finishing stage starts with" not in seen["prompts"][-1]
        assert "top-up" not in seen["prompts"][-1]


def test_finishing_without_attribution_keeps_its_review_allocation(editor, monkeypatch):
    agent, _, seen = editor
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)
    with pytest.raises(mod.CleanupIncomplete, match="attribution"):
        asyncio.run(agent(problem="P", document=TEX, finishing_only=True))
    cmd = seen["commands"][0]
    assert float(cmd[cmd.index("--max-budget-usd") + 1]) == pytest.approx(6)
    assert "former budget share is now available" not in seen["prompts"][0]
    assert "your own saved draft" not in seen["prompts"][0]
    assert "FINISHING STAGE:" in seen["prompts"][0]


def test_finishing_repair_does_not_claim_a_replaced_manuscript_is_its_own(editor):
    agent, _, seen = editor
    asyncio.run(agent(problem="P", document=TEX, baseline=TEX))
    replacement = TEX + "\n% Mechanical repair from the harness.\n"
    asyncio.run(agent(problem="P", document=replacement, baseline=TEX, finishing_only=True,
                      constraints="Mechanical repairs only; do not change the mathematics."))
    prompt = seen["prompts"][-1]
    assert "FINISHING STAGE:" in prompt and "your own saved draft" not in prompt
    assert "does not authorize substantive edits during a mechanical repair" in prompt
    assert "Finishing stage starts with" not in prompt
    assert seen["commands"][-1][seen["commands"][-1].index("--tools") + 1] == ""


def test_finishing_is_sticky_across_editor_invocations(editor, monkeypatch):
    agent, _, seen = editor
    clock = [1000.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    stages = []

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        if not stages:
            clock[0] += 50
        stages.append(callbacks["progress"]()["stage"])

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    seen["completion"] = False
    with pytest.raises(RuntimeError, match="invocation limit"):
        asyncio.run(agent(problem="P", document=TEX))
    assert stages == ["finishing", "finishing", "finishing"]
    assert [cmd[cmd.index("--tools") + 1] for cmd in seen["commands"]] == ["Task", "", ""]
    for prompt in seen["prompts"][1:]:
        assert "Finishing stage is active now" in prompt
        assert "FINISHING-ONLY CONTINUATION" not in prompt


@pytest.mark.parametrize("shutdown", ["tools", "reviews"])
def test_real_deadline_during_shutdown_settles_before_propagating(editor, monkeypatch, shutdown):
    agent, ctx, _ = editor
    timeout = None
    original_exit = mod.CleanupReviews.__aexit__

    async def drain():
        task = asyncio.create_task(asyncio.sleep(.04))
        timeout.reschedule(asyncio.get_running_loop().time() + .005)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        if shutdown == "tools":
            await drain()

    async def review_exit(self, *exc):
        if shutdown == "reviews":
            self.tasks["synthetic"] = asyncio.create_task(asyncio.sleep(.04))
            timeout.reschedule(asyncio.get_running_loop().time() + .005)
            # Let the task finish without cancellation so the actual review
            # context's shielded drain is where the deadline arrives.
            self.closing = True
        return await original_exit(self, *exc)

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    monkeypatch.setattr(mod.CleanupReviews, "__aexit__", review_exit)

    async def run():
        nonlocal timeout
        async with asyncio.timeout(None) as timeout:
            await agent(problem="P", document=TEX)

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    state = _assert_settled(ctx, 0)
    assert state["recorded_usd"] == pytest.approx(.25)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_mandatory_retry_has_headroom_when_parent_is_only_the_cleanup_reserve(editor, monkeypatch):
    agent, ctx, _ = editor
    agent.component_config = {"cleanup": {"max_invocation_usd": 10, "review_retry_reserve_fraction": .2}}
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)
    original = _MeteredTurn.run
    calls = []

    async def review(self, inp):
        result = await original(self, inp)
        if self.component_config["usage"]["type"] == "codex_jsonl":
            calls.append(self)
            if len(calls) == 1:
                self.tracker.add_usd(2.4)
                raise RuntimeError("stopped mandatory reviewer")
        return result

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        first = await callbacks["review_start"]("Attribution", "attribution")
        assert (await callbacks["review_status"](first["review_id"], 20))["status"] == "failed"
        second = await callbacks["review_start"]("Focused attribution retry", "attribution")
        assert (await callbacks["review_status"](second["review_id"], 20))["status"] == "completed"

    monkeypatch.setattr(_MeteredTurn, "run", review)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"
    state = _assert_settled(ctx, 0)
    assert state["review_topups"][0]["usd"] == pytest.approx(2)
    assert ctx.budgets.root().spec.max_usd == 10


def test_partial_workflow_resumes_real_dummy_cli_after_settled_timeout(editor, monkeypatch):
    from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow

    _, ctx, seen = editor
    workflow = FirstProofBatch3Workflow(ctx)
    run, command = _MeteredTurn.run, _MeteredTurn._command_for
    turns = []

    def edit_command(self, inp):
        cmd = command(self, inp)
        cmd[-1] += "p=Path('answer.tex'); p.write_text(p.read_text().replace('partial result.', 'partial result. Saved edit.')); "
        return cmd

    async def settled_timeout(self, inp):
        result = await run(self, inp)
        turns.append(self)
        if len(turns) == 1:
            raise TimeoutError("synthetic first-pass timeout after joined worker shutdown")
        return result

    async def compile_document(document, *, deadline):
        return True, 1, "compiled"

    monkeypatch.setattr(_MeteredTurn, "_command_for", edit_command)
    monkeypatch.setattr(_MeteredTurn, "run", settled_timeout)
    monkeypatch.setattr(workflow, "_compile", compile_document)
    inp = workflow.Inputs(problem="P", problem_id="p", cleanup_backend="claude_code", min_partial_continuation_seconds=1)
    result = asyncio.run(workflow._partial(inp, workflow.Outputs(problem_id="p"), TEX, "", "",
                                          deadline=time.monotonic()+50))
    assert len(turns) == 2 and result.partial_ready and not result.submission_approved
    assert "Saved edit" in result.answer_tex.read_text()
    assert "--session-id" in seen["commands"][0] and "--resume" in seen["commands"][1]
    assert seen["commands"][1][seen["commands"][1].index("--tools")+1] == ""
    state = json.loads(next((ctx.root_workdir / "cleanup_sessions").glob("*/session.json")).read_text())
    assert state["recorded_usd"] == pytest.approx(0.5)
    assert ctx.budgets.root().counters.usd == pytest.approx(0.5)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_native_tools_cannot_write_supervisor_or_spawn_unmetered_clis(editor):
    agent, _, seen = editor
    result = asyncio.run(agent(problem="P", document=TEX))
    cmd = seen["commands"][0]
    assert cmd[cmd.index("--tools") + 1] == "Task"
    assert "--bare" not in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in cmd
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {"disableAllHooks": True}
    denied = cmd[cmd.index("--disallowedTools") + 1].split(",")
    assert {"Bash", "Write", "Read", "Edit", "Skill"} <= set(denied)
    assert not {"Task", "Agent"} & set(denied)
    agents = json.loads(cmd[cmd.index("--agents") + 1])
    assert all(a["tools"] == ["mcp__cleanup__read", "mcp__cleanup__files"] for a in agents.values())
    # Only the named reviewers may be delegated to, and they cannot delegate further.
    assert set(agents) == {"correctness-reviewer", "exposition-reviewer"}
    env = seen["envs"][0]
    assert env["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"] == "1"
    assert "Agent(general-purpose)" in cmd[cmd.index("--disallowedTools") + 1]
    assert env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"] == "1"

    async def check():
        callbacks = seen["tools"][0]
        with pytest.raises(ValueError, match="only answer"):
            await callbacks["write_file"]("../session.json", "{}")
        with pytest.raises(ValueError, match="not part"):
            await callbacks["read_file"]("../session.json")
        with pytest.raises(ValueError, match="size limit"):
            await callbacks["write_file"]("answer.tex", "x" * (mod.MAX_FILE_BYTES + 1))
        value = await callbacks["read_file"]("answer.tex")
        assert value["content"] == result.answer_tex

    asyncio.run(check())


def test_missing_completion_continues_same_session_but_is_bounded(editor):
    agent, ctx, seen = editor
    seen["completion"] = False
    with pytest.raises(RuntimeError, match="invocation limit"):
        asyncio.run(agent(problem="P", document=TEX))
    assert len(seen["commands"]) == 3
    assert ctx.budgets.root().counters.usd == pytest.approx(0.75)
    ids = [cmd[cmd.index("--resume" if "--resume" in cmd else "--session-id") + 1]
           for cmd in seen["commands"]]
    assert len(set(ids)) == 1
    assert all("--resume" in cmd for cmd in seen["commands"][1:])


def test_session_rejects_changed_problem_and_unreconciled_turn(editor):
    agent, ctx, seen = editor
    asyncio.run(agent(problem="P", document=TEX))
    with pytest.raises(ValueError, match="different inputs"):
        asyncio.run(agent(problem="Other", document=TEX))
    state_path = ctx.root_workdir / "cleanup_sessions/standalone/session.json"
    state = json.loads(state_path.read_text())
    state["in_flight"] = True
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="reconcile"):
        asyncio.run(agent(problem="P", document=TEX))
    assert len(seen["commands"]) == 1


def test_no_usage_from_unconfirmed_worker_locks_session_and_cannot_publish(editor, monkeypatch):
    agent, ctx, seen = editor
    run = _MeteredTurn.run

    async def unconfirmed(self, inp):
        try:
            return await run(self, inp)
        finally:
            self.metering_stream = SimpleNamespace(worker_stopped=False)

    monkeypatch.setattr(_MeteredTurn, "_command_for", lambda *_: [sys.executable, "-c", "print('no usage')"])
    monkeypatch.setattr(_MeteredTurn, "run", unconfirmed)
    with pytest.raises(RuntimeError, match="usage"):
        asyncio.run(agent(problem="P", document=TEX))
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert state["in_flight"]
    assert (ctx.root_workdir / "cleanup-accounting-uncertain.json").exists()
    with pytest.raises(mod.CleanupAccountingUncertain, match="unresolved"):
        asyncio.run(agent(problem="P", document=TEX, session_key="new-episode"))


def test_recorded_overrun_preserves_completed_manuscript_without_accounting_lock(editor):
    agent, ctx, _ = editor
    agent.tracker.spec = BudgetSpec(max_usd=0.1, max_wallclock_s=60)
    result = asyncio.run(agent(problem="P", document=TEX))
    assert result.status == "ready"
    assert agent.tracker.counters.usd == pytest.approx(0.25)
    assert ctx.budgets.root().counters.usd == pytest.approx(0.25)
    assert not (ctx.root_workdir / "cleanup-accounting-uncertain.json").exists()
    with pytest.raises(BudgetExhausted):
        asyncio.run(agent(problem="P", document=TEX))


def test_definitive_spawn_failure_does_not_require_reconciliation(editor, monkeypatch):
    agent, ctx, _ = editor
    original = _MeteredTurn._command_for
    monkeypatch.setattr(_MeteredTurn, "_command_for", lambda *_: ["/nonexistent-cleanup-test-executable"])
    with pytest.raises(mod.SandboxSpawnError):
        asyncio.run(agent(problem="P", document=TEX))
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"] and not state["resumable"]
    assert not (ctx.root_workdir / "cleanup-accounting-uncertain.json").exists()
    monkeypatch.setattr(_MeteredTurn, "_command_for", original)
    assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"


@pytest.mark.parametrize("failure", ["exit", "exception", "cancel"])
def test_fully_billed_editor_failure_can_be_retried(editor, monkeypatch, failure):
    agent, ctx, seen = editor
    command = _MeteredTurn._command_for
    run = _MeteredTurn.run

    def fail_exit(self, inp):
        cmd = command(self, inp)
        cmd[-1] += "sys.exit(1)"
        return cmd

    async def fail_after_billing(self, inp):
        await run(self, inp)
        if failure == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("post-billing failure")

    if failure == "exit":
        monkeypatch.setattr(_MeteredTurn, "_command_for", fail_exit)
    else:
        monkeypatch.setattr(_MeteredTurn, "run", fail_after_billing)
    expected = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected) as caught:
        asyncio.run(agent(problem="P", document=TEX))
    assert not isinstance(caught.value, mod.CleanupAccountingUncertain)
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"] and state["resumable"]
    assert state["recorded_usd"] == pytest.approx(0.25)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    monkeypatch.setattr(_MeteredTurn, "_command_for", command)
    monkeypatch.setattr(_MeteredTurn, "run", run)
    assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"
    assert "--resume" in seen["commands"][-1]
    assert ctx.budgets.root().counters.usd == pytest.approx(0.5)


def test_billed_but_unconfirmed_worker_shutdown_stays_locked(editor, monkeypatch):
    agent, ctx, _ = editor
    run = _MeteredTurn.run

    async def unconfirmed(self, inp):
        out = await run(self, inp)
        self.metering_stream = SimpleNamespace(worker_stopped=False)
        return out

    monkeypatch.setattr(_MeteredTurn, "run", unconfirmed)
    with pytest.raises(mod.CleanupAccountingUncertain):
        asyncio.run(agent(problem="P", document=TEX))
    assert mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_episode_spending_ceiling_survives_repairs_and_agent_recreation(editor):
    agent, ctx, seen = editor
    config = {"cleanup": {"max_invocation_usd": 0.3, "max_episode_usd": 0.4}}
    agent.component_config = config
    ctx.component_configs = {"CleanupSession": config}
    asyncio.run(agent(problem="P", document=TEX))
    other = CleanupSession(ctx)
    # Retain the completed output even when final recorded spend exceeds the cap.
    assert asyncio.run(other(problem="P", document=TEX, findings="Repair notation")).status == "ready"
    caps = [float(cmd[cmd.index("--max-budget-usd") + 1]) for cmd in seen["commands"]]
    assert caps == pytest.approx([0.144, 0.072])
    with pytest.raises(RuntimeError, match="episode spending ceiling"):
        asyncio.run(CleanupSession(ctx)(problem="P", document=TEX))
    assert len(seen["commands"]) == 2
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert state["recorded_usd"] == pytest.approx(0.5)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_episode_ledger_includes_helpers_joined_after_editor_exit(editor, monkeypatch):
    agent, ctx, seen = editor
    agent.component_config = {"cleanup": {"max_invocation_usd": 2}}
    turns = []
    make_turn = agent._turn

    def capture_turn(*args, **kwargs):
        turn = make_turn(*args, **kwargs)
        turns.append(turn)
        return turn

    monkeypatch.setattr(agent, "_turn", capture_turn)

    @asynccontextmanager
    async def late_helper(**callbacks):
        yield {"mcpServers": {}}
        await callbacks["codex_review"]("Review the result")

    monkeypatch.setattr(mod, "cleanup_tools", late_helper)
    asyncio.run(agent(problem="P", document=TEX))
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert ctx.budgets.root().counters.usd > 0.25
    assert state["recorded_usd"] == pytest.approx(ctx.budgets.root().counters.usd)
    assert len(seen["commands"]) == 2 and not state["in_flight"]
    lead, helper = turns
    invocation = lead.tracker.parent
    assert invocation is helper.tracker.parent.parent
    assert invocation.parent is agent.tracker and invocation.spec.max_usd == 1.6
    # A known helper overrun must reduce the lead's live allowance as well.
    helper.tracker.add_usd(0.9)
    assert mod._remaining_usd(lead.tracker) == pytest.approx(1.6 - invocation.counters.usd)


def test_budget_monitor_preserves_exited_worker_while_waiter_drains(editor, monkeypatch):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent

    agent, _, _ = editor
    cfg = {"cmd": ["claude", "-p"], "completion_signal": "exit",
           "usage": {"type": "claude_json", "cost_config": "models/anthropic/fable_51"}}
    turn = agent._turn(cfg, "test-race", 0.1, 60)
    stream = SimpleNamespace(done=True, stdout=json.dumps({"type": "result", "usage": {"input_tokens": 1},
                                                         "total_cost_usd": 0.25}))

    async def terminate():
        pytest.fail("the completed worker must not be killed by the monitor")

    async def wait(*args, **kwargs):
        await asyncio.sleep(1.1)
        return mod.CLIDoneRecord(status="done", summary="completed")

    stream.terminate = terminate
    monkeypatch.setattr(ConfigurableCLIAgent, "_wait_for_done", wait)
    result = asyncio.run(turn._wait_for_done(stream, Path("unused")))
    assert result.status == "done"


def test_shared_budget_is_checked_against_live_unbilled_lead_usage(editor, monkeypatch):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent

    agent, ctx, _ = editor
    cfg = {"cmd": ["claude", "-p"], "completion_signal": "exit",
           "usage": {"type": "claude_json", "auth_mode": "api",
                     "cost_config": "models/anthropic/fable_51"}}
    turn = agent._turn(cfg, "test-editor", 8, 60)
    stream = SimpleNamespace(done=False, stdout=json.dumps({"type": "result", "usage": {"input_tokens": 1},
                                                "total_cost_usd": 0.25}))
    terminated = []

    async def terminate():
        terminated.append(True)

    async def wait(*args, **kwargs):
        await asyncio.Future()

    stream.terminate = terminate
    monkeypatch.setattr(ConfigurableCLIAgent, "_wait_for_done", wait)
    ctx.budgets.root().add_usd(9.9)
    result = asyncio.run(turn._wait_for_done(stream, Path("unused")))
    assert result.status == "error" and terminated == [True]


@pytest.mark.parametrize("kind,has_result,allowed", [
    ("claude_json", True, True), ("claude_json", False, False), ("codex_jsonl", True, False),
])
def test_incomplete_usage_capture_requires_authoritative_invocation_total(editor, monkeypatch, kind, has_result, allowed):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent

    agent, _, _ = editor
    cfg = {"cmd": ["claude", "-p"] if kind == "claude_json" else ["codex", "exec"],
           "usage": {"type": kind, "cost_config": "models/anthropic/fable_51"}}
    turn = agent._turn(cfg, "test-meter", 8, 60)
    turn.metering_stream = SimpleNamespace(usage_capture_dropped_chars=5)
    billed = []

    async def bill(*args):
        billed.append(True)

    monkeypatch.setattr(ConfigurableCLIAgent, "record_cli_usage", bill)
    raw = json.dumps({"type": "result", "usage": {"input_tokens": 1}, "total_cost_usd": 0.25}) if has_result else ""
    if allowed:
        asyncio.run(turn.record_cli_usage(raw, "", None))
        assert turn.metered and billed == [True]
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            asyncio.run(turn.record_cli_usage(raw, "", None))
        assert not turn.metered and not billed


def test_invalid_terminal_usage_survives_compact_capture():
    from proofstack.sandbox.subprocess import _JsonUsageCapture

    capture = _JsonUsageCapture(max_chars=4096, max_line_chars=4096)
    record = json.dumps({"type": "result", "total_cost_usd": "invalid"})
    capture.feed(record + "\n")
    assert '"result"' in capture.text()
    with pytest.raises(ValueError):
        mod.cost_for_claude_usage(capture.text(), cost_config="models/anthropic/fable_51")


def test_stale_completion_does_not_finish_next_turn(editor):
    agent, _, seen = editor
    asyncio.run(agent(problem="P", document=TEX))
    seen["completion"] = False
    with pytest.raises(RuntimeError, match="invocation limit"):
        asyncio.run(agent(problem="P", document=TEX))


def test_separate_helper_pool_and_read_only_review_snapshot(editor):
    agent, ctx, seen = editor

    async def run():
        result = await agent(problem="P", document=TEX, baseline=TEX)
        callbacks = seen["tools"][0]
        report = await callbacks["codex_review"]("Check that the baseline gap remains explicit")
        return result, report

    result, report = asyncio.run(run())
    assert "Independent review" in report["report"]
    assert (result.workspace / report["path"]).read_text() == report["report"]
    cmd = seen["commands"][-1]
    assert cmd[0] == "codex" and "read-only" in cmd
    assert 'forced_login_method="api"' in cmd
    assert "features.shell_tool=false" in cmd
    assert "tools.web_search=true" in cmd
    assert ctx.budgets.root().counters.usd > 0.25
    lead_cmd = seen["commands"][0]
    # 30% reserved for Codex and 10% for the linear reader out of the $10 run budget.
    assert float(lead_cmd[lead_cmd.index("--max-budget-usd") + 1]) == 6


def test_linear_read_tool_bills_its_own_pool_and_retains_report(editor, monkeypatch):
    agent, ctx, seen = editor
    calls = []

    def fake_read(tex, model_ref, *, client, before_batch, on_result):
        before_batch()
        calls.append((tex, model_ref))
        on_result({"cost": 0.4, "input_tokens": 20, "output_tokens": 10})
        return {"report": "# Linear read\n- **mediant** used before defined", "findings": [{"item": "mediant"}],
                "passages": 3, "cost_usd": 0.4, "model": "claude-sonnet-5", "blocks": []}

    monkeypatch.setattr(mod, "_linear_client", lambda _: SimpleNamespace(model="fake", terminated=False))
    monkeypatch.setattr(mod, "_linear_read", fake_read)

    async def run():
        result = await agent(problem="P", document=TEX, baseline=TEX)
        callbacks = seen["tools"][0]
        report = await callbacks["linear_read"]()
        return result, report

    result, report = asyncio.run(run())
    assert calls == [("An honest partial result." and calls[0][0], "models/anthropic/sonnet_5")]
    assert "An honest partial result" in calls[0][0]
    assert report["findings"] == 1 and report["cost_usd"] == 0.4
    assert report["path"].startswith("reviews/linear-")
    assert (result.workspace / report["path"]).read_text() == report["report"]
    assert ctx.budgets.root().counters.usd >= 0.25 + 0.4
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(0.65)


def test_zero_linear_allowance_disables_tool_and_updates_prompt(editor):
    agent, _, seen = editor
    agent.component_config = {"cleanup": {"linear_read_budget_fraction": 0}}
    asyncio.run(agent(problem="P", document=TEX))
    assert seen["tools"][0]["linear_read"] is None
    assert "mcp__cleanup__linear_read" not in seen["commands"][0][seen["commands"][0].index("--allowedTools") + 1]
    assert "linear reader is disabled" in seen["prompts"][0]
    assert seen["tools"][0]["cancel_reviews"] is not None


def test_exhausted_linear_allowance_does_not_create_another_client(editor, monkeypatch):
    agent, ctx, seen = editor
    clients = []

    class Client:
        model = "fake"
        terminated = False

        def terminate(self):
            self.terminated = True

    def make_client(_):
        client = Client()
        clients.append(client)
        return client

    def read(tex, model_ref, *, client, before_batch, on_result):
        before_batch()
        on_result({"cost": 1.0})  # Exactly exhaust the 10% reader allocation.
        with pytest.raises((BudgetExhausted, ValueError)):
            before_batch()
        return {"report": "Paid report", "findings": [], "passages": 1, "cost_usd": 1, "model": "fake"}

    monkeypatch.setattr(mod, "_linear_client", make_client)
    monkeypatch.setattr(mod, "_linear_read", read)

    async def run():
        await agent(problem="P", document=TEX)
        callback = seen["tools"][0]["linear_read"]
        await callback()
        with pytest.raises((BudgetExhausted, ValueError)):
            await callback()

    asyncio.run(run())
    assert len(clients) == 1
    assert ctx.budgets.root().counters.usd == pytest.approx(1.25)


def test_linear_read_cancel_joins_worker_and_keeps_late_cost(editor, monkeypatch):
    agent, ctx, _ = editor
    started, stopped, release, finished = (threading.Event() for _ in range(4))

    class Client:
        model = "fake"
        terminated = False

        def terminate(self):
            self.terminated = True
            stopped.set()

    def read(tex, model_ref, *, client, before_batch, on_result):
        before_batch()
        on_result({"cost": 0.2, "input_tokens": 10})
        started.set()
        assert stopped.wait(5)
        assert release.wait(5)
        on_result({"cost": 0.3, "output_tokens": 20})
        finished.set()
        raise RuntimeError("reader cancelled")

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        task = asyncio.create_task(callbacks["linear_read"]())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            assert await asyncio.to_thread(stopped.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    monkeypatch.setattr(mod, "_linear_client", lambda _: Client())
    monkeypatch.setattr(mod, "_linear_read", read)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    result = asyncio.run(agent(problem="P", document=TEX))
    assert result.status == "ready"
    assert ctx.budgets.root().counters.usd == pytest.approx(0.75)
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(0.75)
    assert ctx.budgets.root().counters.tokens >= 30
    state = json.loads((result.workspace.parent / "session.json").read_text())
    assert state["recorded_usd"] == pytest.approx(0.75) and not state["in_flight"]
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


@pytest.mark.parametrize("unavailable", [False, True])
def test_linear_read_failure_reconciles_unyielded_provider_receipts(editor, monkeypatch, unavailable):
    from mathagents.provider_trace import active_trace

    agent, ctx, _ = editor

    def read(tex, model_ref, **kwargs):
        active_trace.get().update(("attempt", 0), cost=0.2, input_tokens=10,
                                  usage_unavailable=False, model="fake")
        kwargs["on_result"]({"cost": 0.2, "input_tokens": 10})
        # Another parallel passage completed before the iterator raised.
        active_trace.get().update(("attempt", 1), cost=0.3, input_tokens=20,
                                  usage_unavailable=unavailable, model="fake")
        raise RuntimeError("provider failed")

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        with pytest.raises(RuntimeError, match="provider failed"):
            await callbacks["linear_read"]()
        if unavailable:
            with pytest.raises(mod.CleanupAccountingUncertain, match="linear reader usage"):
                await callbacks["linear_read"]()

    monkeypatch.setattr(mod, "_linear_client", lambda _: SimpleNamespace(model="fake", terminated=False))
    monkeypatch.setattr(mod, "_linear_read", read)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    if unavailable:
        with pytest.raises(mod.CleanupAccountingUncertain):
            asyncio.run(agent(problem="P", document=TEX))
    else:
        assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"
    assert ctx.budgets.root().counters.usd == pytest.approx(0.75)
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(0.75)
    assert mod.cleanup_accounting_unresolved(ctx.root_workdir) is unavailable


def test_codex_review_sees_the_research_notes(editor):
    agent, ctx, seen = editor

    async def run():
        await agent(problem="P", document=TEX, baseline=TEX, research_notes="NOTES-MARKER lemma 4")
        await seen["tools"][0]["codex_review"]("Check the proof against the notes")

    asyncio.run(run())
    prompts = [p.read_text() for p in ctx.root_workdir.rglob("stdin.txt")]
    assert prompts and all("--- research-notes.md ---\nNOTES-MARKER lemma 4" in p for p in prompts)


def _capture_events(agent, monkeypatch):
    events = []
    emit = agent.events.emit

    async def capture(kind, data=None, **kwargs):
        events.append((kind, data))
        return await emit(kind, data, **kwargs)

    monkeypatch.setattr(agent.events, "emit", capture)
    return events


def test_late_helper_without_usage_is_billed_its_full_allowance(editor, monkeypatch):
    agent, ctx, _ = editor
    events = _capture_events(agent, monkeypatch)
    command = _MeteredTurn._command_for

    def helper_without_usage(self, inp):
        if self.component_config["usage"]["type"] == "codex_jsonl":
            return [sys.executable, "-c", "print('no usage')"]
        return command(self, inp)

    @asynccontextmanager
    async def late_helper(**callbacks):
        yield {"mcpServers": {}}
        with pytest.raises(RuntimeError, match="accounting failed"):
            await callbacks["codex_review"]("Review current proof")

    monkeypatch.setattr(_MeteredTurn, "_command_for", helper_without_usage)
    monkeypatch.setattr(mod, "cleanup_tools", late_helper)
    result = asyncio.run(agent(problem="P", document=TEX))
    assert result.status == "ready"
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"] and not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    # $10 invocation, 30% Codex pool.
    assert ctx.budgets.root().counters.usd == pytest.approx(0.25 + 3)
    assert state["recorded_usd"] == pytest.approx(0.25 + 3)
    [estimate] = [data for kind, data in events if kind == "cleanup.helper_usage_estimated"]
    assert estimate["charged_usd"] == pytest.approx(3) and estimate["recorded_usd"] == 0
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(3.25)
    rows = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    [debit] = [row for row in rows if row["kind"] == "model.call" and row["payload"].get("cost_estimated")]
    assert debit["payload"]["cost_usd"] == 3
    assert debit["payload"]["usage_unavailable"] is True
    assert debit["call_id"] == f"cleanup-estimate:{estimate['review_id']}"

    resumed_ctx = RunContext.create(run_id="cleanup", root_workdir=ctx.root_workdir, flat=True,
                                   run_budget=BudgetSpec(max_usd=10, max_wallclock_s=60))
    for _ in range(2):
        asyncio.run(ACWorkflow(resumed_ctx)._apply_resume_budget_offset())
        assert resumed_ctx.budgets.root().counters.usd == pytest.approx(3.25)


@pytest.mark.parametrize("metered", [False, True])
def test_unconfirmed_helper_shutdown_blocks_publication_and_new_work(editor, monkeypatch, metered):
    from proofstack.kinds.cli import _WorkerStopUnconfirmed
    from proofstack.sandbox.base import WorkerStopState

    agent, ctx, seen = editor
    run = _MeteredTurn.run

    async def unconfirmed_helper(self, inp):
        out = await run(self, inp)
        if self.component_config["usage"]["type"] == "codex_jsonl":
            self.metered = metered
            self.metering_stream = SimpleNamespace(worker_stopped=False)
            raise _WorkerStopUnconfirmed(WorkerStopState.UNKNOWN, phase="test")
        return out

    @asynccontextmanager
    async def late_helper(**callbacks):
        yield {"mcpServers": {}}
        with pytest.raises(_WorkerStopUnconfirmed):
            await callbacks["codex_review"]("Review current proof")
        with pytest.raises(mod.CleanupAccountingUncertain):
            await callbacks["codex_review"]("Retry review")

    monkeypatch.setattr(_MeteredTurn, "run", unconfirmed_helper)
    monkeypatch.setattr(mod, "cleanup_tools", late_helper)
    with pytest.raises(mod.CleanupAccountingUncertain, match="helper shutdown or usage"):
        asyncio.run(agent(problem="P", document=TEX))
    assert mod.cleanup_accounting_unresolved(ctx.root_workdir)
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert state["in_flight"]
    assert sum(cmd[0] == "codex" for cmd in seen["commands"]) == 1
    with pytest.raises(mod.CleanupAccountingUncertain):
        asyncio.run(agent(problem="P", document=TEX, session_key="another-episode"))


def test_failed_helper_estimate_persistence_keeps_session_locked(editor, monkeypatch):
    agent, ctx, _ = editor
    command = _MeteredTurn._command_for
    write = agent.events.sink.write

    async def fail_estimate(record):
        if record["kind"] == "model.call" and record["payload"].get("cost_estimated"):
            raise OSError("cannot persist estimate")
        await write(record)

    def helper_without_usage(self, inp):
        if self.component_config["usage"]["type"] == "codex_jsonl":
            return [sys.executable, "-c", "print('no usage')"]
        return command(self, inp)

    @asynccontextmanager
    async def late_helper(**callbacks):
        yield {"mcpServers": {}}
        with pytest.raises(OSError, match="cannot persist estimate"):
            await callbacks["codex_review"]("Review current proof")

    monkeypatch.setattr(agent.events.sink, "write", fail_estimate)
    monkeypatch.setattr(_MeteredTurn, "_command_for", helper_without_usage)
    monkeypatch.setattr(mod, "cleanup_tools", late_helper)
    with pytest.raises(mod.CleanupAccountingUncertain):
        asyncio.run(agent(problem="P", document=TEX))
    assert mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_review_cancelled_at_shutdown_is_billed_its_full_allowance(editor, monkeypatch):
    agent, ctx, _ = editor
    events = _capture_events(agent, monkeypatch)
    command = _MeteredTurn._command_for
    helpers = []
    make_turn = agent._turn

    def capture_turn(cfg, name, *args, **kwargs):
        turn = make_turn(cfg, name, *args, **kwargs)
        if name.startswith("cleanup-codex-"):
            helpers.append(turn)
        return turn

    def slow_helper(self, inp):
        if self.component_config["usage"]["type"] == "codex_jsonl":
            return [sys.executable, "-c",
                    "import time; from pathlib import Path; Path('started').write_text('1'); time.sleep(60)"]
        return command(self, inp)

    @asynccontextmanager
    async def late_helper(**callbacks):
        yield {"mcpServers": {}}
        review = asyncio.create_task(callbacks["codex_review"]("Review current proof"))
        root = ctx.root_workdir / "cleanup_sessions/standalone/codex"
        for _ in range(500):
            if list(root.glob("*/started")):
                break
            await asyncio.sleep(0.02)
        assert list(root.glob("*/started"))
        # As cleanup_tools' shutdown does: cancel, then join and discard the error.
        review.cancel()
        [error] = await asyncio.gather(review, return_exceptions=True)
        assert isinstance(error, (asyncio.CancelledError, RuntimeError))

    monkeypatch.setattr(agent, "_turn", capture_turn)
    monkeypatch.setattr(_MeteredTurn, "_command_for", slow_helper)
    monkeypatch.setattr(mod, "cleanup_tools", late_helper)
    started = time.monotonic()
    result = asyncio.run(agent(problem="P", document=TEX))
    assert time.monotonic() - started < 30
    assert isinstance(result, CleanupSession.Outputs) and result.status == "ready"
    [helper] = helpers
    assert not helper.metered
    assert helper.metering_stream.worker_stopped
    assert helper.tracker.counters.usd == pytest.approx(3)
    assert helper.tracker.parent.counters.usd == pytest.approx(3)
    assert ctx.budgets.root().counters.usd == pytest.approx(0.25 + 3)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(3.25)
    [estimate] = [data for kind, data in events if kind == "cleanup.helper_usage_estimated"]
    assert estimate["charged_usd"] == pytest.approx(3) and estimate["recorded_usd"] == 0


def test_safe_read_rejects_symlinks_hardlinks_and_large_files(tmp_path):
    source = tmp_path / "source"
    source.write_text("x")
    (tmp_path / "link").symlink_to(source)
    with pytest.raises(OSError):
        _read_file(tmp_path, "link")
    os.link(source, tmp_path / "hard")
    with pytest.raises(ValueError, match="unsafe"):
        _read_file(tmp_path, "hard")
    (tmp_path / "large").write_bytes(b"x" * (mod.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="oversized"):
        _read_file(tmp_path, "large")


def test_paid_preflight_rejects_missing_key_without_spawn(tmp_path, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(mod.CleanupUnavailable, match="ANTHROPIC_API_KEY"):
        asyncio.run(CleanupSession(ctx)._preflight(CleanupSettings()))


@pytest.mark.parametrize("error", [FileNotFoundError("claude"), RuntimeError("unsupported CLI"),
                                   mod.subprocess.TimeoutExpired("claude", 20)])
def test_runtime_preflight_classifies_unavailable_cli_without_paid_calls(monkeypatch, error):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def unavailable(**kwargs):
        raise error

    monkeypatch.setattr(mod, "check_cleanup_runtime", unavailable)
    with pytest.raises(mod.CleanupUnavailable):
        asyncio.run(mod.check_cleanup_available(CleanupSettings()))


def test_preflight_without_codex_does_not_require_openai_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(mod, "check_cleanup_runtime", lambda **kwargs: called.append(kwargs))
    asyncio.run(mod.check_cleanup_available(CleanupSettings(codex_budget_fraction=0)))
    assert called == [{"require_codex": False}]
    with pytest.raises(mod.CleanupUnavailable, match="OPENAI_API_KEY"):
        asyncio.run(mod.check_cleanup_available(CleanupSettings()))
    assert len(called) == 1


def test_settings_and_preset():
    preset = load_preset("cleanup_session")
    assert preset.workflow_cls is CleanupSession
    assert preset.inputs["partial"] is True
    assert preset.component_configs["CleanupSession"]["cleanup"]["effort"] == "xhigh"
    assert CleanupSettings().max_invocations == 3
    assert CleanupSettings().max_invocation_usd == 150
    assert CleanupSettings().max_episode_usd == 500
    for name in ("max_invocation_usd", "max_episode_usd"):
        for value in (-1, 0, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                CleanupSettings(**{name: value})
    for value in (-1, 1, float("nan")):
        with pytest.raises(ValueError):
            CleanupSettings(codex_budget_fraction=value)
    for value in (-1, 1, float("nan")):
        with pytest.raises(ValueError):
            CleanupSettings(linear_read_budget_fraction=value)
    assert CleanupSettings().linear_read_model == "models/anthropic/sonnet_5"


@pytest.mark.parametrize("workflow", ["cleanup_session", "firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_workflow_cleanup_episode_defaults(workflow):
    preset = load_preset(workflow)
    settings = CleanupSettings.model_validate(preset.component_configs.get("CleanupSession", {}).get("cleanup", {}))
    assert settings.max_episode_usd == 500
    assert settings.max_invocation_usd == 150


def test_mcp_transport_auth_edit_and_lifecycle():
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from proofstack.agents.cleanup_tools import cleanup_tools

    data = {"answer.tex": "Old passage"}

    async def read_file(path):
        return {"content": data[path]}

    async def write_file(path, content):
        data[path] = content
        return {"path": path}

    async def list_files():
        return list(data)

    async def compile_document():
        return {"compiled": True, "pages": 1}

    async def review(task):
        return {"report": task}

    async def run():
        async with cleanup_tools(compile_document=compile_document, codex_review=review,
                                 read_file=read_file, write_file=write_file, list_files=list_files,
                                 progress=lambda: {"stage": "finishing", "seconds_remaining": 20}) as cfg:
            server = cfg["mcpServers"]["cleanup"]
            assert server["timeout"] == 3_600_000
            async with httpx.AsyncClient() as client:
                assert (await client.get(server["url"])).status_code == 401
            async with httpx.AsyncClient(headers=server["headers"]) as client:
                async with streamable_http_client(server["url"], http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        names = {tool.name for tool in (await session.list_tools()).tools}
                        assert names == {"read", "write", "edit", "files", "compile", "review"}
                        result = await session.call_tool("edit", {"path": "answer.tex", "old": "Old", "new": "New"})
                        assert not result.isError
                        assert data["answer.tex"] == "New passage"
                        result = await session.call_tool("edit", {"path": "answer.tex", "old": "Old", "new": "New"})
                        assert result.isError

    asyncio.run(run())


def test_mcp_read_pages_large_files_and_cuts_long_reports():
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from proofstack.agents.cleanup_tools import MAX_READ_CHARS, READ_CHUNK_CHARS, cleanup_tools

    notes = "".join(chr(0x41 + i % 26) for i in range(2 * READ_CHUNK_CHARS + 123)) + "$\\mathcal{M}$ tail"
    long_report = "r" * (READ_CHUNK_CHARS + 5)
    data = {"research-notes.md": notes, "reviews/x.md": long_report, "answer.tex": "Old passage"}

    async def read_file(path):
        return {"content": data[path], "sha256": "full"}

    async def write_file(path, content):
        data[path] = content
        return {"path": path}

    async def list_files():
        return list(data)

    async def compile_document():
        return {"compiled": True, "pages": 1}

    async def review(task):
        return {"report": long_report, "path": "reviews/x.md"}

    async def run():
        async with cleanup_tools(compile_document=compile_document, codex_review=review,
                                 read_file=read_file, write_file=write_file, list_files=list_files,
                                 progress=lambda: {"stage": "finishing", "seconds_remaining": 20}) as cfg:
            server = cfg["mcpServers"]["cleanup"]
            async with httpx.AsyncClient(headers=server["headers"]) as client:
                async with streamable_http_client(server["url"], http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                        assert tools["read"].meta == {"anthropic/maxResultSizeChars": 200_000}
                        assert {"offset", "limit"} <= set(tools["read"].inputSchema["properties"])

                        async def call(name, args):
                            result = await session.call_tool(name, args)
                            assert not result.isError, result
                            # Claude Code sees this text and diverts it above 50,000 characters.
                            assert len(result.content[0].text) < 50_000
                            payload = json.loads(result.content[0].text)
                            assert payload["cleanup_control"] == {"stage": "finishing", "seconds_remaining": 20}
                            return payload

                        pieces, offset = [], 0
                        while offset is not None:
                            page = await call("read", {"path": "research-notes.md", "offset": offset})
                            assert page["length"] == len(notes) and page["sha256"] == "full"
                            assert ("note" in page) == (page["next_offset"] is not None)
                            pieces.append(page["content"])
                            offset = page["next_offset"]
                        assert len(pieces) == 3 and "".join(pieces) == notes

                        page = await call("read", {"path": "research-notes.md", "offset": 10, "limit": 5})
                        assert page["content"] == notes[10:15] and page["next_offset"] == 15
                        result = await session.call_tool("read", {"path": "research-notes.md", "limit": 10**9})
                        assert len(json.loads(result.content[0].text)["content"]) == min(len(notes), MAX_READ_CHARS)
                        page = await call("read", {"path": "answer.tex"})
                        assert page["content"] == "Old passage" and page["next_offset"] is None
                        assert "note" not in page
                        assert (await session.call_tool("read", {"path": "answer.tex", "offset": -1})).isError

                        report = await call("review", {"task": "check"})
                        assert report["report"] == long_report[:READ_CHUNK_CHARS]
                        assert report["report_length"] == len(long_report)
                        assert f"offset={READ_CHUNK_CHARS}" in report["note"]

                        result = await call("edit", {"path": "research-notes.md", "old": "M}$ tail", "new": "N}$"})
                        assert data["research-notes.md"] == notes.replace("M}$ tail", "N}$")

    asyncio.run(run())


def test_mcp_shutdown_cancels_codex_before_draining_linear_reader(monkeypatch, tmp_path):
    from proofstack.agents import cleanup_tools as transport

    methods = {}

    class MCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, **kwargs):
            def register(fn):
                methods[fn.__name__] = fn
                return fn
            return register

        def streamable_http_app(self):
            return None

    async def run():
        started, draining, release, stopped = [asyncio.Event() for _ in range(4)]
        codex_started, codex_cancelled = asyncio.Event(), asyncio.Event()
        completed = []
        calls = []

        class Server:
            def __init__(self, cfg):
                self.started = self.should_exit = False

            async def serve(self, **kwargs):
                self.started = True
                try:
                    while not self.should_exit:
                        await asyncio.sleep(0.01)
                finally:
                    stopped.set()

        async def linear_reader():
            started.set()
            try:
                await asyncio.Future()
            finally:
                draining.set()
                await release.wait()
                completed.append("linear")

        async def codex(*args, **kwargs):
            codex_started.set()
            try:
                await asyncio.Future()
            finally:
                codex_cancelled.set()
                await release.wait()
                completed.append("codex")

        jobs = mod.CleanupReviews(tmp_path, tmp_path, snapshot=lambda: {}, execute=codex, read_text=_read_file)

        async def unused(*args):
            return {}

        async def owner():
            async with jobs, transport.cleanup_tools(compile_document=unused, codex_review=unused,
                                               read_file=unused, write_file=unused, list_files=unused,
                                               linear_read=linear_reader, cancel_reviews=jobs.cancel):
                await jobs.start("Check citations", "attribution")
                await codex_started.wait()
                calls.append(asyncio.create_task(methods["prefix_read"]()))
                await asyncio.Future()

        monkeypatch.setattr(transport, "FastMCP", MCP)
        monkeypatch.setattr(transport, "_Server", Server)
        task = asyncio.create_task(owner())
        try:
            await asyncio.wait_for(started.wait(), 3)
            task.cancel()
            await asyncio.wait_for(draining.wait(), 3)
            # Neither helper has finished draining. Codex must already have
            # received cancellation, not continue inference during that wait.
            await asyncio.wait_for(codex_cancelled.wait(), 3)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done() and not stopped.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert sorted(completed) == ["codex", "linear"] and stopped.is_set()
            assert all(call.done() for call in calls)
            assert all(job.done() for job in jobs.tasks.values())
            assert all(record["status"] == "cancelled" for record in jobs.records.values())
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, *calls, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="TeX tools not installed")
def test_editor_compile_scrubs_environment_and_forbids_external_inputs(tmp_path, monkeypatch):
    from proofstack.agents import writeup_loop

    observed = []
    popen = writeup_loop.subprocess.Popen

    def inspect(*args, **kwargs):
        observed.append((args[0], kwargs["env"]))
        return popen(*args, **kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-tex")
    monkeypatch.setattr(writeup_loop.subprocess, "Popen", inspect)
    result = writeup_loop._compile_raw(TEX, None, secure=True, deadline=time.monotonic() + 30)
    assert result[:2] == (True, 1), result
    assert all("ANTHROPIC_API_KEY" not in env for _, env in observed)
    assert all("-no-shell-escape" in cmd for cmd, _ in observed if cmd[0] == "pdflatex")
    outside = tmp_path / "outside.tex"
    outside.write_text("External content")
    tex = TEX.replace("An honest partial result.", r"\input{" + str(outside) + "}")
    result = writeup_loop._compile_raw(tex, None, secure=True, deadline=time.monotonic() + 30)
    assert not result[0]


def test_finished_review_that_overshoots_its_allowance_is_still_returned(editor, monkeypatch):
    agent, ctx, _ = editor
    command = _MeteredTurn._command_for

    def expensive_review(self, inp):
        if self.component_config["usage"]["type"] == "codex_jsonl":
            return [sys.executable, "-c", (
                "import sys,json; from pathlib import Path; sys.stdin.read(); "
                "Path('report.md').write_text('Paid-for review'); "
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10_000_000,'output_tokens':200_000}}))")]
        return command(self, inp)

    async def run():
        result = await agent(problem="P", document=TEX, baseline=TEX)
        callbacks = ctx_tools[0]
        report = await callbacks["codex_review"]("Review")
        with pytest.raises(Exception, match="(?i)budget"):
            await callbacks["codex_review"]("Review again")
        return result, report

    ctx_tools = []

    @asynccontextmanager
    async def tools(**callbacks):
        ctx_tools.append(callbacks)
        yield {"mcpServers": {}}

    monkeypatch.setattr(_MeteredTurn, "_command_for", expensive_review)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    result, report = asyncio.run(run())
    assert report["report"] == "Paid-for review"
    assert (result.workspace / report["path"]).read_text() == "Paid-for review"
    # The overshoot is recorded, not hidden: more than the $3 Codex pool.
    assert ctx.budgets.root().counters.usd > 0.25 + 3


@pytest.mark.parametrize("review_state", ["missing", "unread", "general"])
@pytest.mark.parametrize("partial", [False, True])
def test_ready_requires_retrieved_attribution_report(editor, monkeypatch, review_state, partial):
    agent, ctx, _ = editor
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        if review_state != "missing":
            job = await callbacks["review_start"]("Check", "general" if review_state == "general" else "attribution")
            await callbacks["review_start"].__self__.tasks[job["review_id"]]
            if review_state == "general":
                await callbacks["review_status"](job["review_id"])

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    with pytest.raises(mod.CleanupIncomplete, match="attribution review") as error:
        asyncio.run(agent(problem="P", document=TEX, partial=partial))
    assert error.value.output.answer_tex == TEX
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_background_review_recovers_lost_reply_and_is_charged_once(editor, monkeypatch):
    agent, ctx, seen = editor
    monkeypatch.setattr(mod.CleanupReviews, "attribution_retrieved", _attribution_retrieved)

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        await callbacks["review_start"]("Verify attribution", "attribution")  # reply lost
        [job] = (await callbacks["review_status"]())["reviews"]
        repeated = await callbacks["review_start"]("Verify attribution", "attribution")
        assert repeated["review_id"] == job["review_id"]
        report = await callbacks["review_status"](job["review_id"], 20)
        assert report["status"] == "completed" and "Independent review" in report["report"]
        assert (await callbacks["review_status"](job["review_id"]))["report"] == report["report"]

    monkeypatch.setattr(mod, "cleanup_tools", tools)
    result = asyncio.run(agent(problem="P", document=TEX))
    assert result.status == "ready"
    assert sum(cmd[0] == "codex" for cmd in seen["commands"]) == 1
    rows = [json.loads(s) for s in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    costs = [r["payload"]["cost_usd"] for r in rows if r["kind"] == "model.call"]
    assert len(costs) == 2 and ctx.budgets.root().counters.usd == pytest.approx(sum(costs))
    assert "mcp__cleanup__review_status" in seen["commands"][0][seen["commands"][0].index("--allowedTools") + 1]


def test_async_job_shutdown_joins_and_bills_real_dummy_worker(editor, monkeypatch):
    agent, ctx, _ = editor
    command = _MeteredTurn._command_for

    def slow_helper(self, inp):
        if self.component_config["usage"]["type"] == "codex_jsonl":
            return [sys.executable, "-c",
                    "import time; from pathlib import Path; Path('started').write_text('1'); time.sleep(60)"]
        return command(self, inp)

    @asynccontextmanager
    async def tools(**callbacks):
        yield {"mcpServers": {}}
        await callbacks["review_start"]("Check", "attribution")
        async with asyncio.timeout(10):
            while not list((ctx.root_workdir / "cleanup_sessions/standalone/codex").glob("*/started")):
                await asyncio.sleep(0.02)

    monkeypatch.setattr(_MeteredTurn, "_command_for", slow_helper)
    monkeypatch.setattr(mod, "cleanup_tools", tools)
    asyncio.run(agent(problem="P", document=TEX))
    assert ctx.budgets.root().counters.usd == pytest.approx(3.25)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    records = list((ctx.root_workdir / "cleanup_sessions/standalone/review_jobs").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] in {"cancelled", "failed"}


def _edit_then(tail):
    return ("import sys,json,time; from pathlib import Path; sys.stdin.read(); "
            "Path('answer.tex').write_text(Path('answer.tex').read_text().replace('honest','EDITED honest')); " + tail)


def _assert_settled(ctx, estimated):
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"]
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    assert state["estimated_usd"] == pytest.approx(estimated)
    assert state["recorded_usd"] == pytest.approx(ctx.budgets.root().counters.usd)
    return state


@pytest.mark.parametrize("stop", ["deadline", "monitor", "sigkill"])
def test_harness_stopped_editor_is_billed_its_allowance_and_keeps_manuscript(editor, monkeypatch, stop):
    agent, ctx, seen = editor
    events = _capture_events(agent, monkeypatch)
    if stop == "deadline":
        agent.tracker.spec = BudgetSpec(max_usd=10, max_wallclock_s=4)
        tail = "time.sleep(30)"
    elif stop == "monitor":
        agent.component_config = {"cleanup": {"max_invocation_usd": 1}}
        tail = ("print(json.dumps({'type':'assistant','message':{'model':'claude-fable-5-1','id':'m1',"
                "'usage':{'input_tokens':100,'output_tokens':1000000}}}), flush=True); time.sleep(30)")
    else:
        tail = "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"
    monkeypatch.setattr(_MeteredTurn, "_command_for", lambda *_: [sys.executable, "-c", _edit_then(tail)])
    with pytest.raises(mod.CleanupIncomplete) as caught:
        asyncio.run(agent(problem="P", document=TEX))
    output = caught.value.output
    assert output.status == "incomplete" and "EDITED honest" in output.answer_tex
    [estimate] = [data for kind, data in events if kind == "cleanup.editor_usage_estimated"]
    lead = 0.6 if stop == "monitor" else 6
    if stop == "monitor":
        # The streamed lower bound already exceeds the allowance.
        assert estimate["charged_usd"] == 0 and estimate["recorded_usd"] > lead
    else:
        assert estimate["charged_usd"] == pytest.approx(lead)
        assert ctx.budgets.root().counters.usd == pytest.approx(lead)
    state = _assert_settled(ctx, estimate["charged_usd"])
    # No transcript was written, so the next invocation must not --resume.
    assert not state["resumable"]
    # The estimate is in the model-call ledger that resume and usage exports read.
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == pytest.approx(ctx.budgets.root().counters.usd)


def test_outer_cancel_of_running_editor_settles_accounting(editor, monkeypatch):
    agent, ctx, _ = editor
    events = _capture_events(agent, monkeypatch)
    monkeypatch.setattr(_MeteredTurn, "_command_for",
                        lambda *_: [sys.executable, "-c", _edit_then("time.sleep(30)")])

    async def run():
        await asyncio.wait_for(agent(problem="P", document=TEX), timeout=2)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    [estimate] = [data for kind, data in events if kind == "cleanup.editor_usage_estimated"]
    assert estimate["charged_usd"] == pytest.approx(6)
    _assert_settled(ctx, 6)
    assert "EDITED" in (ctx.root_workdir / "cleanup_sessions/standalone/workspace/answer.tex").read_text()


def test_editor_transcript_makes_estimated_session_resumable(editor, monkeypatch):
    agent, ctx, seen = editor
    command = _MeteredTurn._command_for
    tail = ("p=Path('.claude/projects/x'); p.mkdir(parents=True); "
            "(p/(sys.argv[1]+'.jsonl')).write_text('{}'); sys.exit(3)")

    def transcript_then_crash(self, inp):
        cmd = self.component_config["cmd"]
        return [sys.executable, "-c", _edit_then(tail), cmd[-1]]

    monkeypatch.setattr(_MeteredTurn, "_command_for", transcript_then_crash)
    with pytest.raises(mod.CleanupIncomplete):
        asyncio.run(agent(problem="P", document=TEX))
    state = _assert_settled(ctx, 6)
    assert state["resumable"]
    monkeypatch.setattr(_MeteredTurn, "_command_for", command)
    assert asyncio.run(CleanupSession(ctx)(problem="P", document=TEX)).status == "ready"
    assert "--resume" in seen["commands"][-1]


def test_budget_exit_returns_edited_manuscript_as_incomplete(editor, monkeypatch):
    agent, ctx, _ = editor
    tail = ("print(json.dumps({'type':'result','subtype':'error_max_budget_usd','is_error':True,'session_id':'t',"
            "'usage':{'input_tokens':100,'output_tokens':20},'num_turns':1,'total_cost_usd':0.25})); sys.exit(1)")
    monkeypatch.setattr(_MeteredTurn, "_command_for", lambda *_: [sys.executable, "-c", _edit_then(tail)])
    with pytest.raises(mod.CleanupIncomplete, match="editor failed") as caught:
        asyncio.run(agent(problem="P", document=TEX))
    assert "EDITED honest" in caught.value.output.answer_tex
    assert caught.value.output.status == "incomplete"
    _assert_settled(ctx, 0)


def _queued_sandbox(monkeypatch, registry):
    from proofstack.sandbox.memory import MemoryPolicy

    def sandbox(self, settings, seconds, provider):
        return {"backend": "subprocess", "memory_gb": 1, "limit_address_space": False,
                "timeout_s": max(1, int(seconds)),
                "memory_policy": MemoryPolicy(registry=registry, max_workers=1, worker_bytes=1024, reserve_bytes=0),
                "provider_keys": [provider], "env_allowlist": ["PATH"]}
    monkeypatch.setattr(CleanupSession, "_sandbox", sandbox)


def test_cancel_while_queued_for_admission_is_not_started(editor, monkeypatch, tmp_path):
    from proofstack.sandbox import memory

    agent, ctx, _ = editor
    registry = tmp_path / "registry"
    registry.mkdir()
    _queued_sandbox(monkeypatch, registry)
    monkeypatch.setattr(memory, "available_memory", lambda: memory.GiB)
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        marker["token"]: (0, False) for marker in markers
    })
    owner = memory.MemoryLease(memory.MemoryPolicy(registry, 1, 1024, 0),
                               SimpleNamespace(token="occupant", created_at=time.time()))
    assert owner.try_acquire()
    try:
        async def run():
            await asyncio.wait_for(agent(problem="P", document=TEX), timeout=2)

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(run())
    finally:
        owner.close()
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"] and not state["resumable"]
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)
    assert ctx.budgets.root().counters.usd == 0


def test_partial_cleanup_editor_cannot_write_inherited_memory_metadata(editor, monkeypatch, tmp_path):
    from proofstack.sandbox import memory

    agent, ctx, _ = editor
    registry = tmp_path / "registry"
    _queued_sandbox(monkeypatch, registry)
    monkeypatch.setattr(memory, "available_memory", lambda: memory.GiB)
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        marker["token"]: (0, False) for marker in markers
    })
    command = _MeteredTurn._command_for

    def guarded_command(self, inp):
        cmd = command(self, inp)
        # Find the inherited slot by inode, then simulate an accidental write.
        # The rest of the fake editor still completes and reports usage.
        probe = f"""
import errno, os
lock = os.stat({str(registry / 'slot-0.lock')!r})
found = False
for fd in map(int, os.listdir('/dev/fd')):
    try:
        info = os.fstat(fd)
    except OSError:
        continue
    if (info.st_dev, info.st_ino) != (lock.st_dev, lock.st_ino):
        continue
    found = True
    try:
        os.write(fd, bytes([255, 0, 128]))
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        raise AssertionError('cleanup can corrupt shared accounting')
assert found, 'cleanup did not inherit its lifetime lock'
"""
        return cmd[:2] + [probe + "\n" + cmd[2]]

    monkeypatch.setattr(_MeteredTurn, "_command_for", guarded_command)
    result = asyncio.run(agent(problem="P", document=TEX, baseline=TEX, partial=True))
    assert result.status == "ready"
    assert json.loads((registry / "slot-0.json").read_text())["token"]
    assert (registry / "slot-0.lock").stat().st_size == 0
    assert ctx.budgets.root().counters.usd == pytest.approx(0.25)
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_admission_error_before_launch_is_not_started_and_review_is_unbilled(editor, monkeypatch, tmp_path):
    from proofstack.sandbox import memory

    agent, ctx, _ = editor
    events = _capture_events(agent, monkeypatch)
    _queued_sandbox(monkeypatch, tmp_path / "registry")

    def mismatch(self):
        raise RuntimeError("memory registry policy mismatch")

    monkeypatch.setattr(memory.MemoryLease, "try_acquire", mismatch)

    @asynccontextmanager
    async def review_first(**callbacks):
        with pytest.raises(RuntimeError, match="mismatch"):
            await callbacks["codex_review"]("Review")
        yield {"mcpServers": {}}

    monkeypatch.setattr(mod, "cleanup_tools", review_first)
    with pytest.raises(RuntimeError, match="mismatch"):
        asyncio.run(agent(problem="P", document=TEX))
    assert not [kind for kind, _ in events if kind.endswith("usage_estimated")]
    assert ctx.budgets.root().counters.usd == 0
    state = json.loads((ctx.root_workdir / "cleanup_sessions/standalone/session.json").read_text())
    assert not state["in_flight"] and not state["resumable"]
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


@pytest.mark.parametrize("completion", ["[]", '"ready"', "{}", '{"status":"done"}', '{"status":"ready","summary":3}', "{"])
def test_malformed_completion_is_treated_as_missing(editor, monkeypatch, completion):
    agent, ctx, seen = editor
    seen["completion"] = False
    command = _MeteredTurn._command_for

    def malformed(self, inp):
        cmd = command(self, inp)
        cmd[-1] += f"Path('completion.json').write_text({completion!r}); "
        return cmd

    monkeypatch.setattr(_MeteredTurn, "_command_for", malformed)
    with pytest.raises(RuntimeError, match="invocation limit"):
        asyncio.run(agent(problem="P", document=TEX))
    assert len(seen["commands"]) == 3
    assert not mod.cleanup_accounting_unresolved(ctx.root_workdir)


def test_monitor_leaves_lead_cap_to_claude_and_stops_cleanly_on_exhausted_pool(editor, monkeypatch):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent

    agent, ctx, _ = editor
    cfg = {"cmd": ["claude", "-p"], "completion_signal": "exit",
           "usage": {"type": "claude_json", "auth_mode": "api", "cost_config": "models/anthropic/fable_51"}}
    terminated = []

    async def terminate():
        terminated.append(True)

    async def wait(*args, **kwargs):
        await asyncio.sleep(1.5)
        return mod.CLIDoneRecord(status="done", summary="completed")

    monkeypatch.setattr(ConfigurableCLIAgent, "_wait_for_done", wait)
    # Claude stops itself at --max-budget-usd; the monitor must not race it.
    turn = agent._turn(cfg, "test-editor", 0.2, 60)
    turn.live_usd_limit = 0.2 * mod.LEAD_MONITOR_MARGIN
    stream = SimpleNamespace(done=False, terminate=terminate, stdout=json.dumps(
        {"type": "result", "usage": {"input_tokens": 1}, "total_cost_usd": 0.22}))
    assert asyncio.run(turn._wait_for_done(stream, Path("unused"))).status == "done"
    assert not terminated
    # An overspent shared pool stops the worker instead of raising ValueError.
    ctx.budgets.root().add_usd(10.5)
    turn = agent._turn(cfg, "test-editor-2", 8, 60)
    assert asyncio.run(turn._wait_for_done(stream, Path("unused"))).status == "error"
    assert terminated == [True]


def test_resumed_invocations_are_charged_only_their_own_spend(editor, monkeypatch):
    agent, ctx, seen = editor
    costs = iter([0.1, 0.3, 0.2])

    def cumulative(self, inp):
        cmd = self.component_config["cmd"]
        seen["commands"].append(cmd)
        # The first invocation stops without completion.json and is continued.
        done = "" if len(seen["commands"]) == 1 else \
            "Path('completion.json').write_text(json.dumps({'status':'ready'})); "
        return [sys.executable, "-c", "import sys; from pathlib import Path; sys.stdin.read(); "
                + _claude_session_tail(cmd, next(costs)) + done]

    monkeypatch.setattr(_MeteredTurn, "_command_for", cumulative)
    assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"
    assert ctx.budgets.root().counters.usd == pytest.approx(0.4)
    assert asyncio.run(CleanupSession(ctx)(problem="P", document=TEX, findings="Repair")).status == "ready"
    assert ["--resume" in cmd for cmd in seen["commands"]] == [False, True, True]
    root = ctx.budgets.root().counters
    assert root.usd == pytest.approx(0.6)
    assert root.tokens == 3 * 120
    state = _assert_settled(ctx, 0)
    assert state["claude_session_usd"] == pytest.approx(0.6) and state["claude_session_tokens"] == 360


@pytest.mark.parametrize("stop,saved,estimate", [
    ("sigterm", 1.0, 5.85), ("sigterm", 7.0, 7.0), ("sigkill", None, 5.85)])
def test_resume_after_harness_stop_starts_from_claudes_saved_total(editor, monkeypatch, stop, saved, estimate):
    agent, ctx, seen = editor
    command = _MeteredTurn._command_for
    assert asyncio.run(agent(problem="P", document=TEX)).status == "ready"

    def stopped(self, inp):
        cmd = self.component_config["cmd"]
        # Claude saves its cumulative total when SIGTERM lets it exit; a
        # worker that ignores SIGTERM is SIGKILLed after the grace period.
        on_term = (_claude_session_tail(cmd, saved, result=False) + "os._exit(143)"
                   if stop == "sigterm" else "pass")
        return [sys.executable, "-c", _edit_then(
            "import signal,os\ndef term(*_):\n    " + on_term.replace("; ", "\n    ") + "\n"
            "signal.signal(signal.SIGTERM, term if " + repr(stop == "sigterm") + " else signal.SIG_IGN)\n"
            "print(json.dumps({'type':'assistant','message':{'model':'claude-fable-5-1','id':'m1',"
            "'usage':{'input_tokens':100,'output_tokens':20}}}), flush=True)\n"
            "time.sleep(30)")]

    monkeypatch.setattr(_MeteredTurn, "_command_for", stopped)
    other = CleanupSession(ctx)
    other.tracker.spec = BudgetSpec(max_usd=10, max_wallclock_s=4)
    events = _capture_events(other, monkeypatch)
    # The run deadline also trips the live check while the waiter is still
    # stopping the worker; the edited draft must survive either path.
    with pytest.raises(mod.CleanupIncomplete) as caught:
        asyncio.run(other(problem="P", document=TEX, findings="Repair"))
    assert "EDITED honest" in caught.value.output.answer_tex
    [data] = [data for kind, data in events if kind == "cleanup.editor_usage_estimated"]
    assert data["charged_usd"] + data["recorded_usd"] == pytest.approx(estimate)
    assert data["saved_usage_usd"] == (pytest.approx(saved) if saved is not None else None)
    assert data["unreported_allowance_usd"] == pytest.approx(max(0, 5.85 - max(saved or 0, data["recorded_usd"])))
    state = _assert_settled(ctx, data["charged_usd"])
    assert state["editor_usage_estimates"] == [data]
    assert state["claude_session_usd"] == pytest.approx(0.25 + (saved or 0))
    monkeypatch.setattr(_MeteredTurn, "_command_for", command)
    before = ctx.budgets.root().counters.usd
    assert asyncio.run(CleanupSession(ctx)(problem="P", document=TEX, findings="Repair")).status == "ready"
    assert "--resume" in seen["commands"][-1]
    assert ctx.budgets.root().counters.usd - before == pytest.approx(0.25)
    assert ctx.budgets.root().counters.usd == pytest.approx(0.25 + estimate + 0.25)


def test_live_monitor_compares_only_this_process_against_its_cap(editor, monkeypatch):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent

    agent, _, _ = editor
    cfg = {"cmd": ["claude", "-p"], "completion_signal": "exit",
           "usage": {"type": "claude_json", "auth_mode": "api", "cost_config": "models/anthropic/fable_51"}}
    terminated = []

    async def terminate():
        terminated.append(True)

    async def wait(*args, **kwargs):
        await asyncio.sleep(1.5)
        return mod.CLIDoneRecord(status="done", summary="completed")

    monkeypatch.setattr(ConfigurableCLIAgent, "_wait_for_done", wait)
    stream = SimpleNamespace(done=False, terminate=terminate, stdout=json.dumps(
        {"type": "result", "usage": {"input_tokens": 1}, "total_cost_usd": 5.2}))
    turn = agent._turn(cfg, "test-editor", 0.2, 60)
    turn.live_usd_limit = 0.24
    turn.claude_baseline = (5.0, 0)
    assert asyncio.run(turn._wait_for_done(stream, Path("unused"))).status == "done"
    assert not terminated
    turn = agent._turn(cfg, "test-editor-2", 0.2, 60)
    turn.live_usd_limit = 0.24
    turn.claude_baseline = (4.9, 0)
    assert asyncio.run(turn._wait_for_done(stream, Path("unused"))).status == "error"
    assert terminated == [True]
