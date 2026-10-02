from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from proofstack.agents.ac.ac_workflow import ACDAGWorkflow
from proofstack.agents import firstproof_batch3 as batch3
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow, FirstProofBatch3RehearsalWorkflow
from proofstack.agents.writeup_loop import RewriteSeat, RepairSeat
from proofstack.budget import BudgetExhausted
from proofstack.latex_contract import normalize_submission_latex, render_firstproof_latex_contract
from test_firstproof_batch3 import ORIGINAL, TEX, pipeline

REAL_COMPILE = FirstProofBatch3Workflow._compile


@pytest.mark.parametrize("failure", ["timeout", "incomplete"])
def test_partial_continues_saved_cli_edits_only_after_accounting_settles(pipeline, monkeypatch, failure):
    from proofstack.agents.cleanup_session import CleanupIncomplete

    _, _, _, _, ctx, agent = pipeline
    calls = []
    revised = TEX.replace("Proof.", "Saved revision with a known gap.")

    async def edit(ctx_arg, inp, candidate, **kwargs):
        calls.append(kwargs)
        root = ctx.root_workdir / "cleanup_sessions" / kwargs["session_key"]
        if len(calls) == 1:
            (root / "workspace").mkdir(parents=True)
            (root / "session.json").write_text(json.dumps({"in_flight": False, "resumable": True}))
            (root / "workspace/answer.tex").write_text(revised)
            if failure == "incomplete":
                raise CleanupIncomplete("stopped", SimpleNamespace(answer_tex=revised))
            raise TimeoutError("first pass expired")
        assert candidate == revised
        assert kwargs["finishing_only"] and not kwargs["mechanical"]
        assert kwargs["session_baseline"] == calls[0]["session_baseline"]
        return revised

    monkeypatch.setattr(agent, "_edit", edit)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="claude_code")
    out = asyncio.run(agent._partial(inp, agent.Outputs(problem_id="p"), TEX, "notes", "gaps", deadline=time.monotonic()+600))
    assert len(calls) == 2 and out.partial_ready and not out.submission_approved
    assert "Saved revision" in out.answer_tex.read_text()
    assert any("continued after" in e for e in out.cleanup_errors)


@pytest.mark.parametrize("blocked", ["uncertain", "in_flight", "not_resumable", "second_failure", "cancel", "no_budget", "no_time", "too_short", "no_repair_slot"])
def test_partial_continuation_never_bypasses_guards_or_loses_fallback(pipeline, monkeypatch, blocked):
    _, _, _, _, ctx, agent = pipeline
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(batch3, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def phase(name, *, usd, deadline, call):
        return await call(ctx)

    async def edit(ctx_arg, inp, candidate, **kwargs):
        calls.append(kwargs)
        root = ctx.root_workdir / "cleanup_sessions" / kwargs["session_key"]
        (root / "workspace").mkdir(parents=True, exist_ok=True)
        (root / "session.json").write_text(json.dumps({"in_flight": blocked == "in_flight", "resumable": blocked != "not_resumable"}))
        (root / "workspace/answer.tex").write_text(TEX.replace("Proof.", "Unfinished edits."))
        if blocked == "uncertain":
            (ctx.root_workdir / "cleanup-accounting-uncertain.json").write_text("{}")
        if blocked == "cancel":
            raise asyncio.CancelledError()
        if blocked == "no_budget":
            agent.tracker.add_usd(100)
        if blocked == "no_time":
            clock[0] = 1599.0
        if blocked == "too_short":
            clock[0] = 1100.0
        raise TimeoutError("first or second pass expired")

    monkeypatch.setattr(agent, "_phase", phase)
    monkeypatch.setattr(agent, "_edit", edit)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="claude_code",
                       max_partial_cleanup_repairs=1 if blocked == "no_repair_slot" else 2)
    out = agent.Outputs(problem_id="p")
    run = agent._partial(inp, out, TEX, "", "", deadline=1600.0)
    if blocked == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run)
    else:
        asyncio.run(run)
    assert len(calls) == (2 if blocked == "second_failure" else 1)
    assert out.partial_ready and "Unfinished edits" not in out.answer_tex.read_text()
    assert not out.submission_approved
    if blocked != "cancel":
        saved = json.loads((ctx.root_workdir / "batch3-output.json").read_text())
        assert saved["cleanup_errors"] == out.cleanup_errors


@pytest.mark.parametrize("needs_repair", [False, True])
def test_two_hour_window_keeps_repair_time_after_continuation(pipeline, monkeypatch, needs_repair):
    _, _, _, _, ctx, agent = pipeline
    now, caps = [1000.0], []
    monkeypatch.setattr(batch3, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def phase(name, *, usd, deadline, call):
        caps.append((name, deadline - now[0]))
        now[0] = deadline
        return await call(ctx)

    async def edit(ctx_arg, inp, candidate, **kwargs):
        if len(caps) == 1:
            root = ctx.root_workdir / "cleanup_sessions" / kwargs["session_key"]
            (root / "workspace").mkdir(parents=True)
            (root / "session.json").write_text('{"in_flight": false, "resumable": true}')
            (root / "workspace/answer.tex").write_text(TEX)
            raise TimeoutError("rewrite timeout")
        if needs_repair and len(caps) == 2:
            return TEX.replace("Proof.", "Oversized.")
        if len(caps) == 3:
            assert kwargs["mechanical"] and kwargs["finishing_only"]
        return TEX

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, "OK"

    monkeypatch.setattr(agent, "_phase", phase)
    monkeypatch.setattr(agent, "_edit", edit)
    monkeypatch.setattr(agent, "_compile", compile_document)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="claude_code")
    out = asyncio.run(agent._partial(inp, agent.Outputs(problem_id="p"), TEX, "", "", deadline=8200))
    assert caps == [("partial_rewrite", 5400), ("partial_continue", 900)] + (
        [("partial_repair", 600)] if needs_repair else [])
    assert now[0] == (7900 if needs_repair else 7300) and out.partial_ready


def test_continuing_a_mechanical_repair_does_not_allow_substantive_edits(pipeline, monkeypatch):
    _, _, _, _, ctx, agent = pipeline
    calls = []

    async def edit(ctx_arg, inp, candidate, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return TEX.replace("Proof.", "Oversized.")
        assert kwargs["mechanical"] and kwargs["finishing_only"]
        if len(calls) == 2:
            root = ctx.root_workdir / "cleanup_sessions" / kwargs["session_key"]
            (root / "workspace").mkdir(parents=True)
            (root / "session.json").write_text('{"in_flight": false, "resumable": true}')
            (root / "workspace/answer.tex").write_text(TEX)
            raise TimeoutError("mechanical repair timed out")
        return TEX

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, "OK"

    monkeypatch.setattr(agent, "_edit", edit)
    monkeypatch.setattr(agent, "_compile", compile_document)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="claude_code",
                       max_partial_cleanup_repairs=3)
    out = asyncio.run(agent._partial(inp, agent.Outputs(problem_id="p"), TEX, "", "",
                                     deadline=time.monotonic()+7200))
    assert len(calls) == 3 and out.partial_ready


def titled(pages):
    return normalize_submission_latex(
        r"\documentclass[12pt]{article}\title{A partial argument}\author{}\date{}"
        "\n" + r"\begin{document}\maketitle" + "\n"
        + "\n\\newpage\n".join(f"Page {i}. A remaining gap is unresolved." for i in range(pages))
        + "\n" + r"\end{document}"
    )


@pytest.mark.parametrize("prefix", [
    "% \\maketitle\n", r"\verb|\maketitle|",
    r"\begin{verbatim}\maketitle\end{verbatim}",
    r"\newcommand{\example}{\maketitle}",
])
def test_notice_ignores_inactive_title_commands_and_is_idempotent(prefix):
    doc = titled(1).replace(r"\maketitle", prefix + "\n" + r"\maketitle")
    labelled = FirstProofBatch3Workflow._label_partial(doc)
    assert labelled.index("Partial result:") > labelled.rindex(r"\maketitle")
    assert FirstProofBatch3Workflow._label_partial(labelled) == labelled
    assert normalize_submission_latex(labelled) == labelled


def test_notice_upgrades_legacy_label_without_duplicating_it():
    legacy = ("\n\\section*{Partial result: not a complete solution}\n"
              "This attempt did not receive final mathematical acceptance. "
              "Its deadline cleanup has not been mathematically reviewed.\n")
    old = titled(1).replace(r"\begin{document}", r"\begin{document}" + legacy)
    labelled = FirstProofBatch3Workflow._label_partial(old)
    assert labelled.count("Partial result:") == 1
    assert labelled.index("Partial result:") > labelled.index(r"\maketitle")
    assert "section*{Partial result" not in labelled


@pytest.mark.parametrize("body", [
    r"\iffalse\maketitle\fi",
    r"\iftrue\iffalse\maketitle\fi\fi",
    r"\begingroup\iffalse\maketitle\fi\endgroup",
    r"\newif\ifdraft\draftfalse\ifdraft\maketitle\fi",
    r"\end{document}\maketitle",
])
def test_notice_is_not_inserted_in_uncertain_or_dead_title_branch(body):
    doc = titled(1).replace(r"\maketitle", body)
    labelled = FirstProofBatch3Workflow._label_partial(doc)
    assert labelled.index("Partial result:") < labelled.index(body)
    assert FirstProofBatch3Workflow._label_partial(labelled) == labelled


@pytest.mark.parametrize("location", ["conditional", "macro", "after_end"])
def test_hidden_existing_notice_is_relocated_to_visible_position(location):
    notice = r"\noindent\textbf{Partial result: not a complete solution (unreviewed).}\par"
    if location == "conditional":
        doc = TEX.replace("Proof.", r"\iffalse" + notice + r"\fi Proof.")
    elif location == "macro":
        doc = TEX.replace("Proof.", r"\newcommand{\unused}{" + notice + "} Proof.")
    else:
        doc = TEX + notice
    labelled = FirstProofBatch3Workflow._label_partial(doc)
    assert labelled.count(notice) == 1
    assert labelled.index(notice) < labelled.index({
        "conditional": r"\iffalse", "macro": r"\newcommand", "after_end": r"\end{document}",
    }[location])
    assert FirstProofBatch3Workflow._label_partial(labelled) == labelled


def test_comment_only_document_marker_is_a_repairable_error():
    doc = TEX.replace(r"\begin{document}", "% \\begin{document}")
    with pytest.raises(ValueError, match="No active"):
        FirstProofBatch3Workflow._label_partial(doc)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
@pytest.mark.parametrize("limit", [1, 16, 40])
@pytest.mark.parametrize("grouped", [False, True])
def test_title_aware_notice_preserves_exact_page_boundary(pipeline, limit, grouped):
    _, _, _, _, _, agent = pipeline
    inp = FirstProofBatch3RehearsalWorkflow.Inputs(problem=ORIGINAL, problem_id="p", page_limit=limit)
    doc = titled(limit)
    if grouped:
        doc = doc.replace(r"\maketitle", r"{\maketitle}")
    labelled = agent._label_partial(doc)
    assert agent._label_partial(labelled) == labelled

    async def check():
        # Use the actual compiler, not the pipeline fixture's stub.
        deadline = time.monotonic() + 90
        for candidate in (doc, labelled):
            compiled, pages, detail = await REAL_COMPILE(agent, candidate, deadline=deadline)
            assert compiled and pages == inp.page_limit, detail

    asyncio.run(check())


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("pdflatex", "pdftotext")),
                    reason="pdflatex/pdftotext not installed")
@pytest.mark.parametrize("title", [r"\iffalse\maketitle\fi", r"{\maketitle}"])
def test_partial_notice_is_present_in_compiled_pdf(tmp_path, title):
    doc = titled(1).replace(r"\maketitle", title)
    (tmp_path / "notice.tex").write_text(FirstProofBatch3Workflow._label_partial(doc))
    result = subprocess.run(
        ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "notice.tex"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout
    text = subprocess.run(["pdftotext", "notice.pdf", "-"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=10, check=True).stdout
    assert "Partial result: not a complete solution (unreviewed)." in " ".join(text.split())


def test_api_editor_contract_does_not_require_an_unavailable_compiler(pipeline):
    execute, _, seen, options, _, _ = pipeline
    options["research_error"] = BudgetExhausted("run", "usd", 85, 85)
    out = asyncio.run(execute())
    assert out.partial_ready
    prompt = seen["prompts"][0]
    assert "Run `pdflatex` and fix compile errors" not in prompt
    assert "no shell or LaTeX compiler" in prompt
    assert "do not declare UNABLE merely" in prompt
    assert "manuscript measured 1 pages" in prompt
    assert "Aim for at most 15 pages" in prompt
    assert "Run `pdflatex`" in render_firstproof_latex_contract(16)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_partial_overflow_then_syntax_error_get_targeted_repairs(pipeline, monkeypatch):
    execute, calls, _, options, ctx, agent = pipeline
    options.update(research_error=BudgetExhausted("run", "usd", 85, 85), research_document=titled(1))
    repairs = []

    async def rewrite(self, inp):
        calls["partial"] += 1
        return self.Outputs(text=titled(2))

    async def repair(self, inp):
        repairs.append(inp.prompt)
        if len(repairs) == 1:
            return self.Outputs(text=titled(1).replace("Page 0.", r"\UndefinedSyntaxCommand"))
        return self.Outputs(text=titled(1))

    monkeypatch.setattr(RewriteSeat, "run", rewrite)
    monkeypatch.setattr(RepairSeat, "run", repair)
    monkeypatch.setattr(agent, "_compile", REAL_COMPILE.__get__(agent))
    out = asyncio.run(execute(page_limit=1, max_wallclock_s=90))
    assert out.partial_ready and not out.submission_approved and out.pages == 1
    assert calls["partial"] == 1 and calls["review"] == 0
    assert len(repairs) == 2
    assert "pages=2" in repairs[0] and "Aim for at most 1 pages" in repairs[0]
    assert repairs[0].count("compiled=True, pages=2;") == 1
    assert "Undefined control sequence" in repairs[1]
    assert all("not a mathematical review" in p and "smallest changes" in p for p in repairs)
    assert all("correct-but-dense" not in p for p in repairs)
    assert all("Private scratch work" not in p for p in repairs)
    assert all("Latest unresolved findings:" not in p for p in repairs)
    assert list((ctx.root_workdir / "agents").glob("FirstProofBatch3Workflow-*/partial-check-2.json"))


@pytest.mark.parametrize("malformed_rewrite", [False, True])
def test_missing_active_document_marker_can_be_fixed_by_editor(pipeline, monkeypatch, malformed_rewrite):
    execute, calls, seen, options, _, agent = pipeline
    malformed = TEX.replace(r"\begin{document}", "% \\begin{document}")
    options.update(research_error=BudgetExhausted("run", "usd", 85, 85), research_document=malformed)

    async def rewrite(self, inp):
        seen["rewrite_prompt"] = inp.prompt
        calls["partial"] += 1
        return self.Outputs(text=malformed if malformed_rewrite else TEX)

    async def repair(self, inp):
        assert "No active" in inp.prompt
        calls["repair"] += 1
        return self.Outputs(text=TEX)

    monkeypatch.setattr(RewriteSeat, "run", rewrite)
    monkeypatch.setattr(RepairSeat, "run", repair)
    out = asyncio.run(execute())
    assert out.partial_ready and out.error is None
    assert calls["partial"] == 1 and calls["repair"] == int(malformed_rewrite)
    assert calls["review"] == 0
    assert "No active" in seen["rewrite_prompt"]
    assert "Partial result:" in out.answer_tex.read_text()


@pytest.mark.parametrize("cap,expected", [(0, [100]), (1, [75, 25]), (2, [75, 15, 10]),
                                         (5, [75, 15, 2.5, 2.5, 2.5, 2.5])])
def test_partial_call_allocations_favor_rewrite_and_remain_bounded(pipeline, monkeypatch, cap, expected):
    _, _, _, _, _, agent = pipeline
    now, phases = [1000.0], []
    monkeypatch.setattr(batch3, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def phase(name, *, usd, deadline, call):
        phases.append((usd, deadline - now[0]))
        agent.tracker.add_usd(usd)
        now[0] = deadline
        return TEX.replace("Proof.", "Oversized.")

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, "OK"

    monkeypatch.setattr(agent, "_phase", phase)
    monkeypatch.setattr(agent, "_compile", compile_document)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", max_partial_cleanup_repairs=cap)
    out = agent.Outputs(problem_id="p")
    result = asyncio.run(agent._partial(inp, out, TEX, "", "", deadline=8200.0))
    assert [usd for usd, _ in phases] == pytest.approx(expected)
    assert phases[0][1] == pytest.approx(6900 if cap == 0 else 5400)
    assert phases[0][1] > 60 * 60
    assert now[0] <= 7900
    assert result.partial_ready and result.pages == 1


def test_unused_partial_allowances_carry_forward(pipeline, monkeypatch):
    _, _, _, _, _, agent = pipeline
    now, phases = [1000.0], []
    monkeypatch.setattr(batch3, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def phase(name, *, usd, deadline, call):
        phases.append((usd, deadline - now[0]))
        agent.tracker.add_usd(5)
        now[0] += 10
        return TEX.replace("Proof.", "Oversized.")

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, "OK"

    monkeypatch.setattr(agent, "_phase", phase)
    monkeypatch.setattr(agent, "_compile", compile_document)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    asyncio.run(agent._partial(inp, agent.Outputs(problem_id="p"), TEX, "", "", deadline=4300.0))
    assert [usd for usd, _ in phases] == pytest.approx([75, 95 * 15 / 25, 90])
    assert [seconds for _, seconds in phases] == pytest.approx([3000 * 18 / 23, 2990 * 15 / 25, 2980])


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_mechanical_repair_cap_preserves_checkpoint(pipeline, monkeypatch, cap):
    execute, calls, _, options, _, agent = pipeline
    options.update(research_error=BudgetExhausted("run", "usd", 85, 85),
                   edit_document=TEX.replace("Proof.", "Oversized argument."))

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, ""

    monkeypatch.setattr(agent, "_compile", compile_document)
    out = asyncio.run(execute(max_partial_cleanup_repairs=cap))
    assert calls["partial"] == cap + 1 and calls["review"] == 0
    assert out.partial_ready and out.pages == 1
    assert "Oversized" not in out.answer_tex.read_text()


def test_exhausted_usd_does_not_start_a_mechanical_repair(pipeline, monkeypatch):
    execute, calls, _, options, _, agent = pipeline
    options["research_error"] = BudgetExhausted("run", "usd", 85, 85)

    async def rewrite(self, inp):
        calls["partial"] += 1
        self.tracker.add_usd(100)
        exc = BudgetExhausted("run", "usd", 100, 120)
        exc.completed_output = self.Outputs(text=TEX.replace("Proof.", "Oversized argument."))
        raise exc

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, ""

    monkeypatch.setattr(RewriteSeat, "run", rewrite)
    monkeypatch.setattr(agent, "_compile", compile_document)
    out = asyncio.run(execute())
    assert out.partial_ready and out.pages == 1
    assert calls["partial"] == 1 and calls["repair"] == 0


def test_cancelling_a_mechanical_repair_keeps_checkpoint_and_starts_no_more_calls(pipeline, monkeypatch):
    execute, calls, _, options, ctx, agent = pipeline
    options.update(research_error=BudgetExhausted("run", "usd", 85, 85),
                   edit_document=TEX.replace("Proof.", "Oversized argument."))
    entered = asyncio.Event()

    async def compile_document(document, *, deadline):
        return True, 17 if "Oversized" in document else 1, ""

    async def repair(self, inp):
        calls["repair"] += 1
        entered.set()
        await asyncio.sleep(60)
        pytest.fail("cancelled repair continued")

    monkeypatch.setattr(agent, "_compile", compile_document)
    monkeypatch.setattr(RepairSeat, "run", repair)

    async def run():
        task = asyncio.create_task(execute())
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert calls["repair"] == calls["partial"] == 1
    assert calls["review"] == 0
    manifest = json.loads((ctx.root_workdir / "batch3-output.json").read_text())
    assert manifest["partial_ready"] and manifest["pages"] == 1
    assert "Oversized" not in agent._restore_output(agent.Inputs(problem=ORIGINAL, problem_id="p")).answer_tex.read_text()


def test_page_headroom_is_a_target_not_a_stricter_export_cap(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options.update(research_error=BudgetExhausted("run", "usd", 85, 85), pages=16)
    out = asyncio.run(execute(cleanup_page_headroom=2))
    assert out.partial_ready and out.pages == 16
    assert calls["partial"] == 1
    assert "Aim for at most 14 pages" in seen["prompts"][0]


def test_each_author_checkpoints_final_form_and_bad_later_drafts_cannot_erase_it(pipeline, monkeypatch):
    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    out = agent.Outputs(problem_id="p")
    saved = []

    async def research(self, inp):
        workspace = agent._workspace(inp)
        (workspace / ".ac").mkdir(parents=True, exist_ok=True)
        for i, document in enumerate([TEX, TEX + "\n% broken"]):
            (workspace / "answer.tex").write_text(document)
            feedback = await self.ctx.author_checkpoint(workspace, i)
            assert "Final export check" in feedback and "15 pages" in feedback
            assert (workspace / "answer.tex").read_text() == document
            restored = agent._restore_output(inp)
            saved.append(restored.partial_sha256)
            assert restored.partial_ready and restored.pages == 1
        return self.Outputs(problem_id="p", answer_tex=workspace / "answer.tex",
                            research_notes_tex=workspace / "research_notes.tex", references_bib=workspace / "references.bib")

    async def compile_document(document, *, deadline):
        assert "Partial result:" in document
        return (False, 0, "broken") if "broken" in document else (True, 1, "OK")

    monkeypatch.setattr(ACDAGWorkflow, "run", research)
    monkeypatch.setattr(agent, "_compile", compile_document)
    asyncio.run(agent._research(ctx, inp, resume=False, round_bound=2, out=out, deadline=time.monotonic() + 30))
    assert saved[0] == saved[1] and saved[0]
    assert "broken" not in out.answer_tex.read_text()
    assert ctx.author_checkpoint is None  # scoped to this research invocation
    manifest = json.loads((ctx.root_workdir / "batch3-output.json").read_text())
    assert manifest["partial_sha256"] == saved[0]


@pytest.mark.parametrize("overrides", [{"max_partial_cleanup_repairs": -1}, {"max_partial_cleanup_repairs": 6},
                                      {"cleanup_page_headroom": -1}])
def test_partial_cleanup_knobs_are_bounded(overrides):
    with pytest.raises(ValueError):
        FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, **overrides)
