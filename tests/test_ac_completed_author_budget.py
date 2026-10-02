import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from proofstack.agents.ac.ac_workflow import ACWorkflow, _problem_hash
from proofstack.agents.ac.visual_blocks import ACAuthorBlock, ACInitBlock, ACReviewJoinBlock
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.ac_workflow import _CompileResult
from proofstack.agents.ac.author import Author
from proofstack.budget import BudgetExhausted
from proofstack.context import RunContext


@pytest.mark.parametrize("workflow_class", [ACWorkflow, ACAuthorBlock])
@pytest.mark.parametrize("round", [0, 3])
@pytest.mark.parametrize("ready", [False, True])
def test_completed_budget_author_is_checkpointed_as_unreviewed(tmp_path, workflow_class, round, ready):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    wf = workflow_class(ctx)
    workspace = tmp_path / "workspace"
    (workspace / ".ac").mkdir(parents=True)
    (workspace / "problem.txt").write_text("Original problem")
    if round:
        (workspace / ".ac/resume-state.json").write_text(json.dumps({
            "version": 1, "problem_hash": _problem_hash("Original problem"),
            "last_round_run": 2, "next_round": 3, "critic_conversation": [{"role": "user", "content": "GAP"}],
            "terminal_outputs": {"early_stopped": True}, "early_stopped": True,
        }))
    output = Author.Outputs(answer_tex="NEW PAID PROOF", research_notes_tex="NOTES", references_bib="BIB",
                            ready=ready, artifact_status="changed")
    exc = BudgetExhausted("run", "usd", 1, 2)
    exc.completed_output = output
    wf.author = AsyncMock(side_effect=exc)
    with pytest.raises(BudgetExhausted):
        asyncio.run(wf._call_author(workspace, problem="Original problem + runtime feedback", round=round))
    wf.author.assert_awaited_once()
    assert (workspace / "answer.tex").read_text() == "NEW PAID PROOF"
    state = json.loads((workspace / ".ac/resume-state.json").read_text())
    assert state["problem_hash"] == _problem_hash("Original problem")
    assert state["awaiting_review_round"] == round
    assert state["awaiting_author"]["answer_tex"] == "NEW PAID PROOF"
    assert state["awaiting_author"]["ready"] is ready
    assert not state["early_stopped"] and state["terminal_outputs"] is None
    assert state["last_round_run"] == round - 1
    if round:
        assert state["critic_conversation"][0]["content"] == "GAP"


@pytest.mark.parametrize("author_ready", [False, True])
@pytest.mark.parametrize("critic_ready", [False, True])
def test_resumed_budget_draft_requires_author_and_critic_votes(tmp_path, author_ready, critic_ready):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    wf = ACWorkflow(ctx)
    workspace = wf._workspace_path("p", "P")
    workspace.mkdir(parents=True)
    (workspace / "problem.txt").write_text("P")
    exc = BudgetExhausted("run", "usd", 1, 2)
    exc.completed_output = Author.Outputs(answer_tex="PRESERVED", ready=author_ready, artifact_status="changed")
    wf.author = AsyncMock(side_effect=exc)

    async def scenario():
        with pytest.raises(BudgetExhausted):
            await wf._call_author(workspace, problem="P", round=1)
        initialized = await ACInitBlock(ctx)(problem="P", problem_id="p", n_rounds=1, resume_run=True,
                                              enable_council=False, enable_compute=False, enable_final_critic=False)
        author = ACAuthorBlock(ctx)
        author.author = AsyncMock(side_effect=AssertionError("must review preserved draft first"))
        resumed = await author(state=initialized.state)
        author.author.assert_not_awaited()
        assert resumed.ready is author_ready
        assert not resumed.state["early_stopped"]
        join = ACReviewJoinBlock(ctx)
        unreviewed = await join(base_state=resumed.state)
        assert not unreviewed.ready_for_gate
        review = ACCritic.Outputs(review_md="Fresh review", answer_ready=critic_ready, mode="fresh")
        reviewed = await join(base_state={**resumed.state, "round_review": review.model_dump(mode="json")})
        assert reviewed.ready_for_gate is (author_ready and critic_ready)

    asyncio.run(scenario())


def test_failed_artifact_does_not_replace_workspace_at_budget_boundary(tmp_path):
    wf = ACWorkflow(RunContext.create(root_workdir=tmp_path, flat=True))
    (tmp_path / "answer.tex").write_text("old draft")
    exc = BudgetExhausted("run", "usd", 1, 2)
    exc.completed_output = Author.Outputs(answer_tex="failed", artifact_status="failed")
    wf.author = AsyncMock(side_effect=exc)
    with pytest.raises(BudgetExhausted):
        asyncio.run(wf._call_author(tmp_path, problem="P", round=1))
    assert (tmp_path / "answer.tex").read_text() == "old draft"
    assert not (tmp_path / ".ac/resume-state.json").exists()


def test_resume_reviews_preserved_draft_without_repeating_author(tmp_path, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    wf = ACWorkflow(ctx)
    workspace = wf._workspace_path("p", "P")
    workspace.mkdir(parents=True)
    (workspace / "problem.txt").write_text("P")
    exc = BudgetExhausted("run", "usd", 1, 2)
    exc.completed_output = Author.Outputs(answer_tex="PRESERVED", ready=True, artifact_status="changed")
    wf.author = AsyncMock(side_effect=exc)
    with pytest.raises(BudgetExhausted):
        asyncio.run(wf._call_author(workspace, problem="P", round=1))
    # Simulate publication interrupted after checkpointing. Resume must restore it.
    (workspace / "answer.tex").write_text("OLD")
    resumed = ACWorkflow(ctx)
    resumed.author = AsyncMock(side_effect=AssertionError("must review before author"))
    resumed.critic = AsyncMock(return_value=ACCritic.Outputs(review_md="Checked", answer_ready=True))
    monkeypatch.setattr("proofstack.agents.ac.ac_workflow._simple_compile_latex", lambda tex, **kw:
                        _CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1))
    out = asyncio.run(resumed(problem="P", problem_id="p", n_rounds=1, resume_run=True,
                             enable_council=False, enable_compute=False, enable_final_critic=False))
    resumed.author.assert_not_awaited()
    resumed.critic.assert_awaited_once()
    assert resumed.critic.call_args.kwargs["answer_tex"] == "PRESERVED"
    assert out.answer_tex.read_text() == "PRESERVED"
