from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from proofstack.agents.ac.ac_workflow import ACDAGWorkflow, ACWorkflow
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.agents.ac.visual_blocks import ACAuthorBlock
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.context import RunContext


def _context(tmp_path, **delegation):
    def no_network(*args, **kwargs):
        raise AssertionError("parallelism tests must not make provider calls")

    return RunContext.create(
        run_id="parallelism", root_workdir=tmp_path, flat=True,
        component_configs={"Author": {"delegation": delegation}},
        api_client_factory=no_network,
    )


def _author_input(round=1):
    return Author.Inputs(
        problem="Prove P.", round=round, n_rounds=3,
        answer_tex="Current answer", research_notes_tex="Current notes", references_bib="References",
    )


@pytest.mark.parametrize("round", [0, 1])
def test_parallelism_one_has_plain_author_prompts_and_tools_even_with_legacy_enabled(tmp_path, round):
    ctx = _context(tmp_path, enabled=True, max_threads=6, max_tasks_per_wave=6)
    plain = Author(ctx, name="Author")
    multi = MultiAuthor(ctx, name="Author")
    multi.author_parallelism = 1
    inp = _author_input(round)
    assert not multi.delegation_enabled()
    assert multi.render_messages(inp) == plain.render_messages(inp)
    assert multi._render_container_messages(inp, "files") == plain._render_container_messages(inp, "files")
    assert multi.extra_client_kwargs() == plain.extra_client_kwargs()

    configs = []
    ctx.api_client_factory = lambda cfg: configs.append(cfg) or SimpleNamespace()
    plain._build_api_client_with_file_ids(["file-original"])
    multi._build_api_client_with_file_ids(["file-original"])
    assert configs[0] == configs[1]


def test_parallelism_one_refuses_direct_delegate_calls(tmp_path, monkeypatch):
    async def scenario():
        author = MultiAuthor(_context(tmp_path, enabled=True), name="Author")
        author.author_parallelism = 1
        author._render_container_messages(_author_input(), "files")

        async def forbidden_wave(*args, **kwargs):
            pytest.fail("parallelism one started a delegation wave")

        monkeypatch.setattr(author, "_run_wave", forbidden_wave)
        result = await asyncio.to_thread(author._delegate, [{"role": "prover", "task": "Prove L"}])
        assert "disabled" in result.lower()
        assert author._waves_done == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("parallelism", [2, 4, 6])
def test_parallelism_enables_and_runs_that_many_seats_without_the_old_four_task_ceiling(tmp_path, monkeypatch, parallelism):
    active = peak = completed = 0

    async def seat(self, inp):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            completed += 1
            return self.Outputs(report=f"Proved {inp.task}")
        finally:
            active -= 1

    async def scenario():
        author = MultiAuthor(_context(tmp_path, enabled=False, max_threads=1), name="Author")
        author.author_parallelism = parallelism
        messages = author._render_container_messages(_author_input(), "files")
        assert author.delegation_enabled()
        assert f"up to {parallelism} independent tasks" in messages[0]["content"]
        assert author._delegation_cfg()["max_tasks_per_wave"] == parallelism
        tasks = [{"role": "prover", "task": f"lemma {i}"} for i in range(parallelism)]
        rejected = await asyncio.to_thread(author._delegate, tasks + [{"role": "checker", "task": "one too many"}])
        assert "at most" in rejected and author._waves_done == 0
        result = await asyncio.to_thread(author._delegate, tasks)
        assert "error:" not in result
        assert completed == peak == parallelism
        assert len(author._delegation_log) == parallelism

    monkeypatch.setattr(SubAuthorSeat, "run", seat)
    asyncio.run(scenario())


@pytest.mark.parametrize("parallelism", [2, 4, 6])
def test_advanced_task_limit_queues_excess_seats_and_keeps_wave_limit(tmp_path, monkeypatch, parallelism):
    active = peak = completed = 0
    task_count = parallelism + 3

    async def seat(self, inp):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            completed += 1
            return self.Outputs(report=f"Proved {inp.task}")
        finally:
            active -= 1

    async def scenario():
        author = MultiAuthor(_context(
            tmp_path, enabled=False, max_threads=1,
            max_tasks_per_wave=task_count, max_waves=2,
        ), name="Author")
        author.author_parallelism = parallelism
        author._render_container_messages(_author_input(), "files")
        tasks = [{"role": "prover", "task": f"lemma {i}"} for i in range(task_count)]
        for wave in (1, 2):
            result = await asyncio.to_thread(author._delegate, tasks)
            assert f"Delegation wave {wave} of 2" in result
            assert "error:" not in result
        assert peak == parallelism
        assert completed == 2 * task_count
        rejected = await asyncio.to_thread(author._delegate, tasks)
        assert "No delegation waves left" in rejected
        assert completed == 2 * task_count

    monkeypatch.setattr(SubAuthorSeat, "run", seat)
    asyncio.run(scenario())


@pytest.mark.parametrize("workflow", [ACWorkflow, FirstProofBatch3Workflow])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, 2.0, "4"])
def test_author_parallelism_requires_a_strict_positive_integer(workflow, value):
    with pytest.raises(ValueError, match="author_parallelism"):
        workflow.Inputs(problem="P", problem_id="p", author_parallelism=value)


@pytest.mark.parametrize("workflow", [ACWorkflow, FirstProofBatch3Workflow])
def test_author_parallelism_defaults_to_one_and_leaves_compute_and_council_unchanged(workflow):
    default = workflow.Inputs(problem="P", problem_id="p")
    parallel = workflow.Inputs(problem="P", problem_id="p", author_parallelism=6)
    assert default.author_parallelism == 1
    assert default.model_dump(exclude={"author_parallelism"}) == parallel.model_dump(exclude={"author_parallelism"})


class _BeforeWorkspace(Exception):
    pass


@pytest.mark.parametrize("parallelism", [1, 4])
def test_class_workflow_configures_author_before_any_workspace_or_provider_use(tmp_path, monkeypatch, parallelism):
    workflow = ACWorkflow(_context(tmp_path, enabled=True, max_threads=9))
    compute_config = workflow.compute._cache_config_snapshot()
    council_config = workflow.council._cache_config_snapshot()

    def before_workspace(*args, **kwargs):
        assert workflow.author.author_parallelism == parallelism
        assert workflow.author.delegation_enabled() is (parallelism > 1)
        raise _BeforeWorkspace

    monkeypatch.setattr(workflow, "_workspace_path", before_workspace)
    inp = workflow.Inputs(problem="P", problem_id="p", author_parallelism=parallelism)
    with pytest.raises(_BeforeWorkspace):
        asyncio.run(workflow.run(inp))
    assert workflow.compute._cache_config_snapshot() == compute_config
    assert workflow.council._cache_config_snapshot() == council_config


@pytest.mark.parametrize("parallelism", [1, 4])
def test_dag_author_block_applies_parallelism_from_workflow_state(tmp_path, monkeypatch, parallelism):
    block = ACAuthorBlock(_context(tmp_path, enabled=True, max_threads=9))
    inputs = ACWorkflow.Inputs(problem="P", problem_id="p", author_parallelism=parallelism)

    def before_workspace(*args, **kwargs):
        assert block.author.author_parallelism == parallelism
        assert block.author.delegation_enabled() is (parallelism > 1)
        raise _BeforeWorkspace

    monkeypatch.setattr(block, "_workspace", before_workspace)
    with pytest.raises(_BeforeWorkspace):
        asyncio.run(block.run(block.Inputs(state={"inputs": inputs.model_dump()})))


@pytest.mark.parametrize("parallelism", [1, 4, 6])
def test_batch3_research_forwards_parallelism_into_its_real_dag_inputs(tmp_path, monkeypatch, parallelism):
    ctx = _context(tmp_path, enabled=True)
    workflow = FirstProofBatch3Workflow(ctx)
    inp = workflow.Inputs(problem="P", problem_id="p", author_parallelism=parallelism)
    seen = []

    async def research(self, inputs):
        seen.append(inputs)
        assert inputs.author_parallelism == parallelism
        assert inputs.compute_max_parallel_workers == inp.compute_max_parallel_workers
        assert inputs.council_models == inp.council_models
        return self.Outputs(
            problem_id=inputs.problem_id, answer_tex=tmp_path / "answer.tex",
            research_notes_tex=tmp_path / "research_notes.tex", references_bib=tmp_path / "references.bib",
        )

    monkeypatch.setattr(ACDAGWorkflow, "run", research)
    asyncio.run(workflow._research(ctx, inp, resume=False, round_bound=3,
                                  out=workflow.Outputs(problem_id="p"), deadline=time.monotonic() + 60))
    assert len(seen) == 1


def test_changing_parallelism_invalidates_author_resume_cache(tmp_path):
    author = MultiAuthor(_context(tmp_path, enabled=True), name="Author")
    inp = _author_input()
    keys = []
    for parallelism in (1, 4, 6):
        author.author_parallelism = parallelism
        keys.append(author._cache_key(inp))
    assert len(set(keys)) == 3


def test_direct_component_use_keeps_legacy_delegation_settings(tmp_path):
    author = MultiAuthor(_context(tmp_path, enabled=True, max_threads=2, max_tasks_per_wave=3), name="Author")
    assert author.author_parallelism is None
    assert author.delegation_enabled()
    assert author._delegation_cfg()["max_threads"] == 2
    assert author._delegation_cfg()["max_tasks_per_wave"] == 3
