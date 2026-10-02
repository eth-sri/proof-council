from __future__ import annotations

import importlib.util
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "firstproof_entrypoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_firstproof_entrypoint_profiles_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fp = _load_module()


def _clear_firstproof_env(monkeypatch):
    for key in list(fp.os.environ):
        if key.startswith("FIRSTPROOF_"):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("ids", [
    ["p?", "p!", "p-2"],
    ["p-2", "p?", "p!", "p-3", "p?"],
    ["p", "p", "p", "p-2", "p-2-2"],
    ["a" * 120 + "x", "a" * 120 + "y", "a" * 118 + "-2"],
    ["a" * 118 + "-2", "a" * 120 + "x", "a" * 120 + "y"],
    ["", "prob-001", "prob-001-2"],
])
def test_problem_ids_cannot_alias_files_or_run_directories(tmp_path, monkeypatch, ids):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_TMP_PROBLEM_DIR", str(tmp_path / "inputs"))
    settings = replace(fp._settings(), output_dir=tmp_path / "outputs")
    items = [{"id": value, "latex": f"Problem {index}"} for index, value in enumerate(ids)]

    problems = fp._parse_problems(items, settings)

    for attr in ("safe_id", "run_id", "problem_path", "log_path", "output_tex_path"):
        assert len({getattr(problem, attr) for problem in problems}) == len(items), attr
    for problem, item in zip(problems, items):
        assert problem.problem_path.read_text() == item["latex"] + "\n"
        assert 0 < len(problem.safe_id) <= 120
    repeated = fp._parse_problems(items, settings)
    assert [p.safe_id for p in repeated] == [p.safe_id for p in problems]


@pytest.fixture
def submission_image_workflow():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    match = re.search(r"\bFIRSTPROOF_WORKFLOW=(\S+)", dockerfile)
    assert match is not None, "Submission image must select its default workflow"
    return match.group(1)


def test_submission_image_defaults_to_batch3(monkeypatch, submission_image_workflow):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", submission_image_workflow)

    settings = fp._settings()

    assert settings.workflow == "firstproof_batch3"
    assert settings.page_limit == 16
    assert settings.max_parallel == 10
    assert settings.budget_usd_per_question == 1050.0
    from proofstack.registry import load_preset
    preset = load_preset(settings.workflow)
    assert preset.budget.max_usd == 1050.0
    assert preset.inputs["compute_max_parallel_workers"] == 4
    assert settings.compute_max_parallel_workers == 4
    assert settings.author_parallelism == 1
    assert preset.inputs["compute_memory_gb"] == 8
    assert preset.inputs["compute_memory_reserve_gb"] == 16
    assert settings.deadline_seconds == 23 * 3600 + 55 * 60
    assert json.loads((REPO_ROOT / "hardware.json").read_text())["instance_type"] == "r7i.2xlarge"
    assert json.loads((REPO_ROOT / "hardware.json").read_text())["timeout_minutes"] == 1440
    assert json.loads((REPO_ROOT / "hardware.json").read_text())["storage_gb"] == 250
    assert not settings.adaptive_continuation
    assert fp._round_schedule(settings) == [50]


def test_submission_image_budget_and_parallelism_are_overridable(monkeypatch, submission_image_workflow):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", submission_image_workflow)
    monkeypatch.setenv("FIRSTPROOF_BUDGET_USD_PER_QUESTION", "25")
    monkeypatch.setenv("FIRSTPROOF_MAX_PARALLEL", "2")

    settings = fp._settings()

    assert settings.batch3
    assert settings.budget_usd_per_question == 25
    assert settings.max_parallel == 2


def test_cleanup_reserve_uses_schema_default_when_not_explicit_in_preset(monkeypatch):
    from types import SimpleNamespace
    from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow

    monkeypatch.setattr("proofstack.registry.load_preset", lambda name: SimpleNamespace(build_inputs=lambda: {}))
    assert fp._batch3_cleanup_seconds("firstproof_batch3") == FirstProofBatch3Workflow.Inputs.model_fields[
        "partial_cleanup_seconds"].default


@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
@pytest.mark.parametrize("override", [None, "25", "invalid"])
def test_batch3_budget_defaults_and_overrides(monkeypatch, workflow, override):
    from proofstack.registry import load_preset

    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", workflow)
    if override is not None:
        monkeypatch.setenv("FIRSTPROOF_BUDGET_USD_PER_QUESTION", override)
    settings = fp._settings()
    preset = load_preset(workflow)
    assert preset.budget.max_usd == 1050.0
    assert preset.budget.max_usd * 10 == 10500.0
    assert preset.budget.max_wallclock_s == 86100
    assert preset.component_configs["ACCritic"]["research_notes_transport"] == "file"
    assert preset.component_configs["ACCritic"]["research_notes_container"] == "auto"
    assert preset.component_configs["ACCritic"]["max_hosted_tool_calls"] == 30
    assert preset.inputs["partial_cleanup_seconds"] == fp._batch3_cleanup_seconds(workflow) == 7200
    assert preset.inputs["partial_cleanup_reserve_fraction"] == 0.15
    assert preset.budget.max_usd * preset.inputs["partial_cleanup_reserve_fraction"] == pytest.approx(157.50)
    assert settings.budget_usd_per_question == (25.0 if override == "25" else 1050.0)
    if override == "invalid":
        assert any("FIRSTPROOF_BUDGET_USD_PER_QUESTION" in warning for warning in settings.warnings)


@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_batch3_compute_parallelism_has_an_independent_positive_override(monkeypatch, workflow):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", workflow)
    monkeypatch.setenv("FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS", "2")
    settings = fp._settings()
    assert settings.compute_max_parallel_workers == 2
    assert settings.max_parallel == 10


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_batch3_invalid_compute_parallelism_preserves_preset_guard(monkeypatch, value):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3")
    monkeypatch.setenv("FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS", value)
    settings = fp._settings()
    assert settings.compute_max_parallel_workers == 4
    assert any("FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS" in w for w in settings.warnings)


def test_batch3_compute_default_comes_from_selected_preset(tmp_path, monkeypatch):
    from proofstack.registry import load_preset

    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["compute_max_parallel_workers"] = 3
    custom = tmp_path / "custom_batch3.yaml"
    custom.write_text(json.dumps(raw))
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", str(custom))
    assert fp._settings().compute_max_parallel_workers == 3


def test_batch3_invalid_preset_cannot_disable_compute_guard(tmp_path, monkeypatch):
    from proofstack.registry import load_preset

    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["compute_max_parallel_workers"] = 0
    custom = tmp_path / "unsafe_batch3.yaml"
    custom.write_text(json.dumps(raw))
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", str(custom))
    with pytest.raises(ValueError, match="positive compute_max_parallel_workers"):
        fp._settings()


@pytest.mark.parametrize("workflow,expected", [
    ("firstproof_batch3", 1),
    ("firstproof_batch3_multiauthor", 4),
    ("firstproof_smoke_fast", 1),
    ("firstproof_submission", 1),
])
def test_author_parallelism_defaults_to_selected_workflow(monkeypatch, workflow, expected):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", workflow)
    assert fp._settings().author_parallelism == expected


@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
@pytest.mark.parametrize("parallelism", [1, 2, 6])
def test_author_parallelism_can_enable_or_disable_delegation_independently(monkeypatch, workflow, parallelism):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", workflow)
    monkeypatch.setenv("FIRSTPROOF_AUTHOR_PARALLELISM", str(parallelism))
    settings = fp._settings()
    assert settings.author_parallelism == parallelism
    assert settings.compute_max_parallel_workers == 4
    assert settings.max_parallel == 10


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_invalid_author_parallelism_preserves_preset_default(monkeypatch, value):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3_multiauthor")
    monkeypatch.setenv("FIRSTPROOF_AUTHOR_PARALLELISM", value)
    settings = fp._settings()
    assert settings.author_parallelism == 4
    assert any("FIRSTPROOF_AUTHOR_PARALLELISM" in w for w in settings.warnings)


@pytest.mark.parametrize("parallelism", [None, 3])
def test_custom_preset_author_parallelism_falls_back_to_input_default(tmp_path, monkeypatch, parallelism):
    from proofstack.registry import load_preset

    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    if parallelism is None:
        raw["inputs"].pop("author_parallelism", None)
    else:
        raw["inputs"]["author_parallelism"] = parallelism
    custom = tmp_path / "custom_batch3.yaml"
    custom.write_text(json.dumps(raw))
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", str(custom))
    assert fp._settings().author_parallelism == (parallelism or 1)


@pytest.mark.parametrize("parallelism", [0, -1, 1.5, True])
def test_invalid_preset_author_parallelism_is_rejected(tmp_path, monkeypatch, parallelism):
    from proofstack.registry import load_preset

    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["author_parallelism"] = parallelism
    custom = tmp_path / "invalid_batch3.yaml"
    custom.write_text(json.dumps(raw))
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", str(custom))
    with pytest.raises(ValueError, match="positive integer author_parallelism"):
        fp._settings()


def test_author_parallelism_is_not_forwarded_to_an_unrelated_workflow(monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "human_smoke")
    monkeypatch.setenv("FIRSTPROOF_AUTHOR_PARALLELISM", "4")
    assert fp._settings().author_parallelism is None


@pytest.mark.parametrize("workflow", ["firstproof_smoke_fast", "firstproof_submission"])
def test_submission_image_allows_workflow_override(monkeypatch, submission_image_workflow, workflow):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", submission_image_workflow)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", workflow)

    settings = fp._settings()

    assert settings.workflow == workflow
    assert settings.page_limit == 12
    assert settings.n_rounds == 10
    assert settings.adaptive_continuation
    assert settings.compute_max_parallel_workers is None


def test_firstproof_defaults_to_submission_workflow(monkeypatch):
    _clear_firstproof_env(monkeypatch)

    settings = fp._settings()

    assert settings.workflow == "firstproof_submission"
    assert settings.n_rounds == 10
    assert settings.round_batch_size == 5
    assert settings.adaptive_continuation is True
    assert settings.adaptive_max_rounds == 200
    assert settings.budget_usd_per_question == 1000.0


def test_firstproof_ignores_legacy_profile_env_without_warning(monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_PROFILE", "firstproof_submission")

    settings = fp._settings()

    assert settings.workflow == "firstproof_submission"
    assert settings.n_rounds == 10
    assert settings.round_batch_size == 5
    assert settings.adaptive_continuation is True
    assert settings.adaptive_max_rounds == 200
    assert settings.budget_usd_per_question == 1000.0
    assert fp._round_schedule(settings)[:3] == [5, 10, 15]
    assert fp._round_schedule(settings)[-1] == 200
    assert not any("FIRSTPROOF_PROFILE" in warning for warning in settings.warnings)


def test_firstproof_env_overrides_built_in_defaults(monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_N_ROUNDS", "12")
    monkeypatch.setenv("FIRSTPROOF_BUDGET_USD_PER_QUESTION", "123")

    settings = fp._settings()

    assert settings.n_rounds == 12
    assert settings.budget_usd_per_question == 123.0
    assert fp._round_schedule(settings)[:3] == [5, 10, 15]
    assert fp._round_schedule(settings)[-1] == 200


def test_batch3_selects_single_pipeline_and_16_pages(monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3")
    settings = fp._settings()
    assert settings.page_limit == 16
    assert not settings.adaptive_continuation
    assert fp._round_schedule(settings) == [50]
    monkeypatch.setenv("FIRSTPROOF_ADAPTIVE_CONTINUATION", "true")
    monkeypatch.setenv("FIRSTPROOF_PAGE_LIMIT", "20")
    settings = fp._settings()
    assert not settings.adaptive_continuation
    assert settings.page_limit == 16
    assert any("disabling" in w for w in settings.warnings)
    assert any("clamping" in w for w in settings.warnings)


@pytest.fixture
def batch3_adapter(tmp_path, monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3")
    settings = replace(fp._settings(), output_dir=tmp_path)
    problem = fp.Problem(
        ordinal=1, original_id="p", safe_id="p", text="Problem", input_error=None,
        problem_path=tmp_path / "p.input.tex", log_path=tmp_path / "p.log",
        output_tex_path=tmp_path / "p.tex", run_id="run-p",
    )
    run = tmp_path / "workflow_runs" / "run-p"
    (run / "solutions").mkdir(parents=True)
    (run / "solutions" / "p.tex").write_text("UNAPPROVED RESEARCH")
    (run / "submissions").mkdir()
    candidate = run / "submissions" / "p.tex"
    candidate.write_text(fp._ensure_complete_latex(r"\documentclass[12pt]{article}\begin{document}Proof.\end{document}"))
    metadata = run / "run-metadata.json"

    def approve(**overrides):
        metadata.write_text(json.dumps({"outputs": {
            "submission_approved": True,
            "submission_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            **overrides,
        }}))

    return problem, settings, candidate, approve


def test_deadline_during_final_compile_preserves_validated_publication(batch3_adapter, monkeypatch):
    problem, settings, candidate, approve = batch3_adapter
    approve(compiled=True, pages=1, early_stopped=True, rounds_completed=1)
    settings = replace(settings, deadline_at=time.monotonic() + 2)
    calls = []

    async def run(*args, **kwargs):
        return 0

    def compile(*args, deadline, **kwargs):
        calls.append(deadline)
        time.sleep(max(0, deadline - time.monotonic()) + .01)
        return False, "submission compilation timed out or reached the batch deadline"

    monkeypatch.setattr(fp, "_run_subprocess", run)
    monkeypatch.setattr(fp, "_compile_exact_latex", compile)

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await fp._run_problem(problem, settings, asyncio.Semaphore(1))
        return await fp._exception_result(problem, asyncio.CancelledError("batch deadline"), settings)

    result = asyncio.run(scenario())
    assert calls == [settings.deadline_at]
    assert result.status == "deadline_cancelled_with_solution"
    assert result.latex == candidate.read_text()
    assert problem.output_tex_path.read_text() == result.latex
    assert result.rejected_solution_path is None


def test_live_summary_includes_unfinished_problem_spending(batch3_adapter):
    import time
    problem, settings, candidate, approve = batch3_adapter
    events = settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl"
    events.write_text(json.dumps({
        "kind": "model.call", "call_id": "live-call", "payload": {
            "model": "test", "cost_usd": 7.5, "in_tokens": 100, "out_tokens": 20,
            "reasoning_tokens": 15,
        },
    }) + "\n")
    _, summary, records = fp._aggregate_payloads(
        [problem], [None], settings, "test", time.monotonic(), in_progress=True,
    )
    assert summary["completed_count"] == 0
    assert summary["totals"]["cost_usd"] == 7.5
    assert summary["per_problem"][0]["status"] == "running"
    assert summary["per_problem"][0]["totals"]["reasoning_tokens"] == 15
    assert len(records) == 1


@pytest.mark.parametrize("phase", ["research", "candidate_cleanup", "partial_rewrite"])
def test_batch3_cost_exports_exclude_phase_subtotals(batch3_adapter, phase):
    problem, settings, candidate, _ = batch3_adapter
    events_path = settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl"
    events = [
        {"kind": "model.call", "payload": {
            "model": "gpt-5.4-mini", "in_tokens": 8032, "out_tokens": 855,
            "reasoning_tokens": 99, "cost_usd": 0.0098715,
        }},
        {"kind": "batch3.phase_end", "payload": {
            "phase": phase, "cost_usd": 0.0098715,
        }},
    ]
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = fp.ProblemResult(
        original_id=problem.original_id, safe_id=problem.safe_id,
        status="ok", returncode=0, run_id=problem.run_id,
        log_path=problem.log_path, output_tex_path=problem.output_tex_path,
        latex=candidate.read_text(), started_at=fp._utc_now(), finished_at=fp._utc_now(),
        duration_seconds=1,
    )

    asyncio.run(fp._write_aggregates(
        [problem], [result], settings, fp._utc_now(), fp.time.monotonic(), in_progress=False,
    ))

    records = [json.loads(line) for line in (settings.output_dir / "token_usage.jsonl").read_text().splitlines()]
    summary = json.loads((settings.output_dir / "run_summary.json").read_text())
    assert summary["author_parallelism"] == settings.author_parallelism == 1
    assert [record["event_type"] for record in records] == ["model.call"]
    assert summary["totals"] == {
        "input_tokens": 8032, "output_tokens": 855, "reasoning_tokens": 99,
        "total_tokens": 8887, "cost_usd": 0.0098715,
    }


@pytest.mark.parametrize("kind,payload", [
    ("model.call", {"model": "gpt-5.4-mini", "in_tokens": 100, "out_tokens": 20,
                    "reasoning_tokens": 5, "cost_usd": 0.1}),
    ("model.call", {"model": "gpt-5.4-mini", "in_tokens": 100, "out_tokens": 20,
                    "reasoning_out_tokens": 5, "cost_usd": 0.1, "via": "codex_exec_json"}),
    ("multiturn.end", {"model": "gpt-5.4-mini", "in_tokens": 100, "out_tokens": 20,
                       "reasoning_tokens": 5, "cost_usd": 0.1}),
    ("provider.usage", {"model_name": "gpt-5.4-mini", "usage": {
        "prompt_tokens": 100, "completion_tokens": 20, "reasoning_output_tokens": 5, "cost": 0.1,
    }}),
])
def test_usage_exports_keep_billable_events_and_healthchecks(batch3_adapter, kind, payload):
    problem, settings, _, _ = batch3_adapter
    (settings.output_dir / "healthcheck.json").write_text(json.dumps({"probes": [{
        "role": "Author", "model_ref": "models/openai/gpt-54-mini", "input_tokens": 10,
        "output_tokens": 2, "reasoning_tokens": 1, "cost_usd": 0.01,
    }]}))

    record = fp._usage_record(problem, {"kind": kind, "payload": payload})
    healthcheck_records, warnings = fp._collect_healthcheck_usage(settings)

    assert record is not None
    assert record["event_type"] == kind
    assert record["provider"] == "openai"
    assert not warnings
    assert len(healthcheck_records) == 1
    assert fp._token_totals([record, *healthcheck_records]) == {
        "input_tokens": 110, "output_tokens": 22, "reasoning_tokens": 6,
        "total_tokens": 132, "cost_usd": 0.11,
    }


def test_cleanup_allowance_estimates_export_without_fabricated_tokens(batch3_adapter):
    problem, _, _, _ = batch3_adapter
    record = fp._usage_record(problem, {"kind": "model.call", "payload": {
        "model": "gpt-6-astra", "cost_usd": 3, "cost_estimated": True,
        "usage_unavailable": True, "via": "cleanup_codex_allowance_estimate",
    }})
    assert record["cost_usd"] == 3
    assert record["usage_unavailable"] is True
    assert "input_tokens" not in record and "output_tokens" not in record
    assert fp._token_totals([record])["cost_usd"] == 3


def test_batch3_never_falls_back_to_unapproved_drafts(batch3_adapter):
    problem, settings, candidate, approve = batch3_adapter
    assert fp._find_solution_tex(problem, settings) is None
    approve(submission_approved=False)
    assert fp._find_solution_tex(problem, settings) is None
    approve()
    assert fp._find_solution_tex(problem, settings) == candidate
    candidate.write_text(candidate.read_text() + "% changed since review\n")
    assert fp._find_solution_tex(problem, settings) is None
    candidate.unlink()
    assert fp._find_solution_tex(problem, settings) is None


def test_batch3_refuses_post_review_normalization(batch3_adapter):
    problem, settings, candidate, approve = batch3_adapter
    candidate.write_text(r"\documentclass[10pt]{article}\begin{document}Proof.\end{document}")
    approve()
    assert fp._find_solution_tex(problem, settings) is None


def test_batch3_fail_open_finalizer_returns_only_abstention(batch3_adapter):
    problem, settings, _, _ = batch3_adapter
    latex, status, _, _ = asyncio.run(fp._ship_solution_or_fallback(
        problem, settings, reason="interrupted", fallback_status="stopped", solution_status="salvaged",
    ))
    assert status == "stopped"
    assert "UNAPPROVED RESEARCH" not in latex


@pytest.mark.parametrize("explicit_cutoff", [True, False])
@pytest.mark.parametrize("cleanup_seconds", [7200, 5400])
def test_batch3_queued_problem_gets_remaining_container_time(batch3_adapter, monkeypatch, explicit_cutoff, cleanup_seconds):
    problem, settings, _, _ = batch3_adapter
    captured = []

    class Process:
        returncode = 0

        def __init__(self):
            self.stdout = asyncio.StreamReader()

        async def wait(self):
            return 0

    async def spawn(*cmd, **kwargs):
        captured.extend(cmd)
        proc = Process()
        proc.stdout.feed_eof()
        return proc

    monkeypatch.setattr(fp.asyncio, "create_subprocess_exec", spawn)
    from proofstack import registry
    load = registry.load_preset

    def selected_preset(workflow):
        preset = load(workflow)
        return replace(preset, inputs={**preset.inputs, "partial_cleanup_seconds": cleanup_seconds})

    monkeypatch.setattr(registry, "load_preset", selected_preset)
    settings = replace(settings, deadline_at=fp.time.monotonic() + 300,
                       research_deadline_at=fp.time.monotonic() - 10 if explicit_cutoff else None,
                       compute_max_parallel_workers=2, author_parallelism=3)
    asyncio.run(fp._run_subprocess(problem, settings, n_rounds=50))
    limit = next(arg for arg in captured if arg.startswith("max_wallclock_s="))
    assert 230 < float(limit.split("=")[1]) <= 240
    assert "--restart-from" not in captured
    cutoff = next(arg for arg in captured if arg.startswith("research_deadline_unix_s="))
    assert float(cutoff.split("=")[1]) < fp.time.time()
    run_deadline = next(arg for arg in captured if arg.startswith("run_deadline_unix_s="))
    if not explicit_cutoff:
        assert float(run_deadline.split("=")[1]) - float(cutoff.split("=")[1]) == pytest.approx(cleanup_seconds)
    assert "compute_max_parallel_workers=2" in captured
    assert "author_parallelism=3" in captured


def _write_batch3_retry_schedule(problem, settings, *, save_research=True, **overrides):
    from proofstack.agents.ac.ac_workflow import _problem_hash

    schedule = {
        "problem_hash": _problem_hash(problem.text),
        "research_deadline_unix_s": fp.time.time() + 200,
        "run_deadline_unix_s": fp.time.time() + 500,
        "initial_usd": settings.budget_usd_per_question,
        "partial_reserve_usd": settings.budget_usd_per_question * 0.15,
        **overrides,
    }
    path = settings.output_dir / "workflow_runs" / problem.run_id / "batch3-schedule.json"
    path.write_text(json.dumps(schedule))
    if save_research:
        workspace = path.parent / "ac_workspaces" / f"{problem.safe_id}-{_problem_hash(problem.text)}"
        (workspace / ".ac").mkdir(parents=True, exist_ok=True)
        (workspace / "problem.txt").write_text(problem.text)
        (workspace / "answer.tex").write_text("saved manuscript")
        (workspace / ".ac/resume-state.json").write_text(json.dumps({"next_round": 1}))
    return path


@pytest.mark.parametrize("exit_code", [1, -9])
@pytest.mark.parametrize("retry_fails", [False, True])
@pytest.mark.parametrize("save_research", [True, False])
def test_batch3_early_failure_has_one_checkpoint_retry(batch3_adapter, monkeypatch, exit_code, retry_fails, save_research):
    problem, settings, _, approve = batch3_adapter
    attempts = []
    schedule_bytes = None

    async def run(p, s, **kwargs):
        nonlocal schedule_bytes
        assert p is problem and s is settings
        attempts.append(kwargs)
        path = s.output_dir / "workflow_runs" / p.run_id / "batch3-schedule.json"
        if len(attempts) == 1:
            assert kwargs["restart_from"] is None
            schedule_bytes = _write_batch3_retry_schedule(p, s, save_research=save_research).read_bytes()
            return exit_code
        assert path.read_bytes() == schedule_bytes
        assert kwargs == {**attempts[0], "restart_from": problem.run_id}
        if retry_fails:
            return exit_code
        approve(compiled=True, pages=1, early_stopped=True, rounds_completed=3)
        return 0

    async def compile(*args, **kwargs):
        return True, "OK"

    monkeypatch.setattr(fp, "_run_subprocess", run)
    monkeypatch.setattr(fp, "_verify_exact_latex_for_submission", compile)
    result = asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    assert len(attempts) == 2
    assert result.solved != retry_fails
    assert result.returncode == (exit_code if retry_fails else 0)
    assert problem.log_path.read_text().count("checkpoint retry 1/1") == 1


@pytest.mark.parametrize("reason", [
    "success", "parked", "cancelled", "spawn_error", "missing_schedule", "corrupt_schedule",
    "invalid_schedule", "wrong_problem", "expired_schedule", "adapter_drain", "budget",
    "published", "legacy", "programming_error", "programming_error_killed",
])
def test_batch3_does_not_retry_terminal_or_unsafe_failures(batch3_adapter, monkeypatch, reason):
    problem, settings, _, approve = batch3_adapter
    if reason == "legacy":
        settings = replace(settings, batch3=False, n_rounds=5, round_batch_size=5,
                           adaptive_continuation=False)
    if reason == "adapter_drain":
        settings = replace(settings, deadline_at=fp.time.monotonic() + 30)
    path = _write_batch3_retry_schedule(problem, settings)
    if reason == "missing_schedule":
        path.unlink()
    elif reason == "corrupt_schedule":
        path.write_text("{")
    elif reason == "invalid_schedule":
        _write_batch3_retry_schedule(problem, settings, run_deadline_unix_s=float("inf"))
    elif reason == "wrong_problem":
        _write_batch3_retry_schedule(problem, settings, problem_hash="wrong problem")
    elif reason == "expired_schedule":
        _write_batch3_retry_schedule(problem, settings, run_deadline_unix_s=fp.time.time() - 1)
    elif reason == "budget":
        (path.parent / "run-metadata.json").write_text(json.dumps({
            "status": "error", "error": "BudgetExhausted: usd",
        }))
    elif reason == "published":
        approve(compiled=True, pages=1, early_stopped=True)
    elif reason in {"programming_error", "programming_error_killed"}:
        (path.parent / "batch3-output.json").write_text(json.dumps({
            "error": "TypeError: broken adapter", "error_retryable": False,
        }))
    attempts = []

    async def run(*args, **kwargs):
        attempts.append(kwargs)
        if reason == "cancelled":
            raise asyncio.CancelledError
        if reason == "spawn_error":
            raise OSError("cannot spawn")
        return {"success": 0, "parked": 2, "programming_error": 1}.get(reason, -9)

    async def compile(*args, **kwargs):
        return True, "OK"

    monkeypatch.setattr(fp, "_run_subprocess", run)
    monkeypatch.setattr(fp, "_verify_exact_latex_for_submission", compile)
    asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    assert len(attempts) == 1
    assert attempts[0]["restart_from"] is None


def test_batch3_retry_command_preserves_limits_and_targets_existing_run(batch3_adapter, monkeypatch):
    problem, settings, _, _ = batch3_adapter
    commands = []
    settings = replace(settings, deadline_at=fp.time.monotonic() + 600,
                       research_deadline_at=fp.time.monotonic() + 200,
                       author_parallelism=3)

    class Process:
        returncode = -9

        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_eof()

        async def wait(self):
            return self.returncode

    async def spawn(*cmd, **kwargs):
        commands.append(cmd)
        if len(commands) == 1:
            _write_batch3_retry_schedule(problem, settings)
        return Process()

    monkeypatch.setattr(fp.asyncio, "create_subprocess_exec", spawn)
    asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    first, second = commands
    assert "--restart-from" not in first
    assert second[second.index("--restart-from") + 1] == problem.run_id
    for flag in ("--budget-usd", "--run-id", "--problem", "--workflow", "--output"):
        assert second[second.index(flag) + 1] == first[first.index(flag) + 1]
    assert "compute_max_parallel_workers=4" in first and "compute_max_parallel_workers=4" in second
    assert "author_parallelism=3" in first and "author_parallelism=3" in second
    for name in ("research_deadline_unix_s", "run_deadline_unix_s"):
        a, b = [float(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith(name + "=")))
                for cmd in commands]
        assert abs(a - b) < 0.1
    limits = [float(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("max_wallclock_s=")))
              for cmd in commands]
    assert 0 < limits[1] <= limits[0] < 540


def _publish_partial_fixture(problem, settings, candidate, **overrides):
    run = settings.output_dir / "workflow_runs" / problem.run_id
    partial = run / "partials" / "p.tex"
    partial.parent.mkdir(exist_ok=True)
    partial.write_text(candidate.read_text())
    outputs = {
        "submission_approved": False, "partial_ready": True, "compiled": True, "pages": 1,
        "output_kind": "partial_unreviewed", "rounds_completed": 3, "early_stopped": False,
        "partial_sha256": hashlib.sha256(partial.read_bytes()).hexdigest(), **overrides,
    }
    (run / "run-metadata.json").write_text(json.dumps({"outputs": outputs}))
    return partial, outputs


@pytest.mark.parametrize("exit_code", [1, -9])
@pytest.mark.parametrize("retry_fails", [False, True])
@pytest.mark.parametrize("manifest", [False, True])
def test_batch3_partial_checkpoint_does_not_disable_crash_retry(
    batch3_adapter, monkeypatch, exit_code, retry_fails, manifest,
):
    problem, settings, candidate, approve = batch3_adapter
    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    attempts = []
    saved_partial = None

    async def run(p, s, **kwargs):
        nonlocal saved_partial
        attempts.append(kwargs)
        if len(attempts) == 1:
            assert kwargs["restart_from"] is None
            _write_batch3_retry_schedule(p, s)
            saved_partial, outputs = _publish_partial_fixture(p, s, candidate)
            if manifest:
                (run_dir / "batch3-output.json").write_text(json.dumps(outputs))
            assert fp._find_solution_tex(p, s) == saved_partial
            return exit_code
        assert kwargs["restart_from"] == problem.run_id
        assert fp._find_solution_tex(p, s) == saved_partial
        if retry_fails:
            return exit_code
        approve(compiled=True, pages=1, early_stopped=True, rounds_completed=4)
        if manifest:
            outputs = json.loads((run_dir / "run-metadata.json").read_text())["outputs"]
            (run_dir / "batch3-output.json").write_text(json.dumps(outputs))
        return 0

    async def compile(*args, **kwargs):
        return True, "OK"

    monkeypatch.setattr(fp, "_run_subprocess", run)
    monkeypatch.setattr(fp, "_verify_exact_latex_for_submission", compile)
    result = asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    assert len(attempts) == 2
    assert result.solved is not retry_fails
    assert saved_partial.is_file()
    if retry_fails:
        assert result.status == "workflow_error_with_solution"
        assert result.latex == saved_partial.read_text()


def test_batch3_partial_is_exported_but_never_solved(batch3_adapter, monkeypatch):
    problem, settings, candidate, _ = batch3_adapter
    partial, _ = _publish_partial_fixture(problem, settings, candidate)
    assert fp._find_solution_tex(problem, settings) == partial
    assert not fp._author_critic_agreed(problem, settings)
    assert fp._workflow_output_rejection(problem, settings, min_rounds_completed=50) is None
    assert fp._stage_solution_candidate(problem, settings, n_rounds=50, returncode=0) == partial

    async def run(*args, **kwargs):
        return 0

    async def compile(*args, **kwargs):
        return True, "OK"

    monkeypatch.setattr(fp, "_run_subprocess", run)
    monkeypatch.setattr(fp, "_verify_exact_latex_for_submission", compile)
    result = asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    assert result.status == "partial_unreviewed" and not result.solved
    assert result.stages[0].status == "partial_unreviewed" and not result.stages[0].solved
    assert result.latex == partial.read_text()
    partial.write_text(partial.read_text() + "% changed\n")
    assert fp._find_solution_tex(problem, settings) is None


@pytest.mark.parametrize("overrides", [
    {"submission_approved": True}, {"compiled": False}, {"pages": 17},
    {"pages": 0}, {"partial_ready": False}, {"output_kind": "accepted_solution"},
])
def test_partial_gate_rejects_inconsistent_metadata(batch3_adapter, overrides):
    problem, settings, candidate, _ = batch3_adapter
    _publish_partial_fixture(problem, settings, candidate, **overrides)
    assert fp._find_solution_tex(problem, settings) is None
    assert not fp._author_critic_agreed(problem, settings)


def test_partial_checkpoint_survives_missing_terminal_metadata(batch3_adapter):
    problem, settings, candidate, _ = batch3_adapter
    partial, outputs = _publish_partial_fixture(problem, settings, candidate)
    run = settings.output_dir / "workflow_runs" / problem.run_id
    (run / "batch3-output.json").write_text(json.dumps(outputs))
    (run / "run-metadata.json").write_text(json.dumps({"status": "running"}))
    assert fp._find_solution_tex(problem, settings) == partial
    assert not fp._author_critic_agreed(problem, settings)


def test_batch3_checkpoint_overrides_stale_accepted_metadata(batch3_adapter):
    problem, settings, candidate, approve = batch3_adapter
    partial, outputs = _publish_partial_fixture(problem, settings, candidate)
    run = settings.output_dir / "workflow_runs" / problem.run_id
    (run / "batch3-output.json").write_text(json.dumps(outputs))
    approve(early_stopped=True)
    assert fp._find_solution_tex(problem, settings) == partial
    assert not fp._author_critic_agreed(problem, settings)


@pytest.mark.parametrize("checkpoint", ["{", "[]", "null", "{}"])
def test_invalid_batch3_checkpoint_never_replays_older_acceptance(batch3_adapter, checkpoint):
    problem, settings, candidate, approve = batch3_adapter
    approve(early_stopped=True)
    (candidate.parent.parent / "batch3-output.json").write_text(checkpoint)
    assert fp._find_solution_tex(problem, settings) is None
    assert not fp._author_critic_agreed(problem, settings)


@pytest.mark.parametrize("overrides", [{"publication_version": 2}, {"submission_sha256": "../outside"}])
def test_batch3_rejects_invalid_publication_paths(batch3_adapter, overrides):
    problem, settings, _, approve = batch3_adapter
    approve(**overrides)
    assert fp._find_solution_tex(problem, settings) is None


def test_cleanup_releases_all_research_slots_at_shared_cutoff(batch3_adapter):
    _, settings, _, _ = batch3_adapter

    async def run():
        configured = replace(settings, research_deadline_at=fp.time.monotonic() + .03)
        semaphore = asyncio.Semaphore(1)
        first_entered, second_entered = asyncio.Event(), asyncio.Event()
        release = asyncio.Event()

        async def first():
            async with fp._problem_slot(configured, semaphore):
                first_entered.set()
                await release.wait()

        async def second():
            await first_entered.wait()
            async with fp._problem_slot(configured, semaphore):
                second_entered.set()

        tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
        try:
            await asyncio.wait_for(second_entered.wait(), timeout=1)
            assert not tasks[0].done()
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert semaphore._value == 1

    asyncio.run(run())


def test_cancelled_research_slot_does_not_leak_permit(batch3_adapter):
    _, settings, _, _ = batch3_adapter

    async def run():
        configured = replace(settings, research_deadline_at=fp.time.monotonic() + 100)
        semaphore, entered = asyncio.Semaphore(1), asyncio.Event()

        async def worker():
            async with fp._problem_slot(configured, semaphore):
                entered.set()
                await asyncio.sleep(100)

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert semaphore._value == 1

    asyncio.run(run())


def test_renamed_batch3_preset_keeps_submission_policy(tmp_path, monkeypatch):
    _clear_firstproof_env(monkeypatch)
    copied = tmp_path / "rehearsal.yaml"
    copied.write_text((REPO_ROOT / "configs/workflows/firstproof_batch3.yaml").read_text())
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", str(copied))
    settings = fp._settings()
    assert settings.batch3 and settings.page_limit == 16
    assert settings.max_parallel == 10 and not settings.adaptive_continuation
    assert fp._round_schedule(settings) == [50]


def test_invalid_batch3_environment_is_clamped_not_fatal(monkeypatch):
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3")
    monkeypatch.setenv("FIRSTPROOF_N_ROUNDS", "600")
    monkeypatch.setenv("FIRSTPROOF_ADAPTIVE_CONTINUATION", "true")
    monkeypatch.setenv("FIRSTPROOF_PAGE_LIMIT", "20")
    settings = fp._settings()
    assert settings.n_rounds == 500 and settings.page_limit == 16
    assert not settings.adaptive_continuation and len(settings.warnings) >= 3
