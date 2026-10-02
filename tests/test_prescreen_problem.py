from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def intake(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "intake_driver", ROOT / "scripts" / "prescreen_problem.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    problem_dir = tmp_path / "problem"
    submission = problem_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "problem.tex").write_text("ORIGINAL STATEMENT")
    metadata = problem_dir / "metadata.yaml"
    metadata.write_text("title: Original\nprescreen:\n  reviewer_note: keep me\n")
    output = {
        "report": "## Report\nLooks suitable.",
        "cleaned_latex": "```latex\nCLEANED STATEMENT\n```",
        "verdict": (
            "```yaml\nverdict: well-posed\nsummary: Looks suitable.\n"
            "suitable_for_test: true\nflags: []\n```"
        ),
    }
    calls = []
    contexts = []

    class Workflow:
        def __init__(self, ctx):
            pass

        async def __call__(self, **inputs):
            calls.append(inputs)
            return dict(output)

    preset = SimpleNamespace(
        name="prescreen",
        budget=object(),
        component_configs={"cfg_prescreen": {"model": "models/openai/gpt-6-astra-pro"}},
        model_overrides={},
        workflow_cls=Workflow,
        build_inputs=lambda **kwargs: {"problem": kwargs["problem"], "problem_id": kwargs["problem_id"]},
    )
    monkeypatch.setattr(driver, "load_preset", lambda name: preset)
    def context(**kwargs):
        contexts.append(kwargs)
        root = kwargs["root_workdir"] / kwargs["run_id"]
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(root_workdir=root)

    monkeypatch.setattr(driver, "RunContext", SimpleNamespace(create=context))

    def render(report, destination):
        destination.write_bytes(b"PDF PLACEHOLDER")

    original_render = driver._render_pdf
    monkeypatch.setattr(driver, "_render_pdf", render)

    def run(*args):
        monkeypatch.setattr(sys, "argv", [
            "prescreen_problem.py", "--problem-dir", str(problem_dir),
            "--output", str(tmp_path / "runs"), *args,
        ])
        return asyncio.run(driver.amain())

    return SimpleNamespace(
        driver=driver, problem_dir=problem_dir, submission=submission,
        metadata=metadata, output=output, calls=calls, contexts=contexts, run=run,
        original_render=original_render,
    )


def test_success_files_outputs_and_preserves_metadata(intake):
    assert intake.run() == 0
    ps = yaml.safe_load(intake.metadata.read_text())
    assert ps["title"] == "Original"
    assert ps["prescreen"]["reviewer_note"] == "keep me"
    assert ps["prescreen"]["status"] == "done"
    assert ps["prescreen"]["suitable_for_test"] is True
    assert ps["prescreen"]["model"] == "models/openai/gpt-6-astra-pro"
    assert (intake.problem_dir / "cleaned/problem_clean.tex").read_text() == "CLEANED STATEMENT\n"
    assert len(intake.calls) == 1
    with pytest.raises(SystemExit, match="already exists"):
        intake.run()
    assert len(intake.calls) == 1


def test_model_override_is_used_and_recorded(intake):
    assert intake.run("--model", "models/openai/gpt-6-astra-max") == 0
    expected = "models/openai/gpt-6-astra-max"
    assert intake.contexts[0]["component_configs"]["cfg_prescreen"]["model"] == expected
    assert yaml.safe_load(intake.metadata.read_text())["prescreen"]["model"] == expected


def test_multiple_statements_require_explicit_selection(intake):
    (intake.submission / "problem_conj1_only.tex").write_text("LIVE CONJECTURE ONE")
    with pytest.raises(SystemExit, match="--problem-file"):
        intake.run()
    assert not intake.calls


def test_explicit_live_statement_replaces_default_selection(intake):
    revised = intake.submission / "problem_conj1_only.tex"
    revised.write_text("LIVE CONJECTURE ONE")
    assert intake.run("--problem-file", "submission/problem_conj1_only.tex") == 0
    assert intake.calls[0]["problem"] == "LIVE CONJECTURE ONE"
    assert intake.contexts[0]["config_snapshot"]["submission_path"] == str(revised)
    assert yaml.safe_load(intake.metadata.read_text())["prescreen"]["submission_path"] == "submission/problem_conj1_only.tex"


def test_empty_statement_is_rejected_before_model_call(intake):
    (intake.submission / "problem.tex").write_text(" \n")
    with pytest.raises(SystemExit, match="empty"):
        intake.run()
    assert not intake.calls


@pytest.mark.parametrize("verdict", [
    "{}", "[]", "a scalar", "null",
    "verdict: well-posed\nsummary: Fine\nflags: []",
    "verdict: invented\nsummary: Fine\nsuitable_for_test: true\nflags: []",
    "verdict: well-posed\nsummary: Fine\nsuitable_for_test: 2\nflags: []",
    "verdict: well-posed\nsummary: Fine\nsuitable_for_test: true\nflags: easy",
    "verdict: well-posed\nsummary: Fine\nsuitable_for_test: true\nflags: [123]",
    "verdict: well-posed\nsummary: []\nsuitable_for_test: true\nflags: []",
    "verdict: [broken",
])
def test_invalid_verdict_does_not_replace_filed_outputs(intake, verdict):
    assert intake.run() == 0
    files = [intake.metadata, intake.problem_dir / "prescreen/response.json",
             intake.problem_dir / "prescreen/report.md",
             intake.problem_dir / "prescreen/report.pdf",
             intake.problem_dir / "cleaned/problem_clean.tex"]
    before = {p: p.read_bytes() for p in files}
    intake.output["verdict"] = verdict
    intake.output["report"] = "REPLACEMENT REPORT"
    intake.output["cleaned_latex"] = "REPLACEMENT STATEMENT"
    with pytest.raises(SystemExit, match="verdict"):
        intake.run("--force")
    assert {p: p.read_bytes() for p in files} == before


@pytest.mark.parametrize("metadata", ["- item\n", "prescreen: pending\n", "title: [broken\n"])
def test_invalid_metadata_fails_before_paid_work(intake, metadata):
    intake.metadata.write_text(metadata)
    with pytest.raises(SystemExit, match="metadata"):
        intake.run()
    assert not intake.calls
    assert intake.metadata.read_text() == metadata


def test_multiline_verdict_and_false_string_round_trip(intake):
    intake.output["verdict"] = (
        "verdict: mathematical-issues\nsummary: |\n  First line.\n  Second line.\n"
        "suitable_for_test: 'false'\nflags: [statement-incorrect]\n"
    )
    assert intake.run() == 0
    ps = yaml.safe_load(intake.metadata.read_text())["prescreen"]
    assert ps["suitable_for_test"] is False
    assert ps["summary"] == "First line.\nSecond line.\n"
    assert ps["flags"] == ["statement-incorrect"]


@pytest.mark.parametrize("field,value", [
    ("report", []), ("cleaned_latex", {}), ("verdict", None),
    ("cleaned_latex", "```latex\n\n```"), ("report", "  "),
    ("error", "provider failed"), ("last_gasp", True),
])
def test_bad_workflow_output_is_not_filed(intake, field, value):
    before = intake.metadata.read_bytes()
    intake.output[field] = value
    with pytest.raises(SystemExit):
        intake.run()
    assert intake.metadata.read_bytes() == before
    assert not (intake.problem_dir / "prescreen/response.json").exists()
    assert not (intake.problem_dir / "cleaned/problem_clean.tex").exists()
    root = intake.contexts[0]["root_workdir"] / intake.contexts[0]["run_id"]
    assert (root / "prescreen-response.json").is_file()


def test_failed_pdf_rerender_does_not_keep_old_pdf(intake, monkeypatch, capsys):
    assert intake.run() == 0
    old_pdf = intake.problem_dir / "prescreen/report.pdf"
    assert old_pdf.is_file()

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("pandoc", 120)

    monkeypatch.setattr(intake.driver, "_render_pdf", fail)
    assert intake.run("--force") == 0
    assert not old_pdf.exists()
    assert "FAILED" in capsys.readouterr().out


def test_no_metadata_file_is_created_when_absent(intake):
    intake.metadata.unlink()
    assert intake.run() == 0
    assert not intake.metadata.exists()


def test_real_preset_and_runtime_are_used_without_network(intake, monkeypatch):
    from proofstack import RunContext
    from proofstack.agents.configurable_prompt import ConfigurablePromptAgent
    from proofstack.registry import load_preset

    observed = []

    async def no_network(self):
        raise AssertionError("offline test must not create an API client")

    async def reply(self, inputs):
        observed.append(inputs.problem)
        self.render_messages(inputs)
        return self.parse_output("".join(
            f"<{key}>{value}</{key}>" for key, value in intake.output.items()
        ), inputs)

    monkeypatch.setattr(intake.driver, "RunContext", RunContext)
    monkeypatch.setattr(intake.driver, "load_preset", load_preset)
    monkeypatch.setattr(ConfigurablePromptAgent, "_get_client", no_network)
    monkeypatch.setattr(ConfigurablePromptAgent, "run", reply)
    assert intake.run() == 0
    assert observed == ["ORIGINAL STATEMENT"]
    preset = load_preset("prescreen")
    assert preset.budget.max_usd == 80
    assert preset.budget.max_wallclock_s == 14400
    assert yaml.safe_load(intake.metadata.read_text())["prescreen"]["status"] == "done"
    run_dir, = (intake.problem_dir.parent / "runs").iterdir()
    assert intake.run("--force", "--resume-from", str(run_dir)) == 0
    assert observed == ["ORIGINAL STATEMENT"]


@pytest.mark.parametrize("selection", [[], ["--problem-file", "submission/problem.tex"]])
def test_symlinked_submission_dir_is_filed_with_logical_path(intake, tmp_path, selection):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "problem.tex").write_text("STATEMENT VIA SYMLINK")
    for path in intake.submission.iterdir():
        path.unlink()
    intake.submission.rmdir()
    intake.submission.symlink_to(outside, target_is_directory=True)
    assert intake.run(*selection) == 0
    assert intake.calls[0]["problem"] == "STATEMENT VIA SYMLINK"
    ps = yaml.safe_load(intake.metadata.read_text())["prescreen"]
    assert ps["submission_path"] == "submission/problem.tex"


@pytest.mark.skipif(os.name != "posix", reason="renderer uses POSIX process groups")
@pytest.mark.parametrize("interrupted", [False, True])
def test_renderer_cleans_up_children_on_timeout_or_interrupt(intake, monkeypatch, tmp_path, interrupted):
    original_popen = subprocess.Popen
    pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(60)"
    )
    parents = []

    def spawn(cmd, **kwargs):
        assert cmd[0] == "pandoc"
        assert kwargs["start_new_session"] is True
        proc = original_popen([sys.executable, "-c", code], **kwargs)
        parents.append(proc)
        if interrupted:
            communicate = proc.communicate

            def interrupt(**kwargs):
                with pytest.raises(subprocess.TimeoutExpired):
                    communicate(**kwargs)
                raise KeyboardInterrupt

            proc.communicate = interrupt
        return proc

    monkeypatch.setattr(intake.driver.subprocess, "Popen", spawn)
    child = None
    try:
        expected = KeyboardInterrupt if interrupted else subprocess.TimeoutExpired
        with pytest.raises(expected):
            intake.original_render(tmp_path / "report.md", tmp_path / "out.pdf", timeout=1)
        pid = int(pid_path.read_text())
        with contextlib.suppress(psutil.NoSuchProcess):
            child = psutil.Process(pid)
        deadline = time.monotonic() + 5
        while child is not None and time.monotonic() < deadline:
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        else:
            if child is not None:
                pytest.fail("renderer child survived cleanup")
        assert parents[0].poll() is not None
        assert parents[0].stdout.closed
        assert parents[0].stderr.closed
    finally:
        for proc in parents:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if child is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()


@pytest.mark.parametrize("returncode", [0, 1])
def test_renderer_handles_normal_exit(intake, monkeypatch, tmp_path, returncode):
    original_popen = subprocess.Popen

    def spawn(cmd, **kwargs):
        return original_popen([
            sys.executable, "-c",
            f"import sys; sys.stderr.write('compile diagnostic'); sys.exit({returncode})",
        ], **kwargs)

    monkeypatch.setattr(intake.driver.subprocess, "Popen", spawn)
    if returncode:
        with pytest.raises(RuntimeError, match="compile diagnostic"):
            intake.original_render(tmp_path / "report.md", tmp_path / "out.pdf")
    else:
        intake.original_render(tmp_path / "report.md", tmp_path / "out.pdf")


def _run_dir(intake):
    ctx = intake.contexts[-1]
    return ctx["root_workdir"] / ctx["run_id"]


@pytest.mark.parametrize("failed_file", [
    "prescreen/report.md", "cleaned/problem_clean.tex", "metadata.yaml", "prescreen/response.json",
])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_filing_can_resume_without_model_call(intake, monkeypatch, failed_file, existing):
    if existing:
        assert intake.run() == 0
    original_metadata = intake.metadata.read_bytes()
    intake.output["report"] = "NEW REPORT"
    intake.output["cleaned_latex"] = "NEW CLEANED STATEMENT"
    target = intake.problem_dir / failed_file
    replace = Path.replace

    def fail_replace(path, destination):
        if destination == target:
            raise OSError(28, "No space left on device")
        return replace(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="No space left"):
            intake.run("--force", "--model", "models/openai/gpt-6-astra-max")
    assert not (intake.problem_dir / "prescreen/response.json").exists()
    if failed_file != "prescreen/response.json":
        assert intake.metadata.read_bytes() == original_metadata
    before_calls = len(intake.calls)
    before_contexts = len(intake.contexts)

    def no_model(*args, **kwargs):
        pytest.fail("filing recovery must not load a preset or create a workflow")

    monkeypatch.setattr(intake.driver, "load_preset", no_model)
    assert intake.run("--resume-from", str(_run_dir(intake))) == 0
    assert len(intake.calls) == before_calls
    assert len(intake.contexts) == before_contexts
    assert (intake.problem_dir / "prescreen/report.md").read_text() == "NEW REPORT\n"
    assert (intake.problem_dir / "cleaned/problem_clean.tex").read_text() == "NEW CLEANED STATEMENT\n"
    assert json.loads((intake.problem_dir / "prescreen/response.json").read_text()) == intake.output
    meta = yaml.safe_load(intake.metadata.read_text())
    assert meta["prescreen"]["status"] == "done"
    assert meta["prescreen"]["reviewer_note"] == "keep me"
    assert meta["prescreen"]["model"] == "models/openai/gpt-6-astra-max"


def test_atomic_metadata_write_preserves_original_on_partial_write(intake, monkeypatch):
    before = intake.metadata.read_bytes()
    write = Path.write_text

    def partial_write(path, text, **kwargs):
        if path.name == "metadata.yaml":
            write(path, "truncated", **kwargs)
            raise OSError(28, "No space left on device")
        return write(path, text, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", partial_write)
        with pytest.raises(OSError, match="No space left"):
            intake.run()
    assert intake.metadata.read_bytes() == before
    assert not (intake.problem_dir / "prescreen/response.json").exists()
    assert not list(intake.problem_dir.glob(".metadata.yaml-*"))
    assert intake.run("--resume-from", str(_run_dir(intake))) == 0
    assert len(intake.calls) == 1


@pytest.mark.parametrize("field,value", [
    ("problem_id", "another-problem"), ("submission_path", "submission/revised.tex"),
    ("sha256", "incorrect"), ("model", None),
])
def test_resume_rejects_mismatched_or_incomplete_provenance(intake, field, value):
    assert intake.run() == 0
    saved_path = _run_dir(intake) / "prescreen-input.json"
    saved = json.loads(saved_path.read_text())
    saved[field] = value
    saved_path.write_text(json.dumps(saved))
    before = intake.metadata.read_bytes()
    with pytest.raises(SystemExit, match="cannot resume filing"):
        intake.run("--force", "--resume-from", str(_run_dir(intake)))
    assert len(intake.calls) == 1
    assert intake.metadata.read_bytes() == before
    assert (intake.problem_dir / "prescreen/response.json").exists()


def test_resume_rejects_changed_statement(intake):
    assert intake.run() == 0
    (intake.submission / "problem.tex").write_text("REVISED STATEMENT")
    with pytest.raises(SystemExit, match="does not match"):
        intake.run("--force", "--resume-from", str(_run_dir(intake)))
    assert len(intake.calls) == 1


@pytest.mark.parametrize("filename", ["prescreen-input.json", "prescreen-response.json"])
@pytest.mark.parametrize("broken", ["missing", "invalid"])
def test_resume_rejects_missing_or_invalid_checkpoint_without_paid_fallback(intake, filename, broken):
    assert intake.run() == 0
    path = _run_dir(intake) / filename
    if broken == "missing":
        path.unlink()
    else:
        path.write_text("{")
    with pytest.raises(SystemExit, match="cannot resume filing"):
        intake.run("--force", "--resume-from", str(_run_dir(intake)))
    assert len(intake.calls) == 1


def test_resume_revalidates_saved_response_before_filing(intake):
    assert intake.run() == 0
    raw_path = _run_dir(intake) / "prescreen-response.json"
    response = json.loads(raw_path.read_text())
    response["verdict"] = "{}"
    raw_path.write_text(json.dumps(response))
    before = intake.metadata.read_bytes()
    with pytest.raises(SystemExit, match="verdict"):
        intake.run("--force", "--resume-from", str(_run_dir(intake)))
    assert len(intake.calls) == 1
    assert intake.metadata.read_bytes() == before


def test_resume_cannot_override_original_model(intake):
    with pytest.raises(SystemExit):
        intake.run("--resume-from", "unused", "--model", "different")
    assert not intake.calls


def test_interrupted_filing_can_resume(intake, monkeypatch):
    assert intake.run() == 0
    intake.output["report"] = "REVISED REPORT"
    before = intake.metadata.read_bytes()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(intake.driver, "_render_pdf", interrupt)
        with pytest.raises(KeyboardInterrupt):
            intake.run("--force")
    assert not (intake.problem_dir / "prescreen/response.json").exists()
    assert intake.metadata.read_bytes() == before
    assert intake.run("--resume-from", str(_run_dir(intake))) == 0
    assert len(intake.calls) == 2
    assert (intake.problem_dir / "prescreen/report.md").read_text() == "REVISED REPORT\n"


def test_atomic_metadata_update_preserves_symlink_and_permissions(intake, tmp_path):
    target = tmp_path / "shared-metadata.yaml"
    intake.metadata.rename(target)
    target.chmod(0o640)
    intake.metadata.symlink_to(target)
    assert intake.run() == 0
    assert intake.metadata.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o640
    assert yaml.safe_load(target.read_text())["prescreen"]["status"] == "done"


def test_resume_without_metadata_does_not_create_it(intake):
    intake.metadata.unlink()
    assert intake.run() == 0
    assert intake.run("--force", "--resume-from", str(_run_dir(intake))) == 0
    assert len(intake.calls) == 1
    assert not intake.metadata.exists()


def test_provenance_write_failure_prevents_model_call(intake, monkeypatch):
    replace = Path.replace

    def fail(path, destination):
        if destination.name == "prescreen-input.json":
            raise OSError(28, "No space left on device")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="No space left"):
        intake.run()
    assert not intake.calls
