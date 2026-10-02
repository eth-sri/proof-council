"""Real process-tree stop tests; child workflows never call a provider."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import firstproof_entrypoint as fp


def settings_for(root):
    return fp.Settings(
        input_path=root / "input.json", output_dir=root / "output", workflow="firstproof_batch3",
        max_parallel=2, page_limit=16, budget_usd_per_question=10, n_rounds=1, round_batch_size=1,
        compute_codex_sandbox="docker-bypass", runner_script=str(Path(__file__).resolve()),
        warnings=[], deadline_seconds=300, batch3=True,
    )


def adapter_fixture(root):
    settings = settings_for(root)
    fp._settings = lambda: settings
    fp._bootstrap_codex_auth = AsyncMock(return_value=(False, None))
    async def healthcheck(settings):
        return settings
    fp._run_healthcheck = healthcheck
    os.environ["FIRSTPROOF_TMP_PROBLEM_DIR"] = str(root / "inputs")
    return fp.main()


def child_fixture():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    root = output.parent.parent
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *args: sys.exit(130))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    run = output / run_id
    run.mkdir(parents=True, exist_ok=True)
    private = run / "provider-attempts.jsonl"
    private.write_text('{"test": true}\n')
    private.chmod(0o600)
    with (root / "launches.jsonl").open("a") as f:
        f.write(json.dumps({"pid": os.getpid(), "child": child.pid}) + "\n")
    time.sleep(60)
    return 0


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_operator_signal_stops_adapter_tree_without_relaunch_and_exports(tmp_path, sig):
    (tmp_path / "input.json").write_text(json.dumps({"problems": [
        {"id": "a", "latex": "Prove A"}, {"id": "b", "latex": "Prove B"}]}))
    proc = subprocess.Popen([sys.executable, __file__, "--adapter-fixture", str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        end = time.monotonic() + 15
        launches = []
        while time.monotonic() < end:
            try:
                launches = [json.loads(line) for line in (tmp_path / "launches.jsonl").read_text().splitlines()]
            except (OSError, ValueError):
                pass
            if len(launches) == 2:
                break
            assert proc.poll() is None, proc.communicate()[0]
            time.sleep(.02)
        assert len(launches) == 2
        proc.send_signal(sig)
        output = proc.communicate(timeout=25)[0]
        assert proc.returncode == 143, output
        assert len((tmp_path / "launches.jsonl").read_text().splitlines()) == 2
        summary = json.loads((tmp_path / "output/run_summary.json").read_text())
        assert summary["operator_stopped"] and not summary["in_progress"]
        assert all(p["status"].startswith("operator_stopped") for p in summary["per_problem"])
        assert all(path.stat().st_mode & 0o444 == 0o444 for path in (tmp_path / "output").rglob("provider-attempts.jsonl"))
        import psutil
        for row in launches:
            for pid in row.values():
                assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
        for row in locals().get("launches", []):
            for pid in row.values():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("code", [143, -15, 130, -2])
def test_child_operator_exit_is_never_retryable(tmp_path, code):
    settings = settings_for(tmp_path)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    assert not fp._batch3_can_retry(problem, settings, code)


@pytest.mark.parametrize("invalid", [None, "missing", "malformed", "other_problem", "expired", "terminal", "budget", "stop"])
def test_first_author_crash_retries_only_with_a_valid_schedule(tmp_path, invalid):
    settings = settings_for(tmp_path)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    from proofstack.agents.firstproof_batch3 import _problem_hash
    run = settings.output_dir / "workflow_runs/r"
    run.mkdir(parents=True)
    schedule = {"problem_hash": _problem_hash("P"),
        "research_deadline_unix_s": time.time() + 200, "run_deadline_unix_s": time.time() + 300,
        "initial_usd": 10, "partial_reserve_usd": 1}
    if invalid == "other_problem":
        schedule["problem_hash"] = _problem_hash("Q")
    if invalid == "expired":
        schedule["run_deadline_unix_s"] = time.time() - 1
    if invalid != "missing":
        (run / "batch3-schedule.json").write_text("{" if invalid == "malformed" else json.dumps(schedule))
    if invalid == "terminal":
        (run / "batch3-output.json").write_text(json.dumps({"error_retryable": False}))
    if invalid == "budget":
        (run / "run-metadata.json").write_text(json.dumps({"error": "BudgetExhausted"}))
    if invalid == "stop":
        settings.stop_requested.set()
    assert fp._batch3_can_retry(problem, settings, 1) is (invalid is None)


def test_live_aggregate_does_not_double_count_helper_rollups(tmp_path):
    settings = settings_for(tmp_path)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    run = settings.output_dir / "workflow_runs/r"
    run.mkdir(parents=True)
    events = [{"kind": "model.call", "payload": {"cost_usd": 2, "in_tokens": 100, "out_tokens": 20}},
              {"kind": "ac.author.delegate.wave_done", "payload": {"cost_usd": 2}},
              {"kind": "ac.author.subagent_done", "payload": {"usd": 2, "checkpoint_errors": ["failed"],
                  "error": "wave deadline reached"}}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    _, summary, _ = fp._aggregate_payloads([problem], [None], settings, fp._utc_now(), time.monotonic(), in_progress=True)
    assert summary["totals"]["cost_usd"] == 2
    assert summary["per_problem"][0]["health"]["helpers_artifact_degraded"] == 1
    assert summary["per_problem"][0]["health"]["helpers_timed_out"] == 1
    assert summary["updated_at"]


def test_operator_stop_does_not_start_submission_compilation(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    settings.stop_requested.set()
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    problem.output_tex_path.write_text("previous adapter export")
    async def unexpected(*a, **kw):
        raise AssertionError("stop started a new compilation")
    monkeypatch.setattr(fp, "_verified_solution_or_fallback", unexpected)
    latex, status, _, _ = asyncio.run(fp._ship_solution_or_fallback(problem, settings, reason="stop",
        fallback_status="cancelled", solution_status="cancelled_with_solution"))
    assert latex == "previous adapter export" and status == "operator_stopped"


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("invalid", [None, "tampered", "uncompiled", "overlong", "bad_metadata"])
@pytest.mark.parametrize("stop_reason", ["operator", "deadline"])
def test_shutdown_exports_only_validated_publications(tmp_path, monkeypatch, partial, invalid, stop_reason):
    settings = settings_for(tmp_path)
    if stop_reason == "operator":
        settings.stop_requested.set()
    else:
        settings = replace(settings, deadline_at=time.monotonic() - 1)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    problem.output_tex_path.write_text("previous adapter export")
    run = settings.output_dir / "workflow_runs/r"
    document = fp._ensure_complete_latex(r"\documentclass[12pt]{article}\begin{document}Saved result.\end{document}")
    digest = hashlib.sha256(document.encode()).hexdigest()
    publication = run / ("partials" if partial else "submissions") / "p" / f"{digest}.tex"
    publication.parent.mkdir(parents=True)
    publication.write_text(document + ("% changed" if invalid == "tampered" else ""))
    outputs = {"publication_version": 1, "partial_ready": partial, "submission_approved": not partial,
               "output_kind": "partial_unreviewed" if partial else "accepted_solution",
               "compiled": invalid != "uncompiled", "pages": 20 if invalid == "overlong" else 1,
               "partial_sha256" if partial else "submission_sha256": digest}
    (run / "batch3-output.json").write_text("{" if invalid == "bad_metadata" else json.dumps(outputs))
    async def unexpected(*a, **kw):
        raise AssertionError("stop started a new compilation")
    monkeypatch.setattr(fp, "_verified_solution_or_fallback", unexpected)
    result = asyncio.run(fp._exception_result(problem, RuntimeError("stop"), settings))
    status = "operator_stopped" if stop_reason == "operator" else "deadline_cancelled"
    assert result.status == (status + "_with_solution" if invalid is None else status)
    assert result.latex == (document if invalid is None else "previous adapter export")
    assert not result.solved
    assert problem.output_tex_path.read_text() == result.latex


def test_operator_stop_after_stage_snapshot_exports_newer_publication(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    settings.input_path.write_text(json.dumps({"problems": [{"id": "p", "latex": "P"}]}))
    monkeypatch.setenv("FIRSTPROOF_TMP_PROBLEM_DIR", str(tmp_path / "inputs"))
    monkeypatch.setattr(fp, "_bootstrap_codex_auth", AsyncMock(return_value=(False, None)))
    monkeypatch.setattr(fp, "_run_healthcheck", AsyncMock(return_value=settings))
    document = fp._ensure_complete_latex(r"\documentclass[12pt]{article}\begin{document}New partial.\end{document}")
    digest = hashlib.sha256(document.encode()).hexdigest()

    async def run(problem, settings, semaphore, on_stage_complete):
        old = await fp._exception_result(problem, RuntimeError("earlier stage failed"), settings)
        old.in_progress = True
        await on_stage_complete(old)
        root = settings.output_dir / "workflow_runs" / problem.run_id
        path = root / "partials/p" / f"{digest}.tex"
        path.parent.mkdir(parents=True)
        path.write_text(document)
        (root / "batch3-output.json").write_text(json.dumps({"publication_version": 1,
            "partial_ready": True, "submission_approved": False, "output_kind": "partial_unreviewed",
            "partial_sha256": digest, "compiled": True, "pages": 1}))
        settings.stop_requested.set()
        raise asyncio.CancelledError()

    monkeypatch.setattr(fp, "_run_problem", run)
    assert asyncio.run(fp._amain_run(settings)) == 143
    assert (settings.output_dir / "p.tex").read_text() == document
    summary = json.loads((settings.output_dir / "run_summary.json").read_text())
    assert summary["per_problem"][0]["status"] == "operator_stopped_with_solution"
    assert summary["completed_count"] == 1
    assert not summary["per_problem"][0]["solved"]


@pytest.mark.parametrize("phase", ["bootstrap", "healthcheck", "before_start"])
def test_operator_stop_during_startup_cancels_probe_and_never_launches_research(tmp_path, monkeypatch, phase):
    settings = settings_for(tmp_path)
    settings.input_path.write_text(json.dumps({"problems": [{"id": "p", "latex": "P"}]}))
    monkeypatch.setenv("FIRSTPROOF_TMP_PROBLEM_DIR", str(tmp_path / "inputs"))
    monkeypatch.setattr(fp, "_settings", lambda: settings)
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def probe(*args):
        entered.set()
        try:
            await asyncio.sleep(180)
        finally:
            cleaned.set()
    monkeypatch.setattr(fp, "_bootstrap_codex_auth", probe if phase == "bootstrap" else AsyncMock(return_value=(False, None)))
    monkeypatch.setattr(fp, "_run_healthcheck", probe)
    research = AsyncMock(side_effect=AssertionError("research launched after startup stop"))
    monkeypatch.setattr(fp, "_run_problem", research)

    async def scenario():
        if phase == "before_start":
            settings.stop_requested.set()
        task = asyncio.create_task(fp._amain())
        if phase != "before_start":
            await asyncio.wait_for(entered.wait(), 2)
            settings.stop_requested.set()
        assert await asyncio.wait_for(task, 2) == 143
    asyncio.run(scenario())
    assert cleaned.is_set() is (phase != "before_start")
    research.assert_not_called()
    summary = json.loads((settings.output_dir / "run_summary.json").read_text())
    assert summary["operator_stopped"] and not summary["in_progress"]
    assert summary["per_problem"][0]["status"] == "operator_stopped"
    assert (settings.output_dir / "p.tex").stat().st_mode & 0o444 == 0o444


def test_cancelled_auth_bootstrap_terminates_login_process(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("OPENAI_API_KEY", "fake-offline-key")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(fp.shutil, "which", lambda _: "/fake/codex")
    entered = asyncio.Event()
    async def communicate(data):
        assert data == b"fake-offline-key"
        entered.set()
        await asyncio.sleep(180)
    proc = SimpleNamespace(returncode=None, communicate=communicate)
    spawn = AsyncMock(return_value=proc)
    terminate = AsyncMock()
    monkeypatch.setattr(fp.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(fp, "_terminate_workflow_subprocess", terminate)
    async def scenario():
        task = asyncio.create_task(fp._bootstrap_codex_auth())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    terminate.assert_awaited_once_with(proc, grace_s=2.0)
    assert spawn.call_args.kwargs["start_new_session"]


def test_live_events_are_incremental_and_shared_by_usage_and_health(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    path = settings.output_dir / "workflow_runs/r/events.jsonl"
    path.parent.mkdir(parents=True)
    first = {"kind": "model.call", "call_id": "a", "payload": {"cost_usd": 2, "in_tokens": 100}}
    helper = {"kind": "ac.author.subagent_done", "payload": {"error": "wave deadline", "checkpoint_errors": ["failed"]}}
    path.write_text(json.dumps(first) + "\n" + json.dumps(helper) + "\n")
    consumed = []
    original = fp._EventSnapshot.consume
    def consume(self, problem, line, **kwargs):
        consumed.append(line)
        return original(self, problem, line, **kwargs)
    monkeypatch.setattr(fp._EventSnapshot, "consume", consume)
    def snapshot():
        return fp._aggregate_payloads([problem], [None], settings, "now", time.monotonic(), in_progress=True)[1]
    assert snapshot()["totals"]["cost_usd"] == 2
    assert snapshot()["per_problem"][0]["health"]["helpers_finished"] == 1
    assert len(consumed) == 2  # No second health scan and no unchanged-log scan.
    second = {"kind": "model.call", "call_id": "b", "payload": {"cost_usd": 3}}
    line = json.dumps(second)
    with path.open("a") as f:
        f.write(line[:20])
    assert snapshot()["totals"]["cost_usd"] == 2
    with path.open("a") as f:
        f.write(line[20:])
    assert snapshot()["totals"]["cost_usd"] == 5  # Valid final JSON without newline.
    assert snapshot()["totals"]["cost_usd"] == 5
    with path.open("a") as f:
        f.write("\n")
    assert snapshot()["totals"]["cost_usd"] == 5  # Committing the tail does not double count.
    parsed = len(consumed)
    assert snapshot()["totals"]["cost_usd"] == 5
    assert len(consumed) == parsed
    path.write_text(json.dumps(second) + "\n")  # Truncation resets both counters.
    summary = snapshot()
    assert summary["totals"]["cost_usd"] == 3
    assert summary["per_problem"][0]["health"]["helpers_finished"] == 0
    path.rename(path.with_suffix(".old"))
    path.write_text(json.dumps(first) + "\n")
    assert snapshot()["totals"]["cost_usd"] == 2


def test_incremental_snapshot_reconciles_receipts_without_double_counting(tmp_path):
    settings = settings_for(tmp_path)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    run = settings.output_dir / "workflow_runs/r"
    receipt = run / "agents/author/provider-attempts.jsonl"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"attempt_id": "a", "invocation_id": "i", "call_id": "c",
                                 "cost": 2, "input_tokens": 100, "usage_unavailable": False}) + "\n")
    # A receipt is useful even before events.jsonl exists.
    records, _ = fp._collect_token_usage([problem], settings)
    assert fp._token_totals(records)["cost_usd"] == 2
    (run / "events.jsonl").write_text(json.dumps({"kind": "model.call", "call_id": "c",
        "payload": {"cost_usd": 2, "in_tokens": 100}}) + "\n")
    for _ in range(2):
        records, _ = fp._collect_token_usage([problem], settings)
        assert fp._token_totals(records)["cost_usd"] == 2
        assert fp._token_totals(records)["input_tokens"] == 100


def test_rejected_existing_run_is_not_finalized(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    run = settings.output_dir / "workflow_runs/r"
    run.mkdir(parents=True)
    (run / "batch3-schedule.json").write_text("{}")
    monkeypatch.setattr(fp, "_settings", lambda: settings)
    def unexpected(*a, **kw):
        raise AssertionError("rejected launch modified existing artifacts")
    monkeypatch.setattr(fp, "_finalize_output_permissions", unexpected)
    assert asyncio.run(fp._amain()) == 2


def test_hard_exit_export_filters_secrets_and_makes_receipts_readable(tmp_path):
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    receipt = output / "provider-attempts.jsonl"
    receipt.write_text('{"response_id": "resp_one"}')
    receipt.chmod(0o600)
    secret = output / ".env"
    secret.write_text("API_KEY=private")
    secret.chmod(0o600)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/export_firstproof_outputs.py"), str(output)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode in (0, 1), result.stderr
    assert receipt.stat().st_mode & 0o444 == 0o444
    assert not secret.exists() or not (secret.stat().st_mode & 0o044)


def test_live_snapshot_refreshes_while_problem_is_still_running(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    settings.input_path.write_text(json.dumps({"problems": [{"id": "p", "latex": "P"}]}))
    monkeypatch.setenv("FIRSTPROOF_TMP_PROBLEM_DIR", str(tmp_path / "inputs"))
    monkeypatch.setattr(fp, "_bootstrap_codex_auth", AsyncMock(return_value=(False, None)))
    monkeypatch.setattr(fp, "LIVE_SNAPSHOT_INTERVAL_S", .01)
    async def healthcheck(settings):
        return settings
    monkeypatch.setattr(fp, "_run_healthcheck", healthcheck)
    async def scenario():
        running = False
        seen = asyncio.Event()
        async def snapshot(*a, **kw):
            if running and kw["in_progress"]:
                seen.set()
        async def problem(p, settings, *a, **kw):
            nonlocal running
            running = True
            await seen.wait()
            running = False
            return await fp._exception_result(p, RuntimeError("offline dummy"), settings)
        monkeypatch.setattr(fp, "_write_aggregates", snapshot)
        monkeypatch.setattr(fp, "_run_problem", problem)
        assert await asyncio.wait_for(fp._amain_run(settings), 2) == 0
        assert seen.is_set()
    asyncio.run(scenario())


def test_cancelled_snapshot_finishes_before_releasing_aggregate_lock(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    settings.output_dir.mkdir()
    entered, release = threading.Event(), threading.Event()
    def payloads(*a, **kw):
        entered.set()
        assert release.wait(3)
        return {}, {"in_progress": True}, []
    monkeypatch.setattr(fp, "_aggregate_payloads", payloads)
    async def scenario():
        task = asyncio.create_task(fp._write_aggregates([], [], settings, "now", 0, in_progress=True))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            await asyncio.sleep(.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (settings.output_dir / "run_summary.json").exists()
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["latex_first", "latex_second", "pdfinfo"])
@pytest.mark.parametrize("stop_reason", ["cancel", "deadline"])
def test_submission_validation_stops_process_tree(tmp_path, monkeypatch, stage, stop_reason):
    from proofstack.agents.writeup_loop import _GateCanceller  # noqa: F401
    import psutil

    settings = settings_for(tmp_path)
    if stop_reason == "deadline":
        settings = replace(settings, deadline_at=time.monotonic() + 2)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    marker = tmp_path / "child-pid"
    child_script = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    original_popen = subprocess.Popen
    processes, commands, workdirs = [], [], []
    target = {"latex_first": 1, "latex_second": 2, "pdfinfo": 3}[stage]
    def spawn(cmd, **kwargs):
        commands.append(cmd)
        script = child_script if len(commands) == target else (
            "from pathlib import Path; Path('solution.pdf').write_text('fake PDF')"
        )
        assert kwargs["start_new_session"]
        if kwargs.get("cwd") is not None:
            workdirs.append(kwargs["cwd"])
        proc = original_popen([sys.executable, "-c", script], **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(fp.subprocess, "Popen", spawn)
    monkeypatch.setitem(sys.modules, "fitz", None)

    async def scenario():
        task = asyncio.create_task(fp._verify_exact_latex_for_submission(problem, settings, "source"))
        try:
            async with asyncio.timeout(3):
                while not marker.exists() or not marker.read_text():
                    assert not task.done(), task.result()
                    await asyncio.sleep(.01)
            child_pid = int(marker.read_text())
            if stop_reason == "cancel":
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert len(commands) == target  # Cancellation never starts a later pass.
            assert all(proc.poll() is not None for proc in processes)
            assert not any(path.exists() for path in workdirs)
            async with asyncio.timeout(2):
                while psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE:
                    await asyncio.sleep(.01)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for proc in processes:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=3)
    asyncio.run(scenario())


@pytest.mark.parametrize("stop_reason", ["deadline", "operator"])
def test_submission_validation_does_not_spawn_after_stop(tmp_path, monkeypatch, stop_reason):
    settings = settings_for(tmp_path)
    if stop_reason == "deadline":
        settings = replace(settings, deadline_at=time.monotonic() - 1)
    else:
        settings.stop_requested.set()
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    def unexpected(*args, **kwargs):
        raise AssertionError("compiler launched after stop")
    monkeypatch.setattr(fp.subprocess, "Popen", unexpected)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fp._verify_exact_latex_for_submission(problem, settings, "source"))


@pytest.mark.parametrize("status", [None, "ok", "partial_unreviewed", "adapter_error", "deadline_cancelled"])
def test_per_problem_totals_survive_completion(tmp_path, status):
    settings = settings_for(tmp_path)
    settings.output_dir.mkdir()
    (settings.output_dir / "healthcheck.json").write_text(json.dumps({"probes": [{
        "role": "Author", "model_ref": "models/openai/gpt-54-mini",
        "input_tokens": 10, "output_tokens": 2, "cost_usd": .01,
    }]}))
    problems = []
    for name, cost in (("p", 2), ("q", 7)):
        problem = fp.Problem(0, name, name, "P", None, tmp_path / name, tmp_path / (name + ".log"),
                             tmp_path / (name + ".tex"), "r-" + name)
        problems.append(problem)
        path = settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"kind": "model.call", "payload": {
            "cost_usd": cost, "in_tokens": 100, "out_tokens": 20, "reasoning_tokens": 5,
        }}) + "\n")
    problem = problems[0]
    result = None if status is None else fp.ProblemResult(
        original_id="p", safe_id="p", status=status, returncode=0, run_id=problem.run_id,
        log_path=problem.log_path, output_tex_path=problem.output_tex_path, latex="source",
        started_at="now", finished_at="now", duration_seconds=1,
    )
    _, summary, _ = fp._aggregate_payloads(problems, [result, None], settings, "now", time.monotonic(),
                                          in_progress=True)
    assert summary["per_problem"][0]["totals"] == {
        "cost_usd": 2, "input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 5, "total_tokens": 120,
    }
    assert summary["per_problem"][1]["totals"]["cost_usd"] == 7
    assert summary["totals"]["cost_usd"] == 9.01  # Healthcheck spend stays batch-level.


if __name__ == "__main__":
    raise SystemExit(adapter_fixture(Path(sys.argv[2])) if sys.argv[1] == "--adapter-fixture" else child_fixture())
