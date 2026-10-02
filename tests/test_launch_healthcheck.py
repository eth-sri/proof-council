import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec

import pytest

from mathagents.api_client import _is_terminal_api_error
from proofstack import healthcheck
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.registry import load_preset
from proofstack.sandbox.docker import DockerSandbox
from proofstack.sandbox.subprocess import SubprocessSandbox
from proofstack.agents import cleanup_session as cleanup
from proofstack.sandbox.memory import GiB, MemoryRegistryError


@pytest.fixture(autouse=True)
def offline_cleanup_probe(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(cleanup, "check_cleanup_available", probe)
    return probe


@pytest.mark.parametrize("text", ["maximum context length", "input token count", "context_length_exceeded"])
def test_context_400_reaches_existing_recovery(text):
    exc = RuntimeError(text)
    exc.status_code = 400
    assert not _is_terminal_api_error(exc, context_length_recovery=True)
    # Other provider paths do not implement token-halving recovery.
    assert _is_terminal_api_error(exc)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_real_request_errors_remain_terminal(status):
    exc = RuntimeError("invalid request")
    exc.status_code = status
    assert _is_terminal_api_error(exc)
    assert _is_terminal_api_error(exc, response_poll=True) == (status != 404)


@pytest.mark.parametrize("success", [True, False])
def test_native_launch_gate_uses_actual_compute_policy(tmp_path, monkeypatch, success):
    seen = []
    handle = SimpleNamespace(wait=AsyncMock(return_value=0), terminate=AsyncMock(), worker_stopped=True)
    stream = AsyncMock(return_value=handle)
    class Sandbox:
        stream_command = stream
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    def make(spec, **kwargs):
        seen.append(spec)
        return Sandbox()
    probe = AsyncMock(side_effect=None if success else RuntimeError("host failed"))
    monkeypatch.setattr(healthcheck, "make_sandbox", make)
    monkeypatch.setattr(healthcheck, "_require_codex_cli_version", probe)
    preset = load_preset("firstproof_batch3")
    inputs = FirstProofBatch3Workflow.Inputs(**preset.build_inputs(problem="p", problem_id="p"))
    if success:
        asyncio.run(healthcheck.check_compute(inputs.model_dump(), tmp_path))
    else:
        with pytest.raises(RuntimeError, match="Launch healthcheck failed"):
            asyncio.run(healthcheck.check_compute(inputs.model_dump(), tmp_path))
    report = json.loads((tmp_path / "healthcheck.json").read_text())
    assert report["ok"] == success and report["paid_calls"] == 0
    assert seen[0].memory_gb == 8 and not seen[0].limit_address_space
    assert seen[0].provider_keys == ()
    assert seen[0].memory_policy.max_workers == 4
    assert seen[0].memory_policy.registry == tmp_path / "workflow_runs/.proofcouncil-compute-memory"
    probe.assert_awaited_once()
    if success:
        stream.assert_awaited_once_with(["codex", "--version"], timeout_s=30)
        handle.terminate.assert_awaited_once()
        assert report["probes"] == [{"name": "streamed_worker_admission", "ok": True, "worker_stopped": True}]
    else:
        stream.assert_not_awaited()


@pytest.mark.parametrize("backend,sandbox_cls", [("subprocess", SubprocessSandbox), ("docker", DockerSandbox)])
def test_launch_gate_respects_real_backend_signature(tmp_path, monkeypatch, backend, sandbox_cls):
    sandbox = create_autospec(sandbox_cls, instance=True, spec_set=True)
    sandbox.__aenter__.return_value = sandbox
    handle = SimpleNamespace(wait=AsyncMock(return_value=0), terminate=AsyncMock(), worker_stopped=True)
    sandbox.stream_command.return_value = handle
    monkeypatch.setattr(healthcheck, "make_sandbox", lambda *args, **kwargs: sandbox)
    monkeypatch.setattr(healthcheck, "_require_codex_cli_version", AsyncMock())

    report = asyncio.run(healthcheck.check_compute({
        "compute_sandbox_backend": backend,
        "compute_max_parallel_workers": 4 if backend == "subprocess" else 0,
    }, tmp_path))

    assert report["ok"] and report["compute"]["backend"] == backend
    sandbox.stream_command.assert_awaited_once_with(["codex", "--version"], timeout_s=30)
    handle.wait.assert_awaited_once_with(timeout_s=30)
    handle.terminate.assert_awaited_once()
    sandbox.__aexit__.assert_awaited_once()


def test_launch_gate_rejects_restored_memory_policy_mismatch(tmp_path, monkeypatch):
    registry = tmp_path / "workflow_runs/.proofcouncil-compute-memory"
    registry.mkdir(parents=True)
    control = registry / "control.json"
    old_policy = json.dumps({"max_workers": 6, "worker_bytes": 8 * 1024**3, "reserve_bytes": 16 * 1024**3})
    control.write_text(old_policy)
    monkeypatch.setattr(healthcheck, "_require_codex_cli_version", AsyncMock())
    inputs = {"compute_sandbox_backend": "subprocess", "compute_max_parallel_workers": 4,
              "compute_memory_gb": 8, "compute_memory_reserve_gb": 16}
    with pytest.raises(RuntimeError, match="same memory policy"):
        asyncio.run(healthcheck.check_compute(inputs, tmp_path))
    assert control.read_text() == old_policy
    report = json.loads((tmp_path / "healthcheck.json").read_text())
    assert not report["ok"] and report["paid_calls"] == 0


@pytest.mark.parametrize("failure", ["exit", "memory", "timeout", "cancel", "cleanup"])
def test_admission_probe_failures_never_pass_and_always_terminate(tmp_path, monkeypatch, failure):
    handle = SimpleNamespace(
        wait=AsyncMock(return_value=1 if failure == "exit" else 0), terminate=AsyncMock(),
        memory_failure="worker_memory_limit" if failure == "memory" else None,
        worker_stopped=failure != "cleanup",
    )
    if failure in ("timeout", "cancel"):
        handle.wait.side_effect = TimeoutError() if failure == "timeout" else asyncio.CancelledError()

    class Sandbox:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def stream_command(self, *args, **kwargs):
            return handle

    monkeypatch.setattr(healthcheck, "make_sandbox", lambda *args, **kwargs: Sandbox())
    monkeypatch.setattr(healthcheck, "_require_codex_cli_version", AsyncMock())
    expected = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected):
        asyncio.run(healthcheck.check_compute({"compute_sandbox_backend": "subprocess"}, tmp_path))
    handle.terminate.assert_awaited_once()
    report = json.loads((tmp_path / "healthcheck.json").read_text())
    assert not report["ok"] and report["paid_calls"] == 0


@pytest.mark.parametrize("mode", [None, "off", "warn", "strict"])
def test_batch3_adapter_always_requires_gate(tmp_path, monkeypatch, mode):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    if mode is not None:
        monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    monkeypatch.delenv("PROOFSTACK_SANDBOX_BACKEND", raising=False)
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    probe = AsyncMock(side_effect=RuntimeError("native host unavailable"))
    monkeypatch.setattr(healthcheck, "check_compute", probe)
    if mode == "strict":
        with pytest.raises(RuntimeError, match="native host unavailable"):
            asyncio.run(fp._run_healthcheck(settings))
    else:
        updated = asyncio.run(fp._run_healthcheck(settings))
        assert updated.compute_disabled_reason == "native host unavailable"
        assert any("disabling Compute for every problem" in warning for warning in updated.warnings)
    probe.assert_awaited_once()
    assert probe.call_args.args[0]["compute_sandbox_backend"] == "subprocess"


@pytest.mark.parametrize("mode", ["off", "warn", "strict"])
def test_non_batch3_healthcheck_policy_is_unchanged(tmp_path, monkeypatch, mode):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    settings = replace(fp._settings(), workflow="firstproof_submission", batch3=False, output_dir=tmp_path)
    probe = AsyncMock(side_effect=RuntimeError("native host unavailable"))
    monkeypatch.setattr(healthcheck, "check_compute", probe)
    if mode == "strict":
        with pytest.raises(RuntimeError, match="native host unavailable"):
            asyncio.run(fp._run_healthcheck(settings))
    else:
        assert asyncio.run(fp._run_healthcheck(settings)) is settings
        assert settings.compute_disabled_reason is None
        assert bool(settings.warnings) == (mode == "warn")
    assert probe.await_count == (0 if mode == "off" else 1)


def test_preset_disabled_compute_does_not_probe_or_override(tmp_path, monkeypatch):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["enable_compute"] = False
    preset_path = tmp_path / "without_compute.yaml"
    preset_path.write_text(json.dumps(raw))
    settings = replace(fp._settings(), workflow=str(preset_path), batch3=True, output_dir=tmp_path)
    probe = AsyncMock(side_effect=AssertionError("disabled Compute must not start a sandbox"))
    monkeypatch.setattr(healthcheck, "_require_codex_cli_version", probe)
    assert asyncio.run(fp._run_healthcheck(settings)) is settings
    probe.assert_not_awaited()
    report = json.loads((tmp_path / "healthcheck.json").read_text())
    assert report["ok"] and report["compute"] == "disabled" and report["paid_calls"] == 0


@pytest.mark.parametrize("failure", [ValueError("invalid configuration"), OSError("cannot write report")])
def test_non_probe_failures_do_not_degrade(tmp_path, monkeypatch, failure):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    monkeypatch.setattr(healthcheck, "check_compute", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)):
        asyncio.run(fp._run_healthcheck(settings))
    assert settings.compute_disabled_reason is None


def test_fresh_adapter_refuses_existing_checkpoints(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_firstproof_profiles import fp
    settings = replace(fp._settings(), batch3=True, output_dir=tmp_path)
    saved = tmp_path / "workflow_runs/run-p/batch3-schedule.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("unchanged checkpoint")
    monkeypatch.setattr(fp, "_settings", lambda: settings)
    monkeypatch.setattr(fp, "_prepare_output_dir", lambda *_: pytest.fail("must not modify checkpoint"))
    assert asyncio.run(fp._amain()) == 2
    assert saved.read_text() == "unchanged checkpoint"


@pytest.mark.parametrize("mode", [None, "warn", "strict"])
@pytest.mark.parametrize("healthy", [False, True])
def test_startup_preserves_fallbacks_and_propagates_policy_to_every_retry(
    tmp_path, monkeypatch, capsys, mode, healthy, offline_cleanup_probe,
):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    if mode is not None:
        monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    monkeypatch.setenv("FIRSTPROOF_WORKFLOW", "firstproof_batch3")
    monkeypatch.setenv("FIRSTPROOF_TMP_PROBLEM_DIR", str(tmp_path / "problems"))
    settings = replace(fp._settings(), output_dir=tmp_path / "output")
    if not healthy:
        offline_cleanup_probe.side_effect = cleanup.CleanupUnavailable("cleanup runtime unavailable")
    items = [{"id": f"p{i}", "latex": "Prove P."} for i in range(10)]
    monkeypatch.setattr(fp, "_settings", lambda: settings)
    monkeypatch.setattr(fp, "_load_problem_items", lambda *_: items)

    def assert_fallbacks():
        solutions = json.loads((settings.output_dir / "solutions.json").read_text())["solutions"]
        assert len(solutions) == len(items)
        for item, solution in zip(items, solutions):
            assert solution["id"] == item["id"]
            assert r"\begin{document}" in solution["latex"]
            assert (settings.output_dir / f"{item['id']}.tex").read_text() == solution["latex"]
        assert (settings.output_dir / "token_usage.jsonl").exists()

    async def bootstrap():
        assert_fallbacks()
        return False, None

    async def check(*args):
        assert_fallbacks()
        summary = json.loads((settings.output_dir / "run_summary.json").read_text())
        assert summary["in_progress"] and summary["completed_count"] == 0
        if not healthy:
            raise RuntimeError("native host fails")

    captured = []
    attempts = {}

    async def spawn(*cmd, **kwargs):
        assert healthy or mode != "strict", "strict failure must not launch any workflow"
        captured.append(cmd)
        summary = json.loads((settings.output_dir / "run_summary.json").read_text())
        assert summary["compute_disabled_reason"] == (None if healthy else "native host fails")
        assert summary["cleanup_disabled_reason"] == (None if healthy else "cleanup runtime unavailable")
        assert ("cleanup_backend=api" in cmd) == (not healthy)
        assert ("enable_compute=false" in cmd) == (not healthy)
        assert "enable_compute=true" not in cmd
        assert cmd[cmd.index("--workflow") + 1] == "firstproof_batch3"
        assert not any(arg.startswith("enable_council=") for arg in cmd)
        run_id = cmd[cmd.index("--run-id") + 1]
        attempts[run_id] = attempts.get(run_id, 0) + 1
        assert ("--restart-from" in cmd) == (attempts[run_id] > 1)
        code = 1 if attempts[run_id] == 1 else 0
        stdout = asyncio.StreamReader()
        stdout.feed_eof()
        return SimpleNamespace(stdout=stdout, returncode=code, wait=AsyncMock(return_value=code))

    probe = AsyncMock(side_effect=check)
    monkeypatch.setattr(fp, "_bootstrap_codex_auth", bootstrap)
    monkeypatch.setattr(healthcheck, "check_compute", probe)
    monkeypatch.setattr(fp.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(fp, "_batch3_can_retry", lambda problem, settings, code: code == 1)
    fatal = not healthy and mode == "strict"
    assert asyncio.run(fp._amain()) == (2 if fatal else 0)
    probe.assert_awaited_once()
    assert_fallbacks()
    summary = json.loads((settings.output_dir / "run_summary.json").read_text())
    assert not summary["in_progress"]
    assert summary["completed_count"] == len(items)
    if fatal:
        assert not captured
        assert all(row["status"] == "adapter_error" for row in summary["per_problem"])
        assert any("startup healthcheck failed" in warning for warning in summary["warnings"])
    else:
        assert len(captured) == 2 * len(items)
        assert len(attempts) == len(items)
        assert set(attempts.values()) == {2}
        assert all(row["returncode"] == 0 for row in summary["per_problem"])
        if not healthy:
            assert any("disabling Compute for every problem" in warning for warning in summary["warnings"])
            assert "disabling Compute for every problem" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["off", "warn", "strict"])
def test_cleanup_launch_gate_degrades_only_unavailable_cli(tmp_path, monkeypatch, offline_cleanup_probe, mode):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    monkeypatch.setattr(healthcheck, "check_compute", AsyncMock())
    offline_cleanup_probe.side_effect = cleanup.CleanupUnavailable("missing Claude")
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    if mode == "strict":
        with pytest.raises(cleanup.CleanupUnavailable, match="missing Claude"):
            asyncio.run(fp._run_healthcheck(settings))
    else:
        updated = asyncio.run(fp._run_healthcheck(settings))
        assert updated.cleanup_disabled_reason == "missing Claude"
        assert updated.compute_disabled_reason is None
        assert any("selecting API cleanup" in warning for warning in updated.warnings)
    report = json.loads((tmp_path / "cleanup-healthcheck.json").read_text())
    assert not report["ok"] and report["paid_calls"] == 0
    assert report["backend"] == ("claude_code" if mode == "strict" else "api")


@pytest.mark.parametrize("failure", [ValueError("invalid price"), OSError("cannot read configuration")])
def test_cleanup_configuration_failure_is_not_silently_downgraded(
    tmp_path, monkeypatch, offline_cleanup_probe, failure,
):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setattr(healthcheck, "check_compute", AsyncMock())
    offline_cleanup_probe.side_effect = failure
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True, output_dir=tmp_path)
    with pytest.raises(type(failure), match=str(failure)):
        asyncio.run(fp._run_healthcheck(settings))
    assert not json.loads((tmp_path / "cleanup-healthcheck.json").read_text())["ok"]


def test_explicit_api_cleanup_skips_cli_preflight(tmp_path, monkeypatch, offline_cleanup_probe):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["cleanup_backend"] = "api"
    preset_path = tmp_path / "api.yaml"
    preset_path.write_text(json.dumps(raw))
    settings = replace(fp._settings(), workflow=str(preset_path), batch3=True, output_dir=tmp_path)
    monkeypatch.setattr(healthcheck, "check_compute", AsyncMock())
    assert asyncio.run(fp._run_healthcheck(settings)) is settings
    offline_cleanup_probe.assert_not_awaited()


def test_launch_cleanup_configuration_matches_agent_resolution(offline_cleanup_probe, tmp_path):
    from proofstack.context import RunContext
    configs = {
        "*": {"cleanup": {"max_invocation_usd": 7, "max_episode_usd": 12}},
        "proofstack.agents.cleanup_session.CleanupSession": {"cleanup": {"max_invocation_usd": 6}},
        "CleanupSession": {"cleanup": {"codex_budget_fraction": 0, "max_invocation_usd": 5}},
    }
    asyncio.run(cleanup.check_cleanup_launch({"cleanup_backend": "claude_code"}, configs, tmp_path))
    agent = cleanup.CleanupSession(RunContext.create(root_workdir=tmp_path, component_configs=configs))
    expected = cleanup.CleanupSettings.model_validate(agent.component_config["cleanup"])
    offline_cleanup_probe.assert_awaited_once_with(expected)
    assert expected.max_invocation_usd == 5 and expected.max_episode_usd == 12


@pytest.mark.parametrize("mode", ["off", "warn", "strict"])
@pytest.mark.parametrize("role", ["editors", "codex"])
def test_legacy_cleanup_registry_blocks_launch_before_runtime_probe(
    tmp_path, monkeypatch, offline_cleanup_probe, role, mode,
):
    from test_firstproof_profiles import fp, _clear_firstproof_env
    _clear_firstproof_env(monkeypatch)
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", mode)
    raw = load_preset("firstproof_batch3").raw
    raw["inputs"]["enable_compute"] = False
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(json.dumps(raw))
    registry = tmp_path / "workflow_runs/.proofcouncil-cleanup-memory" / role
    registry.mkdir(parents=True)
    control = registry / "control.json"
    before = json.dumps({"max_workers": 10, "worker_bytes": GiB, "reserve_bytes": 4 * GiB})
    control.write_text(before)
    settings = replace(fp._settings(), workflow=str(preset_path), batch3=True, output_dir=tmp_path)
    with pytest.raises(MemoryRegistryError, match=role):
        asyncio.run(fp._run_healthcheck(settings))
    offline_cleanup_probe.assert_not_awaited()
    assert control.read_text() == before
    assert list(registry.iterdir()) == [control]
    report = json.loads((tmp_path / "cleanup-healthcheck.json").read_text())
    assert report["ok"] is False and report["paid_calls"] == 0
    assert report["backend"] == "claude_code"
    assert report["error_type"] == "MemoryRegistryError"
    assert str(control) in report["reason"]


@pytest.mark.parametrize("codex_fraction", [0, 0.3])
def test_cleanup_registry_gate_uses_exact_role_policies(
    tmp_path, monkeypatch, offline_cleanup_probe, codex_fraction,
):
    from proofstack.context import RunContext
    configs = {
        "*": {"cleanup": {"memory_gb": 2, "memory_reserve_gb": 6}},
        "proofstack.agents.cleanup_session.CleanupSession": {"cleanup": {"max_parallel_editors": 3}},
        "CleanupSession": {"cleanup": {"max_parallel_codex_reviews": 2, "codex_budget_fraction": codex_fraction}},
    }
    checked = []
    monkeypatch.setattr(cleanup, "check_memory_registry", checked.append)
    agent = cleanup.CleanupSession(RunContext.create(
        root_workdir=tmp_path / "workflow_runs", component_configs=configs))
    settings = cleanup.CleanupSettings.model_validate(agent.component_config["cleanup"])
    probes = asyncio.run(cleanup.check_cleanup_launch(
        {"cleanup_backend": "claude_code"}, configs, tmp_path / "workflow_runs"))
    providers = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"] if codex_fraction else ["ANTHROPIC_API_KEY"]
    assert checked == [agent._sandbox(settings, 60, provider)["memory_policy"] for provider in providers]
    assert [p.max_workers for p in checked] == ([3, 2] if codex_fraction else [3])
    assert all(p.worker_bytes == 2 * GiB and p.reserve_bytes == 6 * GiB for p in checked)
    assert [p["registry"] for p in probes] == [str(p.registry) for p in checked]
    offline_cleanup_probe.assert_awaited_once_with(settings)


def test_api_cleanup_does_not_inspect_native_registries(tmp_path, monkeypatch, offline_cleanup_probe):
    monkeypatch.setattr(cleanup, "check_memory_registry", lambda *_: pytest.fail("native cleanup disabled"))
    assert asyncio.run(cleanup.check_cleanup_launch({"cleanup_backend": "api"}, {}, tmp_path)) == []
    offline_cleanup_probe.assert_not_awaited()
