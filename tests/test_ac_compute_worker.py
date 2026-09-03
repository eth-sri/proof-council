from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.compute import (  # noqa: E402
    COMPUTE_FILESYSTEM_MIN_FREE_BYTES,
    COMPUTE_FILESYSTEM_MIN_FREE_INODES,
    COMPUTE_FILESYSTEM_RESERVATION_BYTES,
    COMPUTE_HANDOFF_MANIFEST,
    COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES,
    COMPUTE_HANDOFF_MAX_FILES,
    COMPUTE_HANDOFF_MAX_MEMBER_BYTES,
    COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES,
    COMPUTE_WORKSPACE_HARD_LIMIT_BYTES,
    COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES,
    COMPUTE_WORKSPACE_SOFT_LIMIT_BYTES,
    COMPUTE_WORKSPACE_SOFT_LIMIT_ENTRIES,
    _CODEX_LAST_MESSAGE_REL,
    _ZIP_EXCLUDE_TOP,
    _build_codex_cmd,
    _require_codex_cli_version,
    _zip_workspace,
    Compute,
    DEFAULT_COST_CONFIG,
    DEFAULT_HARD_TIMEOUT_S,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SOFT_TIMEOUT_S,
    inspect_compute_handoff,
    render_compute_reply_for_author,
)
from proofstack.context import RunContext  # noqa: E402
from proofstack.cli_usage import parse_claude_json, parse_codex_jsonl  # noqa: E402
from proofstack.kinds.cli import (  # noqa: E402
    CLIAgent,
    CLIDoneRecord,
    _mark_sensitive_workspace_untrusted,
    _quarantine_workspace,
    _write_workspace_active_guard,
    _write_sensitive_quarantine_marker,
    measure_workspace_usage,
)
from proofstack.sandbox.base import (  # noqa: E402
    SandboxSpawnError,
    SandboxSpec,
    WorkerStopState,
)
from proofstack.sandbox.docker import (  # noqa: E402
    DockerSandbox,
    DockerSandboxError,
    _DockerLaunchReceipt,
    _DockerStreamingProcess,
    _container_name_for_root,
    _docker_kill,
    _terminate_docker_run,
)
from proofstack.sandbox.subprocess import (  # noqa: E402
    SubprocessSandbox,
    _BoundedTextBuffer,
    _JsonUsageCapture,
    _ProcessMarker,
    _StreamingProcess,
    _find_marked_processes,
    _terminate_marked_processes,
)


class FakeSandbox(SimpleNamespace):
    async def run_command(self, cmd, **kwargs):
        calls = getattr(self, "calls", None)
        if calls is None:
            calls = []
            self.calls = calls
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd == ["codex", "login", "--with-api-key"]:
            returncode = getattr(self, "login_returncode", 0)
            if returncode == 0:
                codex_home = Path(kwargs["env_extra"]["CODEX_HOME"])
                if codex_home == Path("/codex-home"):
                    raise AssertionError(
                        "API-key FakeSandbox tests must use the subprocess backend"
                    )
                codex_home.mkdir(parents=True, exist_ok=True)
                (codex_home / "auth.json").write_text(
                    json.dumps(
                        {
                            "auth_mode": "apikey",
                            "OPENAI_API_KEY": kwargs.get("input_data"),
                        }
                    ),
                    encoding="utf-8",
                )
            return SimpleNamespace(
                returncode=returncode,
                stdout="Successfully logged in" if returncode == 0 else "",
                stderr=getattr(self, "login_stderr", ""),
            )
        if cmd == ["codex", "login", "status"]:
            return SimpleNamespace(
                returncode=getattr(self, "status_returncode", 0),
                stdout=getattr(self, "status_stdout", "Logged in"),
                stderr=getattr(self, "status_stderr", ""),
            )
        return SimpleNamespace(
            returncode=getattr(self, "version_returncode", 0),
            stdout=getattr(self, "version_stdout", "codex-cli 0.144.0"),
            stderr="",
        )

    async def write_file(self, relpath: str, content: str) -> Path:
        path = Path(self.root) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


def _fake_subscription_home(root: Path) -> Path:
    home = root / "fake-home"
    auth_path = home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": "test-only-token"},
            }
        ),
        encoding="utf-8",
    )
    return home


def test_compute_codex_command_uses_current_exec_flags() -> None:
    cmd = _build_codex_cmd(
        model=DEFAULT_MODEL,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        sandbox=SandboxSpec(backend="subprocess"),
        codex_sandbox="docker-bypass",
    )

    assert cmd[:2] == ["codex", "exec"]
    assert cmd[cmd.index("-m") + 1] == "gpt-5.6-sol"
    assert cmd[cmd.index("-c") + 1] == 'model_reasoning_effort="max"'
    assert "--ignore-user-config" not in cmd
    assert "--ephemeral" not in cmd
    assert "--output-last-message" in cmd
    assert cmd[cmd.index("--output-last-message") + 1] == _CODEX_LAST_MESSAGE_REL
    # First Proof already runs inside an isolated submitter container.
    # Bypass Codex's nested bubblewrap sandbox so exec/apply_patch/finish
    # work in unprivileged Docker/Podman.
    assert "--sandbox" not in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert cmd[-1] == "-"


def test_compute_inputs_default_to_sol_max_with_matching_cost_config() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="compute",
            compute_workspace=Path(temp_dir),
        )

    assert inp.model == "gpt-5.6-sol"
    assert inp.reasoning_effort == "max"
    assert inp.cost_config == "models/openai/gpt-56-sol-pro"
    assert inp.model == DEFAULT_MODEL
    assert inp.reasoning_effort == DEFAULT_REASONING_EFFORT
    assert inp.cost_config == DEFAULT_COST_CONFIG
    assert inp.soft_timeout_s == DEFAULT_SOFT_TIMEOUT_S == 7200
    assert inp.hard_timeout_s == DEFAULT_HARD_TIMEOUT_S == 9000
    assert inp.workspace_soft_limit_bytes == COMPUTE_WORKSPACE_SOFT_LIMIT_BYTES
    assert inp.workspace_hard_limit_bytes == COMPUTE_WORKSPACE_HARD_LIMIT_BYTES
    assert inp.workspace_soft_limit_entries == COMPUTE_WORKSPACE_SOFT_LIMIT_ENTRIES
    assert inp.workspace_hard_limit_entries == COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES
    assert inp.filesystem_min_free_bytes == COMPUTE_FILESYSTEM_MIN_FREE_BYTES
    assert inp.filesystem_min_free_inodes is None
    assert inp.filesystem_reservation_bytes == COMPUTE_FILESYSTEM_RESERVATION_BYTES
    assert inp.filesystem_reservation_dir is None
    assert inp.handoff_max_files == COMPUTE_HANDOFF_MAX_FILES
    assert inp.handoff_max_compressed_bytes == COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES
    assert inp.handoff_max_uncompressed_bytes == COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES
    assert inp.handoff_max_member_bytes == COMPUTE_HANDOFF_MAX_MEMBER_BYTES


def test_compute_rejects_soft_timeout_at_or_after_hard_timeout() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        try:
            Compute.Inputs(
                problem="P",
                problem_id="p",
                round=1,
                instructions="compute",
                compute_workspace=temp / "compute",
                soft_timeout_s=100,
                hard_timeout_s=100,
            )
        except ValueError as e:
            assert "soft_timeout_s must be less than hard_timeout_s" in str(e)
        else:
            raise AssertionError("invalid timeout ordering was accepted")


def test_compute_rejects_invalid_workspace_limit_ordering() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            Compute.Inputs(
                problem="P",
                problem_id="p",
                round=1,
                instructions="compute",
                compute_workspace=Path(temp_dir),
                workspace_soft_limit_bytes=100,
                workspace_hard_limit_bytes=100,
            )
        except ValueError as e:
            assert "workspace_soft_limit_bytes must be less" in str(e)
        else:
            raise AssertionError("invalid workspace limit ordering was accepted")


def test_compute_rejects_invalid_workspace_entry_limit_ordering() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            Compute.Inputs(
                problem="P",
                problem_id="p",
                round=1,
                instructions="compute",
                compute_workspace=Path(temp_dir),
                workspace_soft_limit_entries=100,
                workspace_hard_limit_entries=100,
            )
        except ValueError as e:
            assert "workspace_soft_limit_entries must be less" in str(e)
        else:
            raise AssertionError("invalid workspace entry ordering was accepted")


def test_compute_workspace_entry_cap_can_be_raised() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="compute",
            compute_workspace=Path(temp_dir),
            workspace_soft_limit_entries=150_000,
            workspace_hard_limit_entries=200_000,
        )

    assert inp.workspace_soft_limit_entries == 150_000
    assert inp.workspace_hard_limit_entries == 200_000


def test_compute_rejects_handoff_member_limit_without_manifest_room() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            Compute.Inputs(
                problem="P",
                problem_id="p",
                round=1,
                instructions="compute",
                compute_workspace=Path(temp_dir),
                handoff_max_uncompressed_bytes=2 * 1024 * 1024,
                handoff_max_member_bytes=2 * 1024 * 1024,
            )
        except ValueError as e:
            assert "must leave room for the handoff manifest" in str(e)
        else:
            raise AssertionError("invalid handoff byte limits were accepted")


def test_compute_soft_timeout_is_capped_below_remaining_workflow_budget() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(run_id="test", root_workdir=Path(temp_dir), flat=True)
        agent = Compute(ctx)

        assert agent._effective_soft_timeout_s(9000) == 7200
        assert agent._effective_soft_timeout_s(3600) == 1800


def test_compute_rejects_old_codex_cli_before_starting_worker() -> None:
    sandbox = FakeSandbox(
        root=Path("."),
        version_stdout="codex-cli 0.143.0",
    )

    try:
        asyncio.run(_require_codex_cli_version(sandbox))
    except RuntimeError as e:
        assert "0.144.0 or newer" in str(e)
        assert "0.143.0" in str(e)
    else:
        raise AssertionError("old Codex CLI version was accepted")


def test_dockerfile_pins_and_smokes_codex_cli() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    deploy_text = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    sandbox_text = (ROOT / "deploy" / "sandbox" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    pwc_text = (ROOT / "deploy" / "sandbox" / "Dockerfile.pwc").read_text(
        encoding="utf-8"
    )

    assert "@openai/codex@${OPENAI_CODEX_VERSION}" in text
    assert "ARG OPENAI_CODEX_VERSION=0.144.0" in text
    assert 'codex --version | grep -q -- "${OPENAI_CODEX_VERSION}"' in text
    assert "codex exec --help | grep -q -- '--output-last-message'" in text
    assert "gmpy2 python-flint z3-solver cvxpy" in text
    assert "git file column time" in text
    assert "> /usr/local/bin/finish" in text
    assert "FINISH_DONE_PATH" in text
    for runtime_text in (deploy_text, sandbox_text):
        assert "ARG OPENAI_CODEX_VERSION=0.144.0" in runtime_text
        assert "@openai/codex@${OPENAI_CODEX_VERSION}" in runtime_text
        assert 'codex --version | grep -q -- "${OPENAI_CODEX_VERSION}"' in runtime_text
    assert "> /usr/local/bin/finish" in pwc_text
    assert "FINISH_DONE_PATH" in pwc_text


def test_compute_collect_falls_back_to_codex_last_message() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        (root / ".pwc" / "runtime").mkdir(parents=True)
        (root / _CODEX_LAST_MESSAGE_REL).write_text(
            "final notes from codex", encoding="utf-8"
        )
        (root / "scratch").mkdir()
        (root / "scratch" / "failed.db").write_bytes(b"discard")

        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="do the computation",
            compute_workspace=root,
        )
        out = asyncio.run(
            agent.collect(
                SimpleNamespace(root=root),
                inp,
                CLIDoneRecord(status="done", summary="(no done.json written)"),
            )
        )

        assert out.response_md == "final notes from codex"
        assert out.status == "done"
        assert out.zip_path is not None
        assert Path(out.zip_path).exists()
        assert list((root / "scratch").iterdir()) == []
        with zipfile.ZipFile(out.zip_path) as zf:
            assert not any(name.startswith("scratch/") for name in zf.namelist())


def test_compute_collect_redacts_response_and_excludes_it_from_handoff() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        secret = "subscription-token-that-must-not-leak"
        agent._handoff_secrets = (secret,)
        root = temp / "compute"
        (root / "responses").mkdir(parents=True)
        (root / "responses" / "response_round_1.md").write_text(
            f"result followed by {secret}", encoding="utf-8"
        )

        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="do the computation",
            compute_workspace=root,
        )
        out = asyncio.run(
            agent.collect(
                SimpleNamespace(root=root),
                inp,
                CLIDoneRecord(status="done", summary=f"summary {secret}"),
            )
        )

        assert secret not in out.response_md
        assert secret not in out.summary
        assert "[redacted-codex-credential]" in out.response_md
        assert out.zip_path is not None
        assert secret.encode("utf-8") not in Path(out.zip_path).read_bytes()
        assert out.handoff_stats["credential_files"] == 1


def test_compute_always_uses_scrubbed_codex_home() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        root.mkdir()
        (root / "scratch").mkdir()
        (root / "scratch" / "stale-cancelled-run.db").write_bytes(b"stale")
        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="do the computation",
            compute_workspace=root,
        )
        sandbox = FakeSandbox(root=root)
        agent._host_codex_auth_text = json.dumps(
            {"auth_mode": "chatgpt", "tokens": {"access_token": "test-token"}}
        )

        asyncio.run(agent.setup(sandbox, inp))
        env = agent.extra_env(sandbox, inp)
        codex_home = agent._codex_home_host
        assert codex_home is not None

        assert env == {"CODEX_HOME": "/codex-home"}
        assert codex_home.is_dir()
        assert root.resolve() not in codex_home.parents
        assert ctx.root_workdir.resolve() not in codex_home.parents
        assert not (root / ".codex-home").exists()
        assert (root / "code" / "__init__.py").exists()
        assert (root / "data").is_dir()
        assert (root / "papers").is_dir()
        assert (root / "notes").is_dir()
        assert (root / "scratch").is_dir()
        assert list((root / "scratch").iterdir()) == []
        assert (root / "archive").is_dir()
        assert (root / "compute_utils.py").exists()
        assert (root / "sitecustomize.py").exists()

        (codex_home / "session.json").write_text("{}", encoding="utf-8")
        (root / "scratch" / "partial-run.db").write_bytes(b"partial")
        asyncio.run(agent.teardown(sandbox, inp))
        assert not codex_home.exists()
        assert list((root / "scratch").iterdir()) == []


def test_compute_reports_unverified_transient_auth_cleanup() -> None:
    from proofstack.kinds.cli import CLIAgent
    from proofstack.transient_auth import remove_codex_auth_parent

    async def fake_super_run(self, inp):
        return Compute.Outputs()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        home = _fake_subscription_home(temp)
        ctx = RunContext.create(
            run_id="test", root_workdir=temp / "run", flat=True
        )
        agent = Compute(ctx)
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="compute",
            compute_workspace=temp / "compute",
            sandbox_backend="subprocess",
        )
        cleanup_calls: list[tuple[Path, Path]] = []

        def fail_cleanup(parent: Path, run_path: Path) -> None:
            cleanup_calls.append((parent, Path(run_path)))
            raise RuntimeError("transient Codex authentication cleanup failed")

        with mock.patch.object(Path, "home", return_value=home), mock.patch.object(
            CLIAgent, "run", fake_super_run
        ), mock.patch(
            "proofstack.agents.ac.compute.require_codex_auth_parent_removed",
            side_effect=fail_cleanup,
        ), pytest.raises(
            RuntimeError,
            match="transient Codex authentication cleanup failed",
        ):
            asyncio.run(agent.run(inp))

        assert len(cleanup_calls) == 1
        parent, run_path = cleanup_calls[0]
        assert ctx.root_workdir.resolve() not in parent.parents
        assert agent._codex_auth_parent is None
        assert remove_codex_auth_parent(parent, run_path)


def test_compute_never_forwards_provider_keys_to_codex_exec() -> None:
    from unittest import mock

    from proofstack.kinds.cli import CLIAgent

    async def fake_super_run(self, inp):
        return Compute.Outputs()

    def spec_for(home: Path) -> SandboxSpec:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test", root_workdir=Path(temp_dir) / "run", flat=True
            )
            agent = Compute(ctx)
            inp = Compute.Inputs(
                problem="P",
                problem_id="p",
                round=1,
                instructions="compute",
                compute_workspace=Path(temp_dir) / "ws",
                sandbox_backend="subprocess",
            )
            with mock.patch.object(Path, "home", return_value=home), mock.patch.object(
                CLIAgent, "run", fake_super_run
            ), mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-test-paid-key"}, clear=False
            ):
                asyncio.run(agent.run(inp))
            return agent.SANDBOX

    with tempfile.TemporaryDirectory() as home_dir:
        home = Path(home_dir)
        assert spec_for(home).provider_keys == ()
        (home / ".codex").mkdir()
        (home / ".codex" / "auth.json").write_text(
            json.dumps(
                {"auth_mode": "chatgpt", "tokens": {"access_token": "token"}}
            ),
            encoding="utf-8",
        )
        assert spec_for(home).provider_keys == ()


def test_compute_rejects_missing_or_unrecognized_auth_before_cli_spawn() -> None:
    from unittest import mock

    from proofstack.kinds.cli import CLIAgent

    async def fake_super_run(self, inp):
        raise AssertionError("CLI must not start without valid authentication")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        home = temp / "home"
        home.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="compute",
            compute_workspace=temp / "ws",
            sandbox_backend="subprocess",
        )

        with mock.patch.object(Path, "home", return_value=home), mock.patch.object(
            CLIAgent, "run", fake_super_run
        ), mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            try:
                asyncio.run(agent.run(inp))
            except RuntimeError as e:
                assert "OPENAI_API_KEY is not set" in str(e)
            else:
                raise AssertionError("missing Codex authentication was accepted")

        auth_dir = home / ".codex"
        auth_dir.mkdir()
        (auth_dir / "auth.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(Path, "home", return_value=home), mock.patch.object(
            CLIAgent, "run", fake_super_run
        ), mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test-paid-key"}, clear=False
        ):
            try:
                asyncio.run(agent.run(inp))
            except RuntimeError as e:
                assert "unrecognized schema" in str(e)
            else:
                raise AssertionError("unrecognized Codex authentication was accepted")


def test_compute_bootstraps_paid_key_into_transient_codex_home() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        root.mkdir()
        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="compute",
            compute_workspace=root,
            sandbox_backend="subprocess",
        )
        api_key = "sk-test-paid-key"
        agent._paid_codex_api_key = api_key
        sandbox = FakeSandbox(root=root)

        asyncio.run(agent.setup(sandbox, inp))
        codex_home = agent._codex_home_host
        assert codex_home is not None
        auth_path = codex_home / "auth.json"
        assert json.loads(auth_path.read_text(encoding="utf-8")) == {
            "auth_mode": "apikey",
            "OPENAI_API_KEY": api_key,
        }
        assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
        assert agent._copied_codex_auth is True
        assert agent._subscription_codex_auth is False
        assert agent._paid_codex_api_key is None

        login_call = next(
            call
            for call in sandbox.calls
            if call["cmd"] == ["codex", "login", "--with-api-key"]
        )
        assert login_call["input_data"] == api_key
        assert login_call["env_extra"] == {"CODEX_HOME": str(codex_home)}
        assert all(api_key not in arg for arg in login_call["cmd"])
        assert any(
            call["cmd"] == ["codex", "login", "status"]
            for call in sandbox.calls
        )

        asyncio.run(agent.teardown(sandbox, inp))
        assert not codex_home.exists()


def test_compute_redacts_api_key_when_codex_login_fails() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        root.mkdir()
        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="compute",
            compute_workspace=root,
            sandbox_backend="subprocess",
        )
        api_key = "sk-test-secret-never-log"
        agent._paid_codex_api_key = api_key
        sandbox = FakeSandbox(
            root=root,
            login_returncode=1,
            login_stderr=f"rejected credential {api_key}",
        )

        try:
            asyncio.run(agent.setup(sandbox, inp))
        except RuntimeError as e:
            assert api_key not in str(e)
            assert "[redacted]" in str(e)
        else:
            raise AssertionError("failed Codex login was accepted")
        finally:
            asyncio.run(agent.teardown(sandbox, inp))


def test_subprocess_sandbox_pipes_input_without_putting_it_in_command() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox = SubprocessSandbox(
            SandboxSpec(backend="subprocess", provider_keys=()),
            root=Path(temp_dir) / "sandbox",
        )
        secret = "stdin-only-value"
        result = asyncio.run(
            sandbox.run_command(
                [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                input_data=secret,
            )
        )
        assert result.returncode == 0
        assert result.stdout.strip() == secret
        assert secret not in result.cmd


def test_compute_recognizes_token_schema_auth_as_subscription() -> None:
    import json as _json

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        root.mkdir()
        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="compute",
            compute_workspace=root,
        )
        # What run() captures from the CURRENT codex login schema.
        agent._host_codex_auth_text = _json.dumps(
            {"OPENAI_API_KEY": None, "tokens": {"access_token": "t"}}
        )
        asyncio.run(agent.setup(FakeSandbox(root=root), inp))
        assert agent._copied_codex_auth is True
        assert agent._subscription_codex_auth is True
        asyncio.run(agent.teardown(FakeSandbox(root=root), inp))
        assert agent._subscription_codex_auth is False


def test_compute_usage_subscription_auth_charges_no_usd() -> None:
    import json as _json

    stdout = _json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test", root_workdir=Path(temp_dir) / "run", flat=True
        )
        agent = Compute(ctx)
        agent._last_cost_config = DEFAULT_COST_CONFIG
        calls: list[dict] = []

        async def emit(kind, payload):
            calls.append({"kind": kind, **payload})

        agent.events = SimpleNamespace(emit=emit)
        agent.tracker = SimpleNamespace(
            usd=0.0,
            tokens=0,
            add_usd=lambda a: setattr(agent.tracker, "usd", agent.tracker.usd + a),
            add_tokens=lambda n: setattr(
                agent.tracker, "tokens", agent.tracker.tokens + n
            ),
        )

        agent._subscription_codex_auth = True
        asyncio.run(agent.record_cli_usage(stdout, "", CLIDoneRecord(status="done")))
        assert agent.tracker.usd == 0.0
        assert agent.tracker.tokens == 1100
        assert calls[-1]["cost_usd"] == 0.0
        assert calls[-1]["subscription"] is True
        assert calls[-1]["api_equivalent_usd"] > 0

        agent._subscription_codex_auth = False
        asyncio.run(agent.record_cli_usage(stdout, "", CLIDoneRecord(status="done")))
        assert agent.tracker.usd > 0.0
        assert calls[-1]["cost_usd"] == calls[-1]["api_equivalent_usd"] > 0
        assert calls[-1]["subscription"] is False


def test_compute_paid_usage_fails_closed_without_a_completed_turn() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test", root_workdir=Path(temp_dir) / "run", flat=True
        )
        agent = Compute(ctx)
        agent._subscription_codex_auth = False

        with pytest.raises(
            RuntimeError,
            match="paid Codex call produced no parseable usage record",
        ):
            asyncio.run(
                agent.record_cli_usage(
                    '{"type":"thread.started"}\n',
                    "",
                    CLIDoneRecord(status="timeout"),
                )
            )

        agent._subscription_codex_auth = True
        asyncio.run(
            agent.record_cli_usage(
                '{"type":"thread.started"}\n',
                "",
                CLIDoneRecord(status="timeout"),
            )
        )


def test_compute_rejects_invalid_paid_cost_config_before_spawn() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        home = temp / "home"
        home.mkdir()
        ctx = RunContext.create(
            run_id="test", root_workdir=temp / "run", flat=True
        )
        agent = Compute(ctx)

        with mock.patch(
            "proofstack.agents.ac.compute.Path.home", return_value=home
        ), mock.patch.dict(os.environ, {"OPENAI_API_KEY": "paid-test-key"}):
            try:
                asyncio.run(
                    agent(
                        problem="P",
                        problem_id="prob-001",
                        round=1,
                        instructions="compute",
                        compute_workspace=temp / "compute",
                        cost_config="models/openai/does-not-exist",
                        sandbox_backend="subprocess",
                    )
                )
            except RuntimeError as e:
                assert "Paid Codex cost configuration must be valid" in str(e)
            else:
                raise AssertionError("invalid paid cost configuration was accepted")

        events_path = temp / "run" / "events.jsonl"
        events = events_path.read_text(encoding="utf-8")
        assert "cli.spawn" not in events
        assert agent._paid_codex_api_key is None


def test_compute_utils_serializes_numpy_and_complex_values() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        root = temp / "compute"
        root.mkdir()
        inp = Compute.Inputs(
            problem="P",
            problem_id="prob-001",
            round=1,
            instructions="do the computation",
            compute_workspace=root,
        )
        sandbox = FakeSandbox(root=root)
        agent._host_codex_auth_text = json.dumps(
            {"auth_mode": "chatgpt", "tokens": {"access_token": "test-token"}}
        )

        asyncio.run(agent.setup(sandbox, inp))
        spec = importlib.util.spec_from_file_location(
            "compute_utils_test", root / "compute_utils.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        import numpy as np

        text = module.safe_json_dumps(
            {"arr": np.array([1, 2]), "scalar": np.int64(3), "z": 1 + 2j},
            sort_keys=True,
        )

        assert '"arr": [1, 2]' in text
        assert '"scalar": 3' in text
        assert '"z": {"imag": 2.0, "real": 1.0}' in text


def test_compute_workspace_zip_excludes_codex_runtime_dirs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / ".codex" / "sessions").mkdir(parents=True)
        (root / ".codex" / "sessions" / "secret.json").write_text(
            "secret", encoding="utf-8"
        )
        (root / ".codex-home").mkdir(parents=True)
        (root / ".codex-home" / "auth.json").write_text("secret", encoding="utf-8")
        (root / ".bash_profile").write_text("export PATH=/tmp/bin:$PATH", encoding="utf-8")
        (root / "notes").mkdir()
        (root / "notes" / "result.txt").write_text("keep", encoding="utf-8")

        out_zip = temp / "workspace.zip"
        _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert "notes/result.txt" in names
        assert not any(name.startswith(".codex/") for name in names)
        assert not any(name.startswith(".codex-home/") for name in names)
        assert ".bash_profile" not in names


def test_compute_handoff_excludes_package_cache_directories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        cache_files = (
            root / ".cache" / "uv" / "wheels" / "numpy.whl",
            root / ".local" / "lib" / "python3.14" / "scipy.py",
            root / ".sage" / "cache" / "gap.sobj",
            root / ".npm" / "_cacache" / "index",
            root / "code" / ".venv" / "lib" / "sympy.py",
            root / "code" / "__pycache__" / "verify.pyc",
        )
        for path in cache_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("cache", encoding="utf-8")
        (root / "code" / "verify.py").write_text("keep", encoding="utf-8")
        (root / "data").mkdir()
        (root / "data" / "table.csv").write_text("keep", encoding="utf-8")
        (root / "notes").mkdir()
        (root / "notes" / ".cache").write_text("keep", encoding="utf-8")
        (root / "notes" / ".sage").write_text("keep", encoding="utf-8")

        out_zip = temp / "workspace.zip"
        _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert names == {
            COMPUTE_HANDOFF_MANIFEST,
            "code/verify.py",
            "data/table.csv",
            "notes/.cache",
            "notes/.sage",
        }


def test_compute_handoff_keeps_regular_files_named_like_cache_dirs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "notes").mkdir(parents=True)
        cache_names = (".cache", ".venv", ".npm", ".sage", "__pycache__", ".local")
        for name in cache_names:
            (root / name).write_text("keep", encoding="utf-8")
            (root / "notes" / name).write_text("keep", encoding="utf-8")

        out_zip = temp / "workspace.zip"
        _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        expected = {COMPUTE_HANDOFF_MANIFEST}
        expected.update(cache_names)
        expected.update(f"notes/{name}" for name in cache_names)
        assert names == expected


def test_compute_handoff_symlink_cannot_smuggle_cache_content() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "code" / ".venv").mkdir(parents=True)
        (root / "code" / ".venv" / "big.bin").write_text("cache", encoding="utf-8")
        (root / ".codex-home").mkdir(parents=True)
        (root / ".codex-home" / "auth.json").write_text("secret", encoding="utf-8")
        (root / "code" / "result.txt").write_text("keep", encoding="utf-8")
        (root / "notes").mkdir()
        (root / "notes" / "leak.bin").symlink_to(root / "code" / ".venv" / "big.bin")
        (root / "notes" / ".cache").symlink_to(root / ".codex-home" / "auth.json")
        (root / "notes" / "ok.txt").symlink_to(root / "code" / "result.txt")

        out_zip = temp / "workspace.zip"
        _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert names == {
            COMPUTE_HANDOFF_MANIFEST,
            "code/result.txt",
            "notes/ok.txt",
        }


def test_compute_handoff_checks_credential_sensitive_symlink_target_name() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "code").mkdir(parents=True)
        (root / "notes").mkdir()
        (root / "code" / "auth.json").write_text(
            "credential-shaped but not a currently known token",
            encoding="utf-8",
        )
        (root / "notes" / "innocent.txt").symlink_to(root / "code" / "auth.json")

        out_zip = temp / "workspace.zip"
        stats = _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            assert set(zf.namelist()) == {COMPUTE_HANDOFF_MANIFEST}
        assert stats["credential_files"] == 2


def test_compute_handoff_omits_copied_credentials_without_naming_them() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "notes").mkdir(parents=True)
        secret = "access-token-that-must-never-enter-a-handoff"
        (root / "notes" / "result.txt").write_text("keep", encoding="utf-8")
        (root / "notes" / "copied-token.txt").write_bytes(
            b"x" * (1024 * 1024 - 4) + secret.encode("utf-8")
        )
        (root / "notes" / f"{secret}.zip").write_text("hidden", encoding="utf-8")
        (root / "notes" / "auth.json").write_text("credentials", encoding="utf-8")

        out_zip = temp / "workspace.zip"
        stats = _zip_workspace(
            root,
            out_zip,
            exclude_top=_ZIP_EXCLUDE_TOP,
            secret_values=(secret,),
        )

        archive_bytes = out_zip.read_bytes()
        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())
            manifest = zf.read(COMPUTE_HANDOFF_MANIFEST)

        assert "notes/result.txt" in names
        assert "notes/copied-token.txt" not in names
        assert "notes/auth.json" not in names
        assert not any(secret in name for name in names)
        assert secret.encode("utf-8") not in archive_bytes
        assert secret.encode("utf-8") not in manifest
        assert stats["credential_files"] == 3
        assert secret not in repr(stats)


def test_compute_workspace_usage_redacts_secret_bearing_paths() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        secret = "credential-value-in-filename"
        agent._handoff_secrets = (secret,)
        workspace = temp / "compute"
        workspace.mkdir()
        (workspace / f"large-{secret}.txt").write_text("content", encoding="utf-8")

        usage = agent._measure_workspace(workspace)

        assert secret not in repr(usage)
        assert "[redacted-codex-credential]" in usage["largest_files"][0]["path"]


def test_compute_workspace_handoff_is_bounded_and_curated() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "responses").mkdir(parents=True)
        (root / "responses" / "response_round_2.md").write_text(
            "important result", encoding="utf-8"
        )
        (root / "notes").mkdir()
        (root / "notes" / "strategy.md").write_text("keep", encoding="utf-8")
        for dirname in ("code", "data", "papers"):
            (root / dirname).mkdir()
        for idx in range(1_005):
            (root / "code" / f"generated_{idx:04d}.py").write_text(
                f"VALUE = {idx}\n", encoding="utf-8"
            )
        (root / "data" / "result.csv").write_text("n,value\n1,2\n", encoding="utf-8")
        (root / "papers" / "source.tex").write_text("paper", encoding="utf-8")
        transient = root / "code" / "gaptempdir123"
        transient.mkdir()
        (transient / "thousands.tmp").write_text("discard", encoding="utf-8")
        pycache = root / "code" / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"discard")

        out_zip = temp / "workspace.zip"
        stats = _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())
            manifest = zf.read(COMPUTE_HANDOFF_MANIFEST).decode("utf-8")

        assert len(names) <= COMPUTE_HANDOFF_MAX_FILES
        assert stats["archive_entries"] == len(names)
        assert stats["omitted_files"] > 0
        assert "responses/response_round_2.md" in names
        assert "notes/strategy.md" in names
        assert "data/result.csv" in names
        assert "papers/source.tex" in names
        assert not any("gaptempdir" in name for name in names)
        assert not any("__pycache__" in name for name in names)
        normalized_manifest = " ".join(manifest.split())
        assert "Omitted by handoff cap:" in manifest
        assert "do not extract the whole archive" in normalized_manifest


def test_compute_handoff_enforces_member_and_total_byte_limits() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "notes").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "notes" / "summary.md").write_text("result", encoding="utf-8")
        (root / "data" / "oversized.bin").write_bytes(b"x" * 1_100_000)
        (root / "data" / "first.bin").write_bytes(b"a" * 700_000)
        (root / "data" / "second.bin").write_bytes(b"b" * 700_000)

        out_zip = temp / "workspace.zip"
        stats = _zip_workspace(
            root,
            out_zip,
            exclude_top=_ZIP_EXCLUDE_TOP,
            max_files=20,
            max_compressed_bytes=3_000_000,
            max_uncompressed_bytes=2 * 1024 * 1024,
            max_member_bytes=1_000_000,
        )

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert stats["attachable"] is True
        assert stats["oversized_files"] == 1
        assert stats["byte_limited_files"] == 1
        assert "notes/summary.md" in names
        assert "data/oversized.bin" not in names
        assert len({"data/first.bin", "data/second.bin"} & names) == 1


def test_compute_handoff_keeps_archives_and_bulk_tables_local_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / "data").mkdir(parents=True)
        (root / "archive").mkdir()
        (root / "data" / "nested.zip").write_bytes(b"archive")
        (root / "archive" / "raw.db").write_bytes(b"database")
        with (root / "data" / "raw.tsv").open("wb") as handle:
            handle.truncate(6 * 1024 * 1024)
        (root / "data" / "certificate.txt").write_text(
            "compact certificate", encoding="utf-8"
        )

        out_zip = temp / "workspace.zip"
        stats = _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert "data/certificate.txt" in names
        assert "data/nested.zip" not in names
        assert "data/raw.tsv" not in names
        assert not any(name.startswith("archive/") for name in names)
        assert stats["local_only_files"] == 2


def test_compute_handoff_inspection_checks_every_provider_limit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "handoff.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("large.bin", b"x" * 100)

        assert not inspect_compute_handoff(
            path,
            max_files=10,
            max_compressed_bytes=10_000,
            max_uncompressed_bytes=10_000,
            max_member_bytes=99,
        ).attachable
        assert not inspect_compute_handoff(
            path,
            max_files=10,
            max_compressed_bytes=1,
            max_uncompressed_bytes=10_000,
            max_member_bytes=10_000,
        ).attachable
        assert not inspect_compute_handoff(
            path,
            max_files=10,
            max_compressed_bytes=10_000,
            max_uncompressed_bytes=99,
            max_member_bytes=10_000,
        ).attachable


def test_compute_handoff_normalizes_pre_1980_timestamps() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        old = root / "old.txt"
        old.write_text("historical", encoding="utf-8")
        os.utime(old, (0, 0))

        out_zip = temp / "workspace.zip"
        _zip_workspace(root, out_zip, exclude_top=_ZIP_EXCLUDE_TOP)

        with zipfile.ZipFile(out_zip) as zf:
            assert zf.getinfo("old.txt").date_time[0] == 1980


def test_workspace_usage_and_hard_limit_stop_are_enforced() -> None:
    class FakeStream:
        terminated = False

        @property
        def worker_stopped(self):
            return self.terminated

        async def terminate(self):
            self.terminated = True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "workspace"
        workspace.mkdir()
        (workspace / "large.bin").write_bytes(b"x" * 11)
        usage = measure_workspace_usage(workspace)
        assert usage["bytes"] == 11
        assert usage["files"] == 1
        assert usage["entries"] == 1

        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_SOFT_LIMIT_BYTES = 5
        agent.WORKSPACE_HARD_LIMIT_BYTES = 10
        agent.WORKSPACE_CHECK_INTERVAL_S = 0
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        stream = FakeStream()
        done = asyncio.run(
            agent._wait_for_done(
                stream,
                temp / "missing-done.json",
                spawn_call_id="call",
                workspace_root=workspace,
            )
        )

        assert done.status == "error"
        assert "workspace safety limit reached" in done.summary
        assert "workspace allocated size" in done.summary
        assert stream.terminated is True
        assert any(kind == "cli.workspace_limit_exceeded" for kind, _ in events)


def test_filesystem_free_space_guard_stops_compute_worker() -> None:
    from unittest import mock

    class FakeStream:
        terminated = False

        @property
        def worker_stopped(self):
            return self.terminated

        async def terminate(self):
            self.terminated = True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "workspace"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 1_000
        agent.WORKSPACE_MIN_FREE_BYTES = 100
        agent.WORKSPACE_CHECK_INTERVAL_S = 0
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        stream = FakeStream()
        fake_usage = {
            "bytes": 0,
            "files": 0,
            "errors": 0,
            "filesystem_free_bytes": 99,
        }
        with mock.patch(
            "proofstack.kinds.cli.measure_workspace_usage",
            return_value=fake_usage,
        ):
            done = asyncio.run(
                agent._wait_for_done(
                    stream,
                    temp / "missing-done.json",
                    spawn_call_id="call",
                    workspace_root=workspace,
                )
        )

        assert done.status == "error"
        assert "filesystem free space" in done.summary
        assert stream.terminated is True
        event = next(payload for kind, payload in events if kind == "cli.workspace_limit_exceeded")
        assert event["reason"] == "filesystem_min_free"


def test_workspace_entry_limit_stops_before_unbounded_scan() -> None:
    class FakeStream:
        terminated = False

        @property
        def worker_stopped(self):
            return self.terminated

        async def terminate(self):
            self.terminated = True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "workspace"
        workspace.mkdir()
        for idx in range(5):
            (workspace / f"empty-{idx}").touch()

        bounded = measure_workspace_usage(workspace, stop_after_entries=3)
        assert bounded["entries"] == 3
        assert bounded["scan_truncated"] is True

        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 1_000
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 3
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_MIN_FREE_INODES = 0
        agent.WORKSPACE_CHECK_INTERVAL_S = 0
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        stream = FakeStream()
        done = asyncio.run(
            agent._wait_for_done(
                stream,
                temp / "missing-done.json",
                spawn_call_id="call",
                workspace_root=workspace,
            )
        )

        assert done.status == "error"
        assert "workspace entry count" in done.summary
        event = next(
            payload for kind, payload in events if kind == "cli.workspace_limit_exceeded"
        )
        assert event["reason"] == "workspace_entry_limit"


def test_workspace_usage_tolerates_scandir_iteration_errors() -> None:
    class BrokenScandir:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("directory changed during scan")

    with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
        "proofstack.kinds.cli.os.scandir",
        return_value=BrokenScandir(),
    ), mock.patch(
        "proofstack.kinds.cli.os.statvfs",
        return_value=SimpleNamespace(
            f_files=0,
            f_favail=0,
            f_blocks=1,
            f_bfree=1,
            f_bavail=1,
            f_frsize=4096,
        ),
    ):
        usage = measure_workspace_usage(Path(temp_dir))

    assert usage["errors"] == 1
    assert usage["entries"] == 0
    assert usage["filesystem_free_inodes"] is None


def test_workspace_scan_errors_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test",
            root_workdir=Path(temp_dir) / "run",
            flat=True,
        )
        agent = Compute(ctx)
        failure = agent._workspace_limit_failure(
            {
                "bytes": 0,
                "allocated_bytes": 0,
                "allocated_bytes_supported": True,
                "entries": 0,
                "errors": 1,
                "filesystem_free_bytes": 1_000_000,
                "filesystem_free_inodes": 1_000_000,
            }
        )

    assert failure is not None
    assert failure[0] == "workspace_scan_error"


def test_missing_inode_accounting_skips_automatic_floor() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test",
            root_workdir=Path(temp_dir) / "run",
            flat=True,
        )
        agent = Compute(ctx)
        agent.WORKSPACE_MIN_FREE_INODES = COMPUTE_FILESYSTEM_MIN_FREE_INODES
        agent.WORKSPACE_REQUIRE_INODE_ACCOUNTING = False
        failure = agent._workspace_limit_failure(
            {
                "bytes": 0,
                "allocated_bytes": 0,
                "allocated_bytes_supported": True,
                "entries": 0,
                "errors": 0,
                "filesystem_free_bytes": 1_000_000,
                "filesystem_free_inodes": None,
            }
        )

    assert failure is None


def test_missing_inode_accounting_fails_for_explicit_floor() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test",
            root_workdir=Path(temp_dir) / "run",
            flat=True,
        )
        agent = Compute(ctx)
        agent.WORKSPACE_MIN_FREE_INODES = 100_000
        agent.WORKSPACE_REQUIRE_INODE_ACCOUNTING = True
        failure = agent._workspace_limit_failure(
            {
                "bytes": 0,
                "allocated_bytes": 0,
                "allocated_bytes_supported": True,
                "entries": 0,
                "errors": 0,
                "filesystem_free_bytes": 1_000_000,
                "filesystem_free_inodes": None,
            }
        )

    assert failure is not None
    assert failure[0] == "filesystem_free_inodes_unknown"


def test_workspace_scan_deletion_races_do_not_stop_worker() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test",
            root_workdir=Path(temp_dir) / "run",
            flat=True,
        )
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 100
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 100
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_MIN_FREE_INODES = 0
        failure = agent._workspace_limit_failure(
            {
                "bytes": 0,
                "allocated_bytes": 0,
                "allocated_bytes_supported": True,
                "entries": 0,
                "errors": 0,
                "transient_errors": 1,
                "filesystem_free_bytes": 1_000_000,
                "filesystem_free_inodes": 1_000_000,
            }
        )

    assert failure is None


def test_sparse_file_is_limited_by_allocated_not_apparent_size() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "workspace"
        workspace.mkdir()
        sparse = workspace / "sparse.bin"
        with sparse.open("wb") as handle:
            handle.truncate(1024 * 1024 * 1024)
        usage = measure_workspace_usage(workspace)
        if not usage["allocated_bytes_supported"]:
            return

        assert usage["bytes"] == 1024 * 1024 * 1024
        assert usage["allocated_bytes"] < usage["bytes"]
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 128 * 1024 * 1024
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 10
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_MIN_FREE_INODES = 0
        assert agent._workspace_limit_failure(usage) is None


def test_workspace_recovery_prompt_and_completion_require_soft_target() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        (root / ".pwc" / "runtime").mkdir(parents=True)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_SOFT_LIMIT_BYTES = 80
        agent.WORKSPACE_HARD_LIMIT_BYTES = 100
        agent.WORKSPACE_SOFT_LIMIT_ENTRIES = 8
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 10
        over = {
            "bytes": 110,
            "allocated_bytes": 110,
            "allocated_bytes_supported": True,
            "entries": 9,
            "errors": 0,
            "filesystem_free_bytes": 1_000_000,
            "filesystem_free_inodes": 1_000_000,
        }
        agent._configure_workspace_recovery(over)
        agent._write_workspace_pressure(root, over, reason="workspace_hard_limit")
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="launch another enumeration",
            compute_workspace=root,
            workspace_soft_limit_bytes=80,
            workspace_hard_limit_bytes=100,
            workspace_soft_limit_entries=8,
            workspace_hard_limit_entries=10,
        )

        prompt = agent.cli_input(inp)
        assert "STORAGE-RECOVERY INVOCATION" in prompt
        assert "launch another enumeration" not in prompt
        assert "continue the Author's commissioned" in prompt
        assert "unless this prompt began with" in prompt
        assert (root / ".pwc" / "runtime" / "STORAGE_PRESSURE").exists()

        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        agent._measure_workspace = lambda _root: dict(over)
        incomplete = asyncio.run(
            agent._finalize_workspace_recovery(
                CLIDoneRecord(status="done", summary="cleanup attempted"),
                root,
                spawn_call_id="call",
            )
        )
        assert incomplete.status == "partial"
        assert "storage recovery incomplete" in incomplete.summary

        below = {
            **over,
            "bytes": 40,
            "allocated_bytes": 40,
            "entries": 4,
        }
        agent._measure_workspace = lambda _root: dict(below)
        complete = asyncio.run(
            agent._finalize_workspace_recovery(
                CLIDoneRecord(status="done", summary="clean"),
                root,
                spawn_call_id="call",
            )
        )
        assert complete.status == "done"
        assert not (root / ".pwc" / "runtime" / "STORAGE_PRESSURE").exists()
        assert any(kind == "cli.workspace_recovery_completed" for kind, _ in events)


def test_workspace_recovery_clear_failure_downgrades_completion() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_SOFT_LIMIT_BYTES = 80
        agent.WORKSPACE_HARD_LIMIT_BYTES = 100
        usage = {
            "bytes": 40,
            "allocated_bytes": 40,
            "allocated_bytes_supported": True,
            "entries": 1,
            "errors": 0,
            "filesystem_free_bytes": 1_000_000,
            "filesystem_free_inodes": 1_000_000,
        }
        agent._configure_workspace_recovery(
            {**usage, "bytes": 110, "allocated_bytes": 110}
        )
        agent._write_workspace_pressure(
            root,
            {**usage, "bytes": 110, "allocated_bytes": 110},
            reason="workspace_hard_limit",
        )
        agent._measure_workspace = lambda _root: dict(usage)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        with mock.patch.object(
            agent,
            "_clear_workspace_recovery_attempts_recorded",
            new=mock.AsyncMock(return_value="OSError: read-only"),
        ):
            result = asyncio.run(
                agent._finalize_workspace_recovery(
                    CLIDoneRecord(status="done", summary="cleanup complete"),
                    root,
                    spawn_call_id="call",
                )
            )
        pressure = json.loads(
            (root / ".pwc" / "runtime" / "STORAGE_PRESSURE").read_text(
                encoding="utf-8"
            )
        )

    assert result.status == "partial"
    assert "attempt state could not be reset" in result.summary
    assert not any(kind == "cli.workspace_recovery_completed" for kind, _ in events)
    assert pressure["reason"] == "workspace_recovery_state_clear_failed"


def test_compute_storage_pressure_prompt_resumes_normal_research() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        inp = Compute.Inputs(
            problem="P",
            problem_id="p",
            round=1,
            instructions="continue the commissioned search",
            compute_workspace=root,
        )

        prompt = agent.cli_input(inp)

    assert not prompt.startswith("URGENT STORAGE-RECOVERY INVOCATION")
    assert "continue the Author's commissioned task" in " ".join(prompt.split())
    assert "continue the commissioned search" in prompt


def test_workspace_recovery_attempt_cap_survives_agent_reconstruction() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        over = {
            "bytes": 110,
            "allocated_bytes": 110,
            "allocated_bytes_supported": True,
            "entries": 11,
            "errors": 0,
            "filesystem_free_bytes": 1_000_000,
            "filesystem_free_inodes": 1_000_000,
        }

        first = Compute(ctx)
        first.WORKSPACE_HARD_LIMIT_BYTES = 100
        first.WORKSPACE_HARD_LIMIT_ENTRIES = 10
        first._configure_workspace_recovery(over)
        first_claim = first._claim_workspace_recovery_attempt(root)
        assert first_claim is not None
        first._commit_workspace_recovery_claim(first_claim)
        state_path = first.workspace_recovery_state_path_for(root)
        assert state_path.is_file()

        reconstructed = Compute(ctx)
        reconstructed.WORKSPACE_HARD_LIMIT_BYTES = 100
        reconstructed.WORKSPACE_HARD_LIMIT_ENTRIES = 10
        reconstructed._configure_workspace_recovery(over)
        assert reconstructed._claim_workspace_recovery_attempt(root) is None

        reconstructed._clear_workspace_recovery_attempts(root)
        assert not state_path.exists()
        retry_claim = reconstructed._claim_workspace_recovery_attempt(root)
        assert retry_claim is not None
        reconstructed._commit_workspace_recovery_claim(retry_claim)
        reconstructed._clear_workspace_recovery_attempts(root)


def test_aborted_workspace_recovery_claim_does_not_consume_attempt() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)

        first = Compute(ctx)
        first_claim = first._claim_workspace_recovery_attempt(root)
        assert first_claim is not None
        first._abort_workspace_recovery_claim(first_claim)
        assert not first.workspace_recovery_state_path_for(root).exists()

        reconstructed = Compute(ctx)
        retry_claim = reconstructed._claim_workspace_recovery_attempt(root)
        assert retry_claim is not None
        reconstructed._commit_workspace_recovery_claim(retry_claim)
        reconstructed._clear_workspace_recovery_attempts(root)


def test_workspace_recovery_checks_global_floor_before_claiming_attempt() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 100
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 100
        agent.WORKSPACE_MIN_FREE_BYTES = 10
        usage = {
            "bytes": 110,
            "allocated_bytes": 110,
            "allocated_bytes_supported": True,
            "entries": 1,
            "errors": 0,
            "filesystem_free_bytes": 5,
            "filesystem_free_inodes": 1_000_000,
        }
        initial_failure = agent._workspace_limit_failure(
            usage,
            recovery_ceiling=False,
        )
        assert initial_failure is not None
        assert initial_failure[0] == "workspace_hard_limit"

        claim, failure = agent._prepare_workspace_recovery_claim(usage, root)

        assert claim is None
        assert failure is not None
        assert failure[0] == "filesystem_min_free"
        assert not agent.workspace_recovery_state_path_for(root).exists()
        assert agent._workspace_recovery_mode is False


def test_workspace_recovery_spawn_failure_releases_claim() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_HARD_LIMIT_BYTES = 1
        WORKSPACE_HARD_LIMIT_ENTRIES = 100
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

    class SpawnFailureSandbox(SimpleNamespace):
        async def stream_command(self, *args, **kwargs):
            raise RuntimeError("spawn failed")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        (root / "existing.bin").write_bytes(b"payload")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        sandbox = SpawnFailureSandbox(root=root)

        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(RuntimeError, match="spawn failed"):
                asyncio.run(agent.run(ProbeInput(workspace=root)))

        assert not agent.workspace_recovery_state_path_for(root).exists()
        reconstructed = ProbeAgent(ctx)
        retry_claim = reconstructed._claim_workspace_recovery_attempt(root)
        assert retry_claim is not None
        reconstructed._abort_workspace_recovery_claim(retry_claim)


def test_workspace_recovery_stdin_failure_does_not_consume_attempt() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_HARD_LIMIT_BYTES = 1
        WORKSPACE_HARD_LIMIT_ENTRIES = 100
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        def cli_input(self, inp):
            return "clean the workspace"

        async def collect(self, sandbox, inp, done):
            return inp

    class BrokenStdin:
        def write(self, payload):
            self.payload = payload

        async def drain(self):
            raise BrokenPipeError("child exited during startup")

        def close(self):
            return None

    class StartedStream:
        done = True
        stdout = ""
        stderr = ""

        def __init__(self):
            self.proc = SimpleNamespace(stdin=BrokenStdin(), returncode=1)

        async def terminate(self):
            self.done = True

    class StartedSandbox(SimpleNamespace):
        async def stream_command(self, *args, **kwargs):
            return StartedStream()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        (root / "existing.bin").write_bytes(b"payload")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        sandbox = StartedSandbox(root=root)

        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="recovery prompt could not be delivered",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=root)))

        assert not agent.workspace_recovery_state_path_for(root).exists()
        reconstructed = ProbeAgent(ctx)
        retry_claim = reconstructed._claim_workspace_recovery_attempt(root)
        assert retry_claim is not None
        reconstructed._abort_workspace_recovery_claim(retry_claim)
        invocation_lock = reconstructed._acquire_workspace_invocation_lock(root)
        reconstructed._release_workspace_lock(invocation_lock)


def test_workspace_recovery_claim_serializes_concurrent_resumes() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        first = Compute(ctx)
        second = Compute(ctx)
        first_claim = first._claim_workspace_recovery_attempt(root)
        assert first_claim is not None

        with pytest.raises(RuntimeError, match="recovery state is busy"):
            second._claim_workspace_recovery_attempt(root)

        first._commit_workspace_recovery_claim(first_claim)
        assert second._claim_workspace_recovery_attempt(root) is None

        first._clear_workspace_recovery_attempts(root)


def test_workspace_invocation_lock_blocks_before_runtime_mutation() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

        async def teardown(self, sandbox, inp):
            self.teardown_called = True

    class TrackingSandbox(SimpleNamespace):
        spawned = False

        async def stream_command(self, *args, **kwargs):
            self.spawned = True
            raise AssertionError("busy workspace must not spawn another worker")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        runtime = root / ".pwc" / "runtime"
        runtime.mkdir(parents=True)
        done_path = runtime / "done.json"
        wrap_up_path = runtime / "WRAP_UP"
        done_path.write_text("active done record", encoding="utf-8")
        wrap_up_path.write_text("active wrap-up signal", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        owner = ProbeAgent(ctx)
        contender = ProbeAgent(ctx)
        contender.teardown_called = False
        sandbox = TrackingSandbox(root=root)
        invocation_lock = owner._acquire_workspace_invocation_lock(root)
        try:
            with mock.patch(
                "proofstack.kinds.cli.make_sandbox",
                return_value=sandbox,
            ):
                with pytest.raises(
                    RuntimeError,
                    match="persistent workspace is already active",
                ):
                    asyncio.run(contender.run(ProbeInput(workspace=root)))
        finally:
            owner._release_workspace_lock(invocation_lock)

        assert done_path.read_text(encoding="utf-8") == "active done record"
        assert wrap_up_path.read_text(encoding="utf-8") == "active wrap-up signal"
        assert sandbox.spawned is False
        assert contender.teardown_called is False


def test_workspace_invocation_lock_is_shared_across_run_roots() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "shared" / "compute"
        root.mkdir(parents=True)
        first = Compute(
            RunContext.create(
                run_id="first",
                root_workdir=temp / "first-run",
                flat=True,
            )
        )
        second = Compute(
            RunContext.create(
                run_id="second",
                root_workdir=temp / "second-run",
                flat=True,
            )
        )

        assert (
            first.workspace_invocation_lock_path_for(root)
            == second.workspace_invocation_lock_path_for(root)
        )
        invocation_lock = first._acquire_workspace_invocation_lock(root)
        try:
            with pytest.raises(
                RuntimeError,
                match="persistent workspace is already active",
            ):
                second._acquire_workspace_invocation_lock(root)
        finally:
            first._release_workspace_lock(invocation_lock)


def test_subprocess_worker_retains_invocation_lock_after_parent_close() -> None:
    async def exercise(temp: Path) -> None:
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(
            run_id="owner",
            root_workdir=temp / "owner-run",
            flat=True,
        )
        owner = Compute(ctx)
        contender = Compute(
            RunContext.create(
                run_id="contender",
                root_workdir=temp / "contender-run",
                flat=True,
            )
        )
        invocation_lock = owner._acquire_workspace_invocation_lock(root)
        sandbox = SubprocessSandbox(
            SandboxSpec(
                backend="subprocess",
                provider_keys=(),
                timeout_s=30,
            ),
            root=root,
            inherited_fds=(invocation_lock,),
        )
        stream = await sandbox.stream_command(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )

        # Simulate abrupt orchestrator death: its descriptor is closed without
        # LOCK_UN, while the spawned worker still owns the shared lease.
        owner._close_workspace_invocation_lock(invocation_lock)
        try:
            with pytest.raises(
                RuntimeError,
                match="persistent workspace is already active",
            ):
                contender._acquire_workspace_invocation_lock(root)
        finally:
            await stream.terminate()

        reacquired = contender._acquire_workspace_invocation_lock(root)
        contender._release_workspace_lock(reacquired)

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_docker_workspace_identity_is_stable_and_blocks_orphans() -> None:
    async def exercise(temp: Path) -> None:
        root = temp / "compute"
        root.mkdir()
        alias = temp / "compute-alias"
        alias.symlink_to(root, target_is_directory=True)
        assert _container_name_for_root(alias) == _container_name_for_root(root)

        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=root,
        )
        docker_cmd = sandbox._build_docker_cmd(
            ["probe"],
            env_extra=None,
            extra_path=[],
            cwd=None,
            interactive=False,
            container_name=sandbox.container_name,
        )
        label_index = docker_cmd.index("--label")
        assert docker_cmd[label_index + 1].endswith(sandbox.container_owner)
        with mock.patch(
            "proofstack.sandbox.docker._docker_container_status",
            new=mock.AsyncMock(return_value="running"),
        ):
            with pytest.raises(
                DockerSandboxError,
                match="refusing to overlap a resumed invocation",
            ):
                await sandbox.ensure_workspace_available()

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_stream_backends_report_definitive_client_spawn_failures() -> None:
    async def exercise(temp: Path) -> None:
        subprocess_sandbox = SubprocessSandbox(
            SandboxSpec(backend="subprocess"),
            root=temp / "subprocess",
        )
        with mock.patch(
            "proofstack.sandbox.subprocess.asyncio.create_subprocess_exec",
            new=mock.AsyncMock(side_effect=FileNotFoundError("missing binary")),
        ):
            with pytest.raises(SandboxSpawnError, match="missing binary"):
                await subprocess_sandbox.stream_command(["missing-probe"])

        docker_sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        with mock.patch(
            "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
            new=mock.AsyncMock(side_effect=FileNotFoundError("missing docker")),
        ):
            with pytest.raises(DockerSandboxError, match="missing docker") as exc:
                await docker_sandbox.stream_command(["probe"])
        assert isinstance(exc.value, SandboxSpawnError)

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--name", "alternate"),
        ("--name=alternate",),
        ("--cidfile", "/tmp/alternate.cid"),
        ("--detach",),
        ("-d",),
        ("-itd",),
        ("--rm=false",),
        ("--label-file", "/tmp/labels"),
        ("--label", "proofstack.workspace-owner=alternate"),
        ("--label=proofstack.workspace-owner=alternate",),
        ("-lproofstack.workspace-owner=alternate",),
    ],
)
def test_docker_extra_args_cannot_override_lifecycle_identity(
    extra_args: tuple[str, ...],
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(DockerSandboxError, match="docker_extra_args cannot"):
            DockerSandbox(
                SandboxSpec(backend="docker", docker_extra_args=extra_args),
                root=Path(temp_dir) / "docker",
            )


def test_docker_extra_args_allow_non_lifecycle_options() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox = DockerSandbox(
            SandboxSpec(
                backend="docker",
                docker_extra_args=(
                    "--label",
                    "proofstack.test=allowed",
                    "-v",
                    f"{temp_dir}:/extra:ro",
                ),
            ),
            root=Path(temp_dir) / "docker",
        )

        assert sandbox.spec.docker_extra_args[0] == "--label"


def test_docker_stream_startup_exposes_cleanup_lifecycle() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        proc = SimpleNamespace(stdout=None, stderr=None)
        stream = SimpleNamespace(
            confirm_launch=mock.AsyncMock(
                side_effect=DockerSandboxError("launch inspect failed")
            ),
            terminate=mock.AsyncMock(),
            worker_stop_state=WorkerStopState.STOPPED,
            launch_settled=True,
        )
        with (
            mock.patch(
                "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=proc),
            ),
            mock.patch(
                "proofstack.sandbox.docker._DockerStreamingProcess",
                return_value=stream,
            ),
        ):
            with pytest.raises(DockerSandboxError, match="launch inspect failed"):
                await sandbox.stream_command(["probe"])

        stream.terminate.assert_awaited_once()
        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        assert sandbox.worker_launch_settled is True

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_docker_stream_launch_uses_a_separate_bounded_deadline() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(
                backend="docker",
                timeout_s=9_000,
                docker_launch_timeout_s=60,
            ),
            root=temp / "docker",
        )
        proc = SimpleNamespace(stdout=None, stderr=None)
        stream = SimpleNamespace(
            confirm_launch=mock.AsyncMock(),
            worker_stop_state=WorkerStopState.SURVIVING,
            launch_settled=True,
        )
        constructor = mock.Mock(return_value=stream)
        with (
            mock.patch(
                "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=proc),
            ),
            mock.patch(
                "proofstack.sandbox.docker._DockerStreamingProcess",
                new=constructor,
            ),
            mock.patch(
                "proofstack.sandbox.docker.time.monotonic",
                return_value=100.0,
            ),
        ):
            assert await sandbox.stream_command(["probe"], timeout_s=9_000) is stream

        call = constructor.call_args
        assert call is not None
        assert call.kwargs["deadline"] == 9_100.0
        assert call.kwargs["launch_deadline"] == 160.0
        stream.confirm_launch.assert_awaited_once()
        call.kwargs["launch_receipt"].cleanup()

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_docker_run_nonzero_exit_does_not_settle_absent_container() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        proc = SimpleNamespace(
            communicate=mock.AsyncMock(return_value=(b"", b"docker failed")),
            returncode=125,
        )
        docker_kill = mock.AsyncMock(return_value=WorkerStopState.UNKNOWN)
        with (
            mock.patch(
                "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=proc),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            result = await sandbox.run_command(["probe"])

        assert result.returncode == 125
        assert "stop state is unknown" in result.stderr
        assert sandbox.worker_stop_state is WorkerStopState.UNKNOWN
        assert sandbox.worker_launch_settled is False
        docker_kill.assert_awaited_once_with(
            sandbox.container_name,
            owner=sandbox.container_owner,
            absence_is_terminal=False,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_docker_run_nonzero_inner_exit_is_settled_by_cidfile() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        proc = SimpleNamespace(
            communicate=mock.AsyncMock(return_value=(b"", b"inner failure")),
            returncode=1,
        )
        cidfiles: list[Path] = []

        async def spawn(*args, **kwargs):
            cidfile = Path(args[args.index("--cidfile") + 1])
            cidfile.write_text("a" * 64, encoding="ascii")
            cidfiles.append(cidfile)
            return proc

        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)
        with (
            mock.patch(
                "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
                new=spawn,
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            result = await sandbox.run_command(["probe"])

        assert result.returncode == 1
        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        assert sandbox.worker_launch_settled is True
        docker_kill.assert_awaited_once_with(
            sandbox.container_name,
            owner=sandbox.container_owner,
            absence_is_terminal=True,
        )
        assert len(cidfiles) == 1
        assert not cidfiles[0].exists()
        assert not cidfiles[0].parent.exists()

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_docker_run_cleanup_observes_late_cidfile_receipt() -> None:
    async def exercise() -> None:
        receipt = _DockerLaunchReceipt()
        proc = SimpleNamespace(kill=mock.Mock())

        async def communicate():
            receipt.path.write_text("d" * 64, encoding="ascii")
            return b"", b""

        proc.communicate = communicate
        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)
        try:
            with mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ):
                stop_state, launch_settled = await _terminate_docker_run(
                    proc,
                    "workspace",
                    owner="our-owner",
                    launch_settled=False,
                    launch_receipt=receipt,
                )

            assert stop_state is WorkerStopState.STOPPED
            assert launch_settled is True
            docker_kill.assert_awaited_once_with(
                "workspace",
                owner="our-owner",
                absence_is_terminal=True,
            )
        finally:
            receipt.cleanup()

    asyncio.run(exercise())


def test_docker_launch_receipt_cleanup_retries_transient_failure() -> None:
    receipt = _DockerLaunchReceipt()
    receipt.path.write_text("a" * 64, encoding="ascii")
    original_unlink = Path.unlink
    attempts = 0

    def flaky_unlink(path: Path, *args, **kwargs) -> None:
        nonlocal attempts
        if path == receipt.path and attempts == 0:
            attempts += 1
            raise OSError("transient unlink failure")
        original_unlink(path, *args, **kwargs)

    try:
        with mock.patch.object(Path, "unlink", new=flaky_unlink):
            assert receipt.cleanup() is False
            assert receipt.path.exists()
            assert receipt._cleaned is False

            assert receipt.cleanup() is True
            assert receipt._cleaned is True
            assert not receipt.path.exists()
            assert not receipt.directory.exists()
    finally:
        receipt.cleanup()


def test_docker_stream_retains_launch_receipt_until_cleanup_succeeds() -> None:
    receipt = SimpleNamespace(cleanup=mock.Mock(side_effect=[False, True]))
    stream = object.__new__(_DockerStreamingProcess)
    stream._launch_receipt = receipt

    stream._cleanup_launch_receipt()
    assert stream._launch_receipt is receipt

    stream._cleanup_launch_receipt()
    assert stream._launch_receipt is None
    assert receipt.cleanup.call_count == 2


def test_docker_sandbox_refuses_to_overwrite_unresolved_lifecycle() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        sandbox._set_worker_lifecycle(
            WorkerStopState.UNKNOWN,
            launch_settled=False,
        )
        spawn = mock.AsyncMock()
        with mock.patch(
            "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
            new=spawn,
        ):
            with pytest.raises(
                DockerSandboxError,
                match="prior worker is active or its launch state is unresolved",
            ):
                await sandbox.run_command(["second-command"])
        spawn.assert_not_awaited()
        assert sandbox.worker_stop_state is WorkerStopState.UNKNOWN
        assert sandbox.worker_launch_settled is False

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_subprocess_setup_command_cancellation_drains_cleanup() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = SubprocessSandbox(
            SandboxSpec(backend="subprocess"),
            root=temp / "subprocess",
        )
        communicate_started = asyncio.Event()

        async def communicate(*, input=None):
            communicate_started.set()
            await asyncio.Future()

        proc = SimpleNamespace(
            communicate=communicate,
            returncode=None,
            pid=12345,
        )
        cleanup = mock.AsyncMock(return_value=True)
        with (
            mock.patch(
                "proofstack.sandbox.subprocess.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=proc),
            ),
            mock.patch(
                "proofstack.sandbox.subprocess._terminate_process_group_uninterruptibly",
                new=cleanup,
            ),
        ):
            task = asyncio.create_task(sandbox.run_command(["probe"]))
            await communicate_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        cleanup.assert_awaited_once_with(
            proc,
            grace_s=5.0,
            process_marker=mock.ANY,
        )
        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        assert sandbox.worker_launch_settled is True

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_subprocess_run_command_cleans_background_descendants() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = SubprocessSandbox(
            SandboxSpec(backend="subprocess", provider_keys=()),
            root=temp / "subprocess",
        )
        with mock.patch(
            "proofstack.sandbox.subprocess._terminate_marked_processes",
            new=mock.AsyncMock(return_value=True),
        ):
            result = await sandbox.run_command(
                [
                    "sh",
                    "-c",
                    "sleep 30 </dev/null >/dev/null 2>&1 & echo $!",
                ]
            )
        descendant_pid = int(result.stdout.strip())

        assert result.returncode == 0
        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        with pytest.raises(ProcessLookupError):
            os.kill(descendant_pid, 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_subprocess_never_reports_stopped_while_setsid_descendant_survives() -> None:
    async def exercise(temp: Path) -> None:
        inspection_available = _find_marked_processes(
            _ProcessMarker(token="unmatched-test-token", created_at=time.time())
        )[1]
        sandbox = SubprocessSandbox(
            SandboxSpec(backend="subprocess", provider_keys=()),
            root=temp / "subprocess",
        )
        result = await sandbox.run_command(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys; "
                    "p = subprocess.Popen("
                    "[sys.executable, '-c', 'import time; time.sleep(30)'], "
                    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                    "stderr=subprocess.DEVNULL, start_new_session=True); "
                    "print(p.pid, flush=True)"
                ),
            ]
        )
        descendant_pid = int(result.stdout.strip())
        try:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                descendant_alive = False
            else:
                descendant_alive = True

            assert not (
                sandbox.worker_stop_state is WorkerStopState.STOPPED
                and descendant_alive
            )
            if inspection_available:
                assert sandbox.worker_stop_state is WorkerStopState.STOPPED
                assert descendant_alive is False
            assert sandbox.worker_stop_state in {
                WorkerStopState.STOPPED,
                WorkerStopState.UNKNOWN,
            }
        finally:
            try:
                os.killpg(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_subprocess_stream_cleanup_is_idempotent_after_verified_stop() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=0)
        stream = _StreamingProcess(
            proc=proc,
            cmd=["probe"],
            deadline=time.monotonic() + 10,
        )
        cleanup = mock.AsyncMock(return_value=True)
        with mock.patch(
            "proofstack.sandbox.subprocess._terminate_process_group_uninterruptibly",
            new=cleanup,
        ):
            await stream.terminate()
            await stream.terminate()

        cleanup.assert_awaited_once()
        assert stream.worker_stop_state is WorkerStopState.STOPPED

    asyncio.run(exercise())


def test_marked_process_cleanup_fails_closed_when_scan_is_incomplete() -> None:
    marker = _ProcessMarker(token="token", created_at=0.0)
    with (
        mock.patch(
            "proofstack.sandbox.subprocess._find_marked_processes",
            return_value=([], False),
        ),
        mock.patch(
            "proofstack.sandbox.subprocess.PROCESS_MARKER_EXIT_TIMEOUT_S",
            0.0,
        ),
    ):
        assert asyncio.run(_terminate_marked_processes(marker)) is False


def test_marked_process_cleanup_terminates_and_verifies_descendant() -> None:
    process = SimpleNamespace(
        terminate=mock.Mock(),
        kill=mock.Mock(),
    )
    scan = mock.Mock(
        side_effect=[
            ([process], True),
            ([], True),
            ([], True),
            ([], True),
            ([], True),
        ]
    )
    with (
        mock.patch(
            "proofstack.sandbox.subprocess._find_marked_processes",
            new=scan,
        ),
        mock.patch(
            "proofstack.sandbox.subprocess.asyncio.sleep",
            new=mock.AsyncMock(),
        ),
    ):
        assert asyncio.run(
            _terminate_marked_processes(
                _ProcessMarker(token="token", created_at=0.0)
            )
        ) is True

    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()


def test_marked_process_cleanup_kills_known_process_after_incomplete_scan() -> None:
    process = SimpleNamespace(
        terminate=mock.Mock(),
        kill=mock.Mock(),
    )
    scan = mock.Mock(
        side_effect=[
            ([process], False),
            ([], True),
            ([], True),
            ([], True),
            ([], True),
        ]
    )
    with (
        mock.patch(
            "proofstack.sandbox.subprocess._find_marked_processes",
            new=scan,
        ),
        mock.patch(
            "proofstack.sandbox.subprocess.asyncio.sleep",
            new=mock.AsyncMock(),
        ),
    ):
        assert asyncio.run(
            _terminate_marked_processes(
                _ProcessMarker(token="token", created_at=0.0)
            )
        ) is True

    process.terminate.assert_called_once_with()


def test_docker_cleanup_only_removes_the_owned_container() -> None:
    async def exercise() -> None:
        docker_control = mock.AsyncMock(return_value=(0, "", ""))
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=mock.AsyncMock(
                    return_value=("running", "foreign-owner", "foreign-id")
                ),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_control",
                new=docker_control,
            ),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.SURVIVING
            )
        docker_control.assert_not_awaited()

        docker_control = mock.AsyncMock(return_value=(0, "", ""))
        identity = mock.AsyncMock(
            side_effect=[
                ("running", "our-owner", "our-id"),
                None,
                None,
                None,
            ]
        )
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_control",
                new=docker_control,
            ),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.STOPPED
            )
        assert docker_control.await_args_list == [
            mock.call("kill", "our-id"),
            mock.call("rm", "--force", "our-id"),
        ]

    asyncio.run(exercise())


def test_docker_cleanup_distinguishes_unknown_from_surviving() -> None:
    async def exercise() -> None:
        with mock.patch(
            "proofstack.sandbox.docker._docker_container_identity",
            new=mock.AsyncMock(side_effect=DockerSandboxError("daemon timeout")),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.UNKNOWN
            )

        identity = mock.AsyncMock(
            side_effect=[
                ("running", "our-owner", "our-id"),
                ("running", "our-owner", "our-id"),
            ]
        )
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_control",
                new=mock.AsyncMock(return_value=(1, "", "daemon busy")),
            ),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.SURVIVING
            )

    asyncio.run(exercise())


def test_docker_cleanup_requires_settled_absence() -> None:
    async def exercise() -> None:
        identity = mock.AsyncMock(
            side_effect=[None, ("creating", "foreign-owner", "foreign-id")]
        )
        sleep = mock.AsyncMock()
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch("proofstack.sandbox.docker.asyncio.sleep", new=sleep),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.SURVIVING
            )
        assert identity.await_count == 2
        sleep.assert_awaited_once()

        identity = mock.AsyncMock(side_effect=[None, None, None])
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch(
                "proofstack.sandbox.docker.asyncio.sleep",
                new=mock.AsyncMock(),
            ),
        ):
            assert (
                await _docker_kill("workspace", owner="our-owner")
                is WorkerStopState.STOPPED
            )
        assert identity.await_count == 3

        identity = mock.AsyncMock(side_effect=[None, None, None])
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch(
                "proofstack.sandbox.docker.asyncio.sleep",
                new=mock.AsyncMock(),
            ),
        ):
            assert (
                await _docker_kill(
                    "workspace",
                    owner="our-owner",
                    absence_is_terminal=False,
                )
                is WorkerStopState.UNKNOWN
            )

    asyncio.run(exercise())


def test_docker_launch_is_observed_beyond_stop_absence_window() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=None)
        identity = mock.AsyncMock(
            side_effect=[None] * 55
            + [("running", "our-owner", "our-id")]
        )
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch(
                "proofstack.sandbox.docker.asyncio.sleep",
                new=mock.AsyncMock(),
            ),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=1_000_000_000.0,
                container_name="workspace",
                container_owner="our-owner",
            )
            await stream.confirm_launch()

        # The old fixed 50-probe limit rejected healthy but slow launches.
        assert identity.await_count == 56
        assert all(call.kwargs["timeout_s"] > 0 for call in identity.await_args_list)
        assert stream.worker_stop_state is WorkerStopState.SURVIVING

        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)
        # This test replaces the base terminator, so model its successful
        # process-group cleanup explicitly.
        stream._process_group_stop_state = WorkerStopState.STOPPED
        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=mock.AsyncMock(),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        docker_kill.assert_awaited_once_with(
            "workspace",
            owner="our-owner",
            absence_is_terminal=True,
        )

    asyncio.run(exercise())


def test_docker_launch_confirmation_respects_command_deadline() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=None)
        identity = mock.AsyncMock(return_value=None)
        sleep = mock.AsyncMock()
        with (
            mock.patch(
                "proofstack.sandbox.docker._docker_container_identity",
                new=identity,
            ),
            mock.patch("proofstack.sandbox.docker.asyncio.sleep", new=sleep),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=0.0,
                container_name="workspace",
                container_owner="our-owner",
            )
            with pytest.raises(DockerSandboxError, match="launch deadline"):
                await stream.confirm_launch()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        identity.assert_not_awaited()
        sleep.assert_not_awaited()

    asyncio.run(exercise())


def test_docker_nonzero_client_exit_does_not_settle_stream_launch() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=125)
        stream = _DockerStreamingProcess(
            proc=proc,
            cmd=["docker", "run"],
            deadline=1_000_000_000.0,
            container_name="workspace",
            container_owner="our-owner",
        )

        with pytest.raises(
            DockerSandboxError,
            match="daemon-side completion is unknown",
        ):
            await stream.confirm_launch()
        await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert stream.launch_settled is False
        assert stream.worker_stop_state is WorkerStopState.UNKNOWN

    asyncio.run(exercise())


def test_docker_stream_cidfile_settles_fast_inner_failure() -> None:
    async def exercise(temp: Path) -> None:
        sandbox = DockerSandbox(
            SandboxSpec(backend="docker"),
            root=temp / "docker",
        )
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=1)
        cidfiles: list[Path] = []

        async def spawn(*args, **kwargs):
            cidfile = Path(args[args.index("--cidfile") + 1])
            cidfile.write_text("b" * 64, encoding="ascii")
            cidfiles.append(cidfile)
            return proc

        with mock.patch(
            "proofstack.sandbox.docker.asyncio.create_subprocess_exec",
            new=spawn,
        ):
            stream = await sandbox.stream_command(["probe"])

        assert stream.launch_settled is True
        assert stream.worker_stop_state is WorkerStopState.UNKNOWN
        assert sandbox.worker_launch_settled is True
        assert sandbox.worker_stop_state is WorkerStopState.UNKNOWN
        assert len(cidfiles) == 1
        assert cidfiles[0].exists()

        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)
        # This test replaces the base terminator, so model its successful
        # process-group cleanup explicitly.
        stream._process_group_stop_state = WorkerStopState.STOPPED
        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=mock.AsyncMock(),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        assert sandbox.worker_launch_settled is True
        assert not cidfiles[0].exists()
        assert not cidfiles[0].parent.exists()
        docker_kill.assert_awaited_once_with(
            sandbox.container_name,
            owner=sandbox.container_owner,
            absence_is_terminal=True,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_unobserved_docker_launch_never_treats_absence_as_terminal() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=None)
        docker_kill = mock.AsyncMock(return_value=WorkerStopState.UNKNOWN)
        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=mock.AsyncMock(),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=1_000_000_000.0,
                container_name="workspace",
                container_owner="our-owner",
            )
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert stream.worker_stop_state is WorkerStopState.UNKNOWN
        docker_kill.assert_awaited_once_with(
            "workspace",
            owner="our-owner",
            absence_is_terminal=False,
        )

    asyncio.run(exercise())


def test_docker_stop_remains_unknown_when_client_group_is_unconfirmed() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=None)
        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)
        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=mock.AsyncMock(),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=1_000_000_000.0,
                container_name="workspace",
                container_owner="our-owner",
            )
            stream._launch_settled = True
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert stream.worker_stop_state is WorkerStopState.UNKNOWN
        docker_kill.assert_awaited_once_with(
            "workspace",
            owner="our-owner",
            absence_is_terminal=False,
        )

    asyncio.run(exercise())


def test_docker_termination_observes_late_cidfile_receipt() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None, returncode=None)
        receipt = _DockerLaunchReceipt()
        docker_kill = mock.AsyncMock(return_value=WorkerStopState.STOPPED)

        async def terminate_client(_stream) -> None:
            receipt.path.write_text("c" * 64, encoding="ascii")
            _stream._process_group_stop_state = WorkerStopState.STOPPED

        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=terminate_client,
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=1_000_000_000.0,
                container_name="workspace",
                container_owner="our-owner",
                launch_receipt=receipt,
            )
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert stream.launch_settled is True
        assert stream.worker_stop_state is WorkerStopState.STOPPED
        assert not receipt.path.exists()
        assert not receipt.directory.exists()
        docker_kill.assert_awaited_once_with(
            "workspace",
            owner="our-owner",
            absence_is_terminal=True,
        )

    asyncio.run(exercise())


def test_docker_confirmed_stop_is_not_downgraded_by_final_cleanup() -> None:
    async def exercise() -> None:
        proc = SimpleNamespace(stdout=None, stderr=None)
        docker_kill = mock.AsyncMock(
            side_effect=[WorkerStopState.STOPPED, WorkerStopState.UNKNOWN]
        )
        with (
            mock.patch.object(
                _StreamingProcess,
                "terminate",
                new=mock.AsyncMock(),
            ),
            mock.patch(
                "proofstack.sandbox.docker._docker_kill",
                new=docker_kill,
            ),
        ):
            stream = _DockerStreamingProcess(
                proc=proc,
                cmd=["docker", "run"],
                deadline=1_000_000_000.0,
                container_name="workspace",
                container_owner="our-owner",
            )
            stream._launch_settled = True
            stream._process_group_stop_state = WorkerStopState.STOPPED
            await stream.terminate()
            await stream.terminate()
            await asyncio.gather(stream._stdout_task, stream._stderr_task)

        assert stream.worker_stop_state is WorkerStopState.STOPPED
        assert stream.worker_stopped is True
        docker_kill.assert_awaited_once_with(
            "workspace",
            owner="our-owner",
            absence_is_terminal=True,
        )

    asyncio.run(exercise())


def test_unconfirmed_worker_stop_suppresses_collection_without_quarantine() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.collected = False
            self.refreshes = 0

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def refresh_sensitive_state(self, sandbox, inp):
            self.refreshes += 1

        async def collect(self, sandbox, inp, done):
            self.collected = True
            return inp

    class UnknownStopStream:
        done = True
        worker_stop_state = WorkerStopState.UNKNOWN
        worker_stopped = False
        proc = SimpleNamespace(stdin=None, returncode=0)
        stdout = "untrusted output"
        stderr = ""
        metering_stdout = ""
        stdout_chars = len(stdout)
        stderr_chars = 0

        def __init__(self, guard_to_remove: Path | None = None):
            self.terminations = 0
            self.guard_to_remove = guard_to_remove

        async def terminate(self):
            self.terminations += 1
            if self.terminations == 1 and self.guard_to_remove is not None:
                self.guard_to_remove.unlink()

    class UnknownStopSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            return self.stream

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        agent.WORKSPACE_RESERVATION_BYTES = 1024 * 1024
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_RESERVATION_DIR = temp / "reservations"
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        stream = UnknownStopStream(
            agent.workspace_active_guard_path_for(workspace)
        )
        sandbox = UnknownStopSandbox(root=workspace, stream=stream)

        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="CLI worker could not be confirmed stopped",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert agent.collected is False
        assert agent.refreshes == 0
        assert stream.terminations == 2
        stop_events = [
            payload
            for kind, payload in events
            if kind == "cli.worker_stop_unconfirmed"
        ]
        assert stop_events == [
            {
                "state": "unknown",
                "phase": "process_exit",
                "response_suppressed": True,
                "handoff_suppressed": True,
                "workspace_quarantine_on_resume": True,
                "workspace_quarantined": False,
            }
        ]
        assert agent.workspace_active_guard_path_for(workspace).exists()
        assert not (
            workspace / ".pwc" / "runtime" / "SENSITIVE_STATE_UNTRUSTED"
        ).exists()
        assert not agent.sensitive_quarantine_path_for(workspace).exists()
        assert not any(
            kind == "cli.sensitive_state_refresh_failed" for kind, _ in events
        )
        assert any(
            kind == "cli.storage_retained"
            and payload["retention_guard_state"] == "repaired"
            for kind, payload in events
        )
        retained_leases = list(
            (temp / "reservations").glob("lease-*.json")
        )
        assert len(retained_leases) == 1
        retained_payload = json.loads(retained_leases[0].read_text(encoding="utf-8"))
        assert retained_payload["retention_guard"] == str(
            agent.workspace_active_guard_path_for(workspace)
        )


def test_storage_reservation_requires_workspace_recovery_guard() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_RESERVATION_BYTES = 1
        WORKSPACE_MIN_FREE_BYTES = 0

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            raise AssertionError("worker must not start")

    class IdleSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            raise AssertionError("worker must not start")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        sandbox = IdleSandbox(root=workspace)

        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="storage reservations require WORKSPACE_RECOVERY_ENABLED",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))


def test_unconfirmed_stop_quarantines_persistent_workspace_without_guard() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

    class UnknownStopStream:
        done = True
        worker_stop_state = WorkerStopState.UNKNOWN
        worker_stopped = False
        proc = SimpleNamespace(stdin=None, returncode=0)
        stdout = ""
        stderr = ""
        metering_stdout = ""
        stdout_chars = 0
        stderr_chars = 0

        async def terminate(self):
            return None

    class UnknownStopSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            return UnknownStopStream()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "persistent"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = UnknownStopSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="CLI worker could not be confirmed stopped",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert (
            workspace / ".pwc" / "runtime" / "SENSITIVE_STATE_UNTRUSTED"
        ).exists()
        assert agent.sensitive_quarantine_path_for(workspace).exists()
        assert not agent.workspace_active_guard_path_for(workspace).exists()
        assert any(
            kind == "cli.worker_stop_unconfirmed"
            and payload["workspace_quarantined"] is True
            and payload["workspace_quarantine_on_resume"] is False
            for kind, payload in events
        )


def test_unconfirmed_worker_guard_preserves_workspace_before_resume() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeOutput(BaseModel):
        workspace: Path
        workspace_recovery_notice: str | None = None

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def setup(self, sandbox, inp):
            assert not (sandbox.root / "unknown-secret.txt").exists()
            (sandbox.root / "fresh.txt").write_text("fresh", encoding="utf-8")

        async def collect(self, sandbox, inp, done):
            return ProbeOutput(workspace=inp.workspace)

        def attach_workspace_recovery_notice(self, out, notice):
            return out.model_copy(update={"workspace_recovery_notice": notice})

    class StoppedStream:
        done = True
        worker_stop_state = WorkerStopState.STOPPED
        worker_stopped = True
        remaining_s = 10.0
        proc = SimpleNamespace(stdin=None, returncode=0)
        stdout = ""
        stderr = ""
        metering_stdout = ""
        stdout_chars = 0
        stderr_chars = 0

        async def terminate(self):
            return None

    class StoppedSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            return StoppedStream()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        (workspace / "unknown-secret.txt").write_text(
            "rotated credential",
            encoding="utf-8",
        )
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(guard, workspace, backend="docker")
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = StoppedSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            out = asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert out.workspace == workspace
        assert out.workspace_recovery_notice is not None
        assert "preserved in host-side quarantine" in out.workspace_recovery_notice
        assert "started with a fresh workspace" in out.workspace_recovery_notice
        assert not (workspace / "unknown-secret.txt").exists()
        assert (workspace / "fresh.txt").read_text(encoding="utf-8") == "fresh"
        assert not guard.exists()
        quarantine_events = [
            payload
            for kind, payload in events
            if kind == "cli.workspace_quarantined_after_unconfirmed_stop"
        ]
        assert len(quarantine_events) == 1
        quarantine_path = Path(quarantine_events[0]["quarantine_path"])
        assert quarantine_events[0]["fresh_workspace_created"] is True
        assert (quarantine_path / "unknown-secret.txt").read_text(
            encoding="utf-8"
        ) == "rotated credential"


def test_compute_workspace_recovery_notice_is_rendered_for_author() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="test",
            root_workdir=Path(temp_dir) / "run",
            flat=True,
        )
        agent = Compute(ctx)
        notice = (
            "A previous persistent workspace was preserved in host-side "
            "quarantine; this invocation started with a fresh workspace."
        )

        out = agent.attach_workspace_recovery_notice(
            Compute.Outputs(status="done", response_md="new result"),
            notice,
        )

        assert isinstance(out, Compute.Outputs)
        assert out.workspace_recovery_notice == notice
        rendered = render_compute_reply_for_author(out)
        assert f"workspace recovery: {notice}" in rendered
        assert "new result" in rendered


def test_teardown_failure_preserves_workspace_and_notifies_next_round() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeOutput(BaseModel):
        workspace: Path
        workspace_recovery_notice: str | None = None

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.setup_calls = 0
            self.teardown_calls = 0

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def setup(self, sandbox, inp):
            self.setup_calls += 1
            if self.setup_calls == 2:
                assert not (sandbox.root / "paper.pdf").exists()
                (sandbox.root / "fresh.txt").write_text("fresh", encoding="utf-8")

        async def collect(self, sandbox, inp, done):
            return ProbeOutput(workspace=inp.workspace)

        def attach_workspace_recovery_notice(self, out, notice):
            return out.model_copy(update={"workspace_recovery_notice": notice})

        async def teardown(self, sandbox, inp):
            self.teardown_calls += 1
            if self.teardown_calls == 1:
                raise RuntimeError("transient teardown failure")

    class StoppedStream:
        done = True
        worker_stop_state = WorkerStopState.STOPPED
        worker_stopped = True
        remaining_s = 10.0
        proc = SimpleNamespace(stdin=None, returncode=0)
        stdout = ""
        stderr = ""
        metering_stdout = ""
        stdout_chars = 0
        stderr_chars = 0

        async def terminate(self):
            return None

    class StoppedSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            return StoppedStream()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        (workspace / "paper.pdf").write_bytes(b"research")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = StoppedSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            first = asyncio.run(agent.run(ProbeInput(workspace=workspace)))
            assert first.workspace_recovery_notice is None
            assert (workspace / "paper.pdf").read_bytes() == b"research"
            assert agent.workspace_active_guard_path_for(workspace).exists()

            second = asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert second.workspace_recovery_notice is not None
        assert (workspace / "fresh.txt").read_text(encoding="utf-8") == "fresh"
        quarantine_event = next(
            payload
            for kind, payload in events
            if kind == "cli.workspace_quarantined_after_unconfirmed_stop"
        )
        quarantined = Path(quarantine_event["quarantine_path"])
        assert (quarantined / "paper.pdf").read_bytes() == b"research"
        assert any(kind == "cli.teardown_error" for kind, _ in events)


def test_pending_workspace_guard_refuses_automatic_recovery() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(
            guard,
            workspace,
            backend="docker",
            phase="launch_pending",
        )

        with pytest.raises(RuntimeError, match="unresolved external launch"):
            asyncio.run(agent._recover_guarded_workspace(workspace, guard))

        assert retained.read_text(encoding="utf-8") == "research state"
        assert guard.exists()
        agent.events.emit.assert_not_awaited()


def test_settled_subprocess_guard_preserves_workspace_in_place() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(
            guard,
            workspace,
            backend="subprocess",
            phase="launch_settled",
        )

        with pytest.raises(
            RuntimeError,
            match="cannot prove that every detached worker is gone",
        ):
            asyncio.run(agent._recover_guarded_workspace(workspace, guard))

        assert retained.read_text(encoding="utf-8") == "research state"
        assert guard.exists()
        assert not (temp / ".proofcouncil-untrusted-workspaces").exists()
        agent.events.emit.assert_not_awaited()


def test_idle_workspace_guard_is_cleared_without_discarding_workspace() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(
            guard,
            workspace,
            backend="docker",
            phase="idle",
        )

        asyncio.run(agent._recover_guarded_workspace(workspace, guard))

        assert retained.read_text(encoding="utf-8") == "research state"
        assert not guard.exists()
        agent.events.emit.assert_awaited_once()
        assert agent.events.emit.await_args.args[0] == "cli.workspace_idle_guard_cleared"


def test_guarded_workspace_quarantine_rolls_back_if_state_cleanup_fails() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        original_identity = workspace.stat().st_ino
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(guard, workspace, backend="docker")

        with mock.patch.object(
            agent,
            "_clear_workspace_recovery_attempts",
            side_effect=RuntimeError("state cleanup failed"),
        ):
            with pytest.raises(RuntimeError, match="state cleanup failed"):
                asyncio.run(agent._recover_guarded_workspace(workspace, guard))

        assert workspace.stat().st_ino == original_identity
        assert retained.read_text(encoding="utf-8") == "research state"
        assert guard.exists()
        quarantines = temp / ".proofcouncil-untrusted-workspaces"
        assert not list(quarantines.glob("*/workspace"))
        agent.events.emit.assert_not_awaited()


def test_guarded_workspace_quarantine_finishes_before_cancellation_releases_lock() -> None:
    async def exercise(temp: Path) -> None:
        workspace = temp / "compute"
        workspace.mkdir()
        stale = workspace / "unknown-secret.txt"
        stale.write_text("rotated credential", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        guard = agent.workspace_active_guard_path_for(workspace)
        _write_workspace_active_guard(guard, workspace, backend="docker")
        entered = threading.Event()
        release = threading.Event()
        real_quarantine = _quarantine_workspace
        quarantined_paths: list[Path] = []

        def blocked_quarantine(root: Path) -> Path:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release workspace quarantine")
            quarantined = real_quarantine(root)
            quarantined_paths.append(quarantined)
            return quarantined

        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        with mock.patch(
            "proofstack.kinds.cli._quarantine_workspace",
            side_effect=blocked_quarantine,
        ):
            task = asyncio.create_task(
                agent._recover_guarded_workspace(workspace, guard)
            )
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert not stale.exists()
        assert not guard.exists()
        assert len(quarantined_paths) == 1
        assert (quarantined_paths[0] / stale.name).read_text(
            encoding="utf-8"
        ) == "rotated credential"

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_cancellation_during_guard_write_preserves_unstarted_workspace() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.torn_down = False

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def teardown(self, sandbox, inp):
            self.torn_down = True

    class UnstartedSandbox(SimpleNamespace):
        stream_calls = 0

        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            self.stream_calls += 1
            raise AssertionError("stream creation must not be reached")

    async def exercise(temp: Path) -> None:
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        sandbox = UnstartedSandbox(root=workspace)
        agent.events = SimpleNamespace(emit=mock.AsyncMock())
        entered = threading.Event()
        release = threading.Event()

        def blocked_write(path: Path, root: Path, **kwargs) -> None:
            _write_workspace_active_guard(path, root, **kwargs)
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release workspace guard write")

        with (
            mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox),
            mock.patch(
                "proofstack.kinds.cli._write_workspace_active_guard",
                side_effect=blocked_write,
            ),
        ):
            task = asyncio.create_task(agent.run(ProbeInput(workspace=workspace)))
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert retained.read_text(encoding="utf-8") == "research state"
        assert not agent.workspace_active_guard_path_for(workspace).exists()
        assert sandbox.stream_calls == 0
        assert agent.torn_down is True

    with tempfile.TemporaryDirectory() as temp_dir:
        asyncio.run(exercise(Path(temp_dir)))


def test_workspace_guard_remains_when_stream_creation_is_uncertain() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.torn_down = False

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

        async def teardown(self, sandbox, inp):
            self.torn_down = True

    class FailedSpawnSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            raise RuntimeError("Docker control connection failed during spawn")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = FailedSpawnSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(RuntimeError, match="control connection failed"):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert agent.torn_down is False
        assert agent.workspace_active_guard_path_for(workspace).exists()
        assert any(
            kind == "cli.worker_stop_unconfirmed"
            and payload["phase"] == "spawn"
            for kind, payload in events
        )


def test_setup_unknown_worker_is_guarded_before_preflight() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.torn_down = False

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def setup(self, sandbox, inp):
            guard = self.workspace_active_guard_path_for(sandbox.root)
            payload = json.loads(guard.read_text(encoding="utf-8"))
            assert payload["phase"] == "launch_pending"
            await sandbox.run_command(["probe", "--version"])

        async def teardown(self, sandbox, inp):
            self.torn_down = True

    class SetupUnknownSandbox(SimpleNamespace):
        worker_stop_state = WorkerStopState.STOPPED
        worker_launch_settled = True
        stream_calls = 0

        async def ensure_workspace_available(self):
            return None

        async def run_command(self, *args, **kwargs):
            self.worker_stop_state = WorkerStopState.UNKNOWN
            self.worker_launch_settled = False
            return SimpleNamespace(returncode=-9, stdout="", stderr="timeout")

        async def stream_command(self, *args, **kwargs):
            self.stream_calls += 1
            raise AssertionError("main worker must not start")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = SetupUnknownSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="CLI worker could not be confirmed stopped",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        guard = agent.workspace_active_guard_path_for(workspace)
        assert json.loads(guard.read_text(encoding="utf-8"))["phase"] == "launch_pending"
        assert sandbox.stream_calls == 0
        assert agent.torn_down is False
        assert any(
            kind == "cli.worker_stop_unconfirmed"
            and payload["phase"] == "setup"
            for kind, payload in events
        )


def test_confirmed_stream_startup_cleanup_clears_workspace_guard() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.refreshes = 0
            self.torn_down = False

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def refresh_sensitive_state(self, sandbox, inp):
            self.refreshes += 1

        async def teardown(self, sandbox, inp):
            self.torn_down = True

    class StoppedStartupSandbox(SimpleNamespace):
        worker_stop_state = WorkerStopState.STOPPED
        worker_launch_settled = True

        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            self.worker_stop_state = WorkerStopState.STOPPED
            self.worker_launch_settled = True
            raise RuntimeError("launch confirmation failed after cleanup")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = StoppedStartupSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(RuntimeError, match="launch confirmation failed"):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert retained.read_text(encoding="utf-8") == "research state"
        assert agent.refreshes == 1
        assert agent.torn_down is True
        assert not agent.workspace_active_guard_path_for(workspace).exists()
        assert not any(
            kind == "cli.worker_stop_unconfirmed" for kind, _ in events
        )


def test_definitive_spawn_failure_preserves_workspace_and_clears_guard() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["missing-probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def __init__(self, ctx):
            super().__init__(ctx)
            self.torn_down = False

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def teardown(self, sandbox, inp):
            self.torn_down = True

    class FailedSpawnSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            raise SandboxSpawnError("executable not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        retained = workspace / "retained.txt"
        retained.write_text("research state", encoding="utf-8")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        sandbox = FailedSpawnSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(SandboxSpawnError, match="executable not found"):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert retained.read_text(encoding="utf-8") == "research state"
        assert not agent.workspace_active_guard_path_for(workspace).exists()
        assert not agent.sensitive_quarantine_path_for(workspace).exists()
        assert agent.torn_down is True
        assert not any(
            kind == "cli.worker_stop_unconfirmed" for kind, _ in events
        )


def test_worker_stop_event_retries_after_first_emit_failure() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="docker", timeout_s=10)
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

    class UnknownStopStream:
        done = True
        worker_stop_state = WorkerStopState.UNKNOWN
        worker_stopped = False
        proc = SimpleNamespace(stdin=None, returncode=0)
        stdout = ""
        stderr = ""
        metering_stdout = ""
        stdout_chars = 0
        stderr_chars = 0

        async def terminate(self):
            return None

    class UnknownStopSandbox(SimpleNamespace):
        async def ensure_workspace_available(self):
            return None

        async def stream_command(self, *args, **kwargs):
            return UnknownStopStream()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "compute"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        stop_attempts = 0
        successful_stop_events: list[dict] = []

        async def emit(kind, payload, **kwargs):
            nonlocal stop_attempts
            if kind == "cli.worker_stop_unconfirmed":
                stop_attempts += 1
                if stop_attempts == 1:
                    raise RuntimeError("temporary event sink failure")
                successful_stop_events.append(payload)

        agent.events = SimpleNamespace(emit=emit)
        sandbox = UnknownStopSandbox(root=workspace)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(
                RuntimeError,
                match="CLI worker could not be confirmed stopped",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=workspace)))

        assert stop_attempts == 2
        assert [event["phase"] for event in successful_stop_events] == ["finalize"]


def test_soft_pressure_does_not_reset_recovery_attempt_cap() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_SOFT_LIMIT_BYTES = 1
        WORKSPACE_HARD_LIMIT_BYTES = 1024 * 1024 * 1024
        WORKSPACE_HARD_LIMIT_ENTRIES = 100
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

    class SpawnFailureSandbox(SimpleNamespace):
        async def stream_command(self, *args, **kwargs):
            raise RuntimeError("spawn failed")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        (root / "payload.bin").write_bytes(b"payload")
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        claim = agent._claim_workspace_recovery_attempt(root)
        assert claim is not None
        agent._commit_workspace_recovery_claim(claim)

        sandbox = SpawnFailureSandbox(root=root)
        with mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox):
            with pytest.raises(RuntimeError, match="spawn failed"):
                asyncio.run(agent.run(ProbeInput(workspace=root)))

        assert agent._read_workspace_recovery_attempts(root) == 1
        agent._clear_workspace_recovery_attempts(root)


def test_recovery_state_clear_failure_blocks_normal_spawn() -> None:
    class ProbeInput(BaseModel):
        workspace: Path

    class ProbeAgent(CLIAgent):
        CLI_CMD = ["probe"]
        SANDBOX = SandboxSpec(backend="subprocess", timeout_s=10)
        WORKSPACE_SOFT_LIMIT_BYTES = 1024 * 1024
        WORKSPACE_HARD_LIMIT_BYTES = 2 * 1024 * 1024
        WORKSPACE_HARD_LIMIT_ENTRIES = 100
        WORKSPACE_RECOVERY_ENABLED = True

        def sandbox_root_for(self, inp):
            return inp.workspace

        async def collect(self, sandbox, inp, done):
            return inp

    class TrackingSandbox(SimpleNamespace):
        spawned = False

        async def stream_command(self, *args, **kwargs):
            self.spawned = True
            raise AssertionError("worker must not spawn with stale recovery state")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = ProbeAgent(ctx)
        claim = agent._claim_workspace_recovery_attempt(root)
        assert claim is not None
        agent._commit_workspace_recovery_claim(claim)
        sandbox = TrackingSandbox(root=root)

        with (
            mock.patch("proofstack.kinds.cli.make_sandbox", return_value=sandbox),
            mock.patch.object(
                agent,
                "_clear_workspace_recovery_attempts_recorded",
                new=mock.AsyncMock(return_value="OSError: read-only"),
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="persisted workspace recovery state could not be reset",
            ):
                asyncio.run(agent.run(ProbeInput(workspace=root)))

        assert sandbox.spawned is False
        assert agent._read_workspace_recovery_attempts(root) == 1
        agent._clear_workspace_recovery_attempts(root)


def test_workspace_recovery_clear_falls_back_to_zero_state() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        claim = agent._claim_workspace_recovery_attempt(root)
        assert claim is not None
        agent._commit_workspace_recovery_claim(claim)

        with mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            agent._clear_workspace_recovery_attempts(root)

        assert agent._read_workspace_recovery_attempts(root) == 0
        retry_claim = agent._claim_workspace_recovery_attempt(root)
        assert retry_claim is not None
        agent._abort_workspace_recovery_claim(retry_claim)
        agent._clear_workspace_recovery_attempts(root)


def test_workspace_recovery_clear_failure_is_reported() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        claim = agent._claim_workspace_recovery_attempt(root)
        assert claim is not None
        agent._commit_workspace_recovery_claim(claim)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        with (
            mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")),
            mock.patch.object(
                agent,
                "_write_workspace_recovery_attempts",
                side_effect=OSError("read-only"),
            ),
        ):
            error = asyncio.run(
                agent._clear_workspace_recovery_attempts_recorded(
                    root,
                    phase="test",
                )
            )

        assert error is not None
        event = next(
            payload
            for kind, payload in events
            if kind == "cli.workspace_recovery_state_clear_failed"
        )
        assert event["phase"] == "test"
        agent._clear_workspace_recovery_attempts(root)


def test_workspace_recovery_attempt_cap_resets_for_recreated_workspace() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "compute"
        root.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        over = {
            "bytes": 110,
            "allocated_bytes": 110,
            "allocated_bytes_supported": True,
            "entries": 11,
            "errors": 0,
            "filesystem_free_bytes": 1_000_000,
            "filesystem_free_inodes": 1_000_000,
        }

        first = Compute(ctx)
        first._configure_workspace_recovery(over)
        first_claim = first._claim_workspace_recovery_attempt(root)
        assert first_claim is not None
        first._commit_workspace_recovery_claim(first_claim)
        old_identity = first._workspace_identity(root)
        root.rename(temp / "compute-old")
        root.mkdir()
        assert first._workspace_identity(root) != old_identity

        first._configure_workspace_recovery(over)
        recreated_claim = first._claim_workspace_recovery_attempt(root)
        assert recreated_claim is not None
        first._commit_workspace_recovery_claim(recreated_claim)
        first._clear_workspace_recovery_attempts(root)


def test_compute_clean_exit_without_finish_is_salvaged_as_partial() -> None:
    class FakeStream:
        done = True
        proc = SimpleNamespace(returncode=0)
        stdout = ""
        stderr = ""

        async def terminate(self):
            return None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        done = asyncio.run(
            agent._wait_for_done(
                FakeStream(),
                temp / "missing-done.json",
                spawn_call_id="call",
            )
        )

    assert done.status == "partial"
    assert "salvaged artifacts from a clean CLI exit" in done.summary
    exit_event = next(payload for kind, payload in events if kind == "cli.exit")
    assert exit_event["status"] == "partial"


def test_filesystem_free_inode_guard_stops_compute_worker() -> None:
    from unittest import mock

    class FakeStream:
        terminated = False

        @property
        def worker_stopped(self):
            return self.terminated

        async def terminate(self):
            self.terminated = True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace = temp / "workspace"
        workspace.mkdir()
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        agent.WORKSPACE_HARD_LIMIT_BYTES = 1_000
        agent.WORKSPACE_HARD_LIMIT_ENTRIES = 1_000
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_MIN_FREE_INODES = 100
        agent.WORKSPACE_CHECK_INTERVAL_S = 0
        events: list[tuple[str, dict]] = []

        async def emit(kind, payload, **kwargs):
            events.append((kind, payload))

        agent.events = SimpleNamespace(emit=emit)
        stream = FakeStream()
        fake_usage = {
            "bytes": 0,
            "files": 0,
            "entries": 0,
            "errors": 0,
            "filesystem_free_bytes": 1_000,
            "filesystem_free_inodes": 99,
        }
        with mock.patch(
            "proofstack.kinds.cli.measure_workspace_usage",
            return_value=fake_usage,
        ):
            done = asyncio.run(
                agent._wait_for_done(
                    stream,
                    temp / "missing-done.json",
                    spawn_call_id="call",
                    workspace_root=workspace,
                )
            )

        assert done.status == "error"
        assert "filesystem free inodes" in done.summary
        event = next(
            payload for kind, payload in events if kind == "cli.workspace_limit_exceeded"
        )
        assert event["reason"] == "filesystem_min_free_inodes"


def test_compute_handoff_scan_has_workspace_entry_cap() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        root = temp / "workspace"
        root.mkdir()
        for idx in range(3):
            (root / f"file-{idx}.txt").write_text("x", encoding="utf-8")

        try:
            _zip_workspace(
                root,
                temp / "handoff.zip",
                exclude_top=_ZIP_EXCLUDE_TOP,
                max_workspace_entries=2,
            )
        except ValueError as e:
            assert "workspace entry count exceeded" in str(e)
        else:
            raise AssertionError("unbounded handoff scan was accepted")


def test_compute_handoff_default_limits_leave_provider_margin() -> None:
    assert COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES == 200 * 1024 * 1024
    assert COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES < 512 * 1024 * 1024
    assert COMPUTE_HANDOFF_MAX_MEMBER_BYTES == 50 * 1024 * 1024
    assert COMPUTE_FILESYSTEM_MIN_FREE_BYTES == 0
    assert COMPUTE_FILESYSTEM_MIN_FREE_INODES == 100_000
    assert COMPUTE_FILESYSTEM_RESERVATION_BYTES == 0


def test_bounded_stream_buffer_retains_complete_line_tail() -> None:
    buffer = _BoundedTextBuffer(6)
    buffer.append("abc\n")
    buffer.append("def\n")

    assert buffer.text() == "def\n"
    assert buffer.retained_chars == 4
    assert buffer.dropped_chars == 4


def test_bounded_stream_buffer_drops_partial_secret_line() -> None:
    buffer = _BoundedTextBuffer(8)
    buffer.append("prefix-secret")
    buffer.append("still-secret\nsafe\n")

    assert buffer.text() == "safe\n"
    assert "secret" not in buffer.text()
    assert buffer.retained_chars + buffer.dropped_chars == len(
        "prefix-secretstill-secret\nsafe\n"
    )


def test_bounded_stream_keeps_compact_usage_records() -> None:
    capture = _JsonUsageCapture(max_chars=4096, max_line_chars=4096)
    codex_event = json.dumps(
        {
            "type": "turn.completed",
            "transcript": "content that must not be retained",
            "usage": {
                "input_tokens": 11,
                "cached_input_tokens": 3,
                "output_tokens": 5,
            },
        }
    )
    claude_event = json.dumps(
        {
            "type": "result",
            "result": "another transcript",
            "num_turns": 2,
            "total_cost_usd": 1.25,
            "usage": {"input_tokens": 7, "output_tokens": 4},
        }
    )
    stream = codex_event + "\n" + claude_event + "\n"
    capture.feed(stream[:17])
    capture.feed(stream[17:])
    capture.finish()

    retained = capture.text()
    assert "content that must not be retained" not in retained
    codex_usage = parse_codex_jsonl(retained)
    assert codex_usage.input_tokens == 11
    assert codex_usage.output_tokens == 5
    claude_usage = parse_claude_json(retained)
    assert claude_usage.input_tokens == 7
    assert claude_usage.output_tokens == 4
    assert claude_usage.total_cost_usd == 1.25


def test_truncated_stream_meters_usage_from_before_retained_tail() -> None:
    async def exercise():
        reader = asyncio.StreamReader()
        proc = SimpleNamespace(stdout=reader, stderr=None)
        stream = _StreamingProcess(
            proc=proc,
            cmd=["fake"],
            deadline=1_000_000_000.0,
            max_capture_chars=80,
        )
        first = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )
        second = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }
        )
        emitted = first + "\n" + ("x" * 200) + "\n" + second + "\n"
        reader.feed_data(emitted.encode("utf-8"))
        reader.feed_eof()
        await asyncio.gather(stream._stdout_task, stream._stderr_task)
        return stream, emitted

    stream, emitted = asyncio.run(exercise())
    assert stream.stdout_dropped_chars > 0
    assert stream.stdout_chars == len(emitted)
    usage = parse_codex_jsonl(stream.metering_stdout)
    assert usage.n_turns == 2
    assert usage.input_tokens == 7
    assert usage.output_tokens == 10


def test_compute_refreshes_rotated_transient_codex_secrets() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        codex_home = temp / "codex-home"
        codex_home.mkdir()
        rotated = "rotated-access-token-that-must-be-redacted"
        (codex_home / "auth.json").write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "tokens": {"access_token": rotated},
                }
            ),
            encoding="utf-8",
        )
        agent._codex_home_host = codex_home
        agent._copied_codex_auth = True
        agent._subscription_codex_auth = True
        agent.events = SimpleNamespace(emit=lambda *args, **kwargs: None)

        asyncio.run(
            agent.refresh_sensitive_state(
                SimpleNamespace(root=temp / "compute"),
                Compute.Inputs(
                    problem="P",
                    problem_id="p",
                    round=1,
                    instructions="compute",
                    compute_workspace=temp / "compute",
                ),
            )
        )

        assert rotated in agent._handoff_secrets
        assert rotated not in agent.sanitize_cli_output(f"output {rotated}")


def test_compute_refresh_fails_closed_when_transient_auth_disappears() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
        agent = Compute(ctx)
        codex_home = temp / "codex-home"
        codex_home.mkdir()
        agent._codex_home_host = codex_home
        agent._copied_codex_auth = True
        agent._subscription_codex_auth = True

        try:
            asyncio.run(
                agent.refresh_sensitive_state(
                    SimpleNamespace(root=temp / "compute"),
                    Compute.Inputs(
                        problem="P",
                        problem_id="p",
                        round=1,
                        instructions="compute",
                        compute_workspace=temp / "compute",
                    ),
                )
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing prepared auth was accepted")


def test_sensitive_quarantine_uses_external_marker_when_workspace_write_fails() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        workspace_marker = temp / "workspace" / "runtime" / "quarantine"
        external_marker = temp / "control" / "quarantine"

        def write_marker(path: Path) -> bool:
            if path == workspace_marker:
                return False
            return _write_sensitive_quarantine_marker(path)

        with mock.patch(
            "proofstack.kinds.cli._write_sensitive_quarantine_marker",
            side_effect=write_marker,
        ):
            persisted = _mark_sensitive_workspace_untrusted(
                workspace_marker,
                external_marker,
            )

        assert persisted == (external_marker,)
        assert not workspace_marker.exists()
        assert external_marker.exists()


def test_compute_suppresses_artifacts_when_final_auth_is_unreadable() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        fake_codex = fake_bin / "codex"
        rotated = "rotated-token-that-must-never-leave-the-workspace"
        fake_codex.write_text(
            f"""#!/bin/sh
if [ "${{1:-}}" = "--version" ]; then
  printf 'codex-cli 0.144.0\n'
  exit 0
fi
if [ "${{1:-}}" = "login" ] && [ "${{2:-}}" = "status" ]; then
  exit 0
fi
cat >/dev/null
mkdir -p responses
printf '%s\n' 'result {rotated}' > responses/response_round_1.md
printf '%s\n' 'transcript {rotated}'
rm -f "$CODEX_HOME/auth.json"
"$FINISH_BIN" '{{"status":"done","summary":"finished"}}'
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        fake_home = _fake_subscription_home(temp)
        compute_workspace = temp / "compute"

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(
                run_id="test", root_workdir=temp / "run", flat=True
            )
            agent = Compute(ctx)
            with (
                mock.patch(
                    "proofstack.agents.ac.compute.Path.home",
                    return_value=fake_home,
                ),
                mock.patch(
                    "proofstack.sandbox.subprocess._terminate_marked_processes",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                try:
                    asyncio.run(
                        agent(
                            problem="P",
                            problem_id="prob-001",
                            round=1,
                            instructions="do the computation",
                            compute_workspace=compute_workspace,
                            sandbox_backend="subprocess",
                            codex_sandbox="workspace-write",
                            filesystem_min_free_bytes=0,
                            filesystem_min_free_inodes=0,
                        )
                    )
                except RuntimeError as e:
                    assert "artifacts were suppressed" in str(e)
                else:
                    raise AssertionError("unreadable final auth was accepted")
        finally:
            os.environ["PATH"] = old_path

        assert rotated in (
            compute_workspace / "responses" / "response_round_1.md"
        ).read_text(encoding="utf-8")
        workspace_marker = (
            compute_workspace
            / ".pwc"
            / "runtime"
            / "SENSITIVE_STATE_UNTRUSTED"
        )
        assert workspace_marker.exists()
        external_marker = agent.sensitive_quarantine_path_for(compute_workspace)
        assert external_marker.exists()
        assert not list((temp / "run").rglob("cli_stdout.log"))
        assert not list((temp / "run").rglob("cli_stderr.log"))
        assert not list((temp / "run").rglob("compute_workspace_round_*.zip"))

        # The sidecar must still block reuse if the model-writable marker is
        # accidentally deleted between invocations.
        workspace_marker.unlink()
        second_agent = Compute(ctx)
        with mock.patch(
            "proofstack.agents.ac.compute.Path.home",
            return_value=fake_home,
        ), pytest.raises(RuntimeError, match="workspace is quarantined"):
            asyncio.run(
                second_agent(
                    problem="P",
                    problem_id="prob-001",
                    round=2,
                    instructions="do the computation",
                    compute_workspace=compute_workspace,
                    sandbox_backend="subprocess",
                    codex_sandbox="workspace-write",
                    filesystem_min_free_bytes=0,
                    filesystem_min_free_inodes=0,
                )
            )


def test_compute_run_captures_codex_last_message_with_finish_signal() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        fake_codex = fake_bin / "codex"
        fake_codex.write_text(
            """#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  printf 'codex-cli 0.144.0\n'
  exit 0
fi
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-last-message" ]; then
    shift
    out="$1"
  fi
  shift || true
done
cat >/dev/null
mkdir -p "$(dirname "$out")"
printf 'fake codex final message\\n' > "$out"
"$FINISH_BIN" '{"status":"done","summary":"fake complete"}'
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        fake_home = _fake_subscription_home(temp)

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            with (
                mock.patch(
                    "proofstack.agents.ac.compute.Path.home",
                    return_value=fake_home,
                ),
                mock.patch(
                    "proofstack.sandbox.subprocess._terminate_marked_processes",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                out = asyncio.run(
                    Compute(ctx)(
                        problem="P",
                        problem_id="prob-001",
                        round=1,
                        instructions="do the computation",
                        compute_workspace=temp / "compute",
                        sandbox_backend="subprocess",
                        codex_sandbox="workspace-write",
                        filesystem_min_free_bytes=0,
                        filesystem_min_free_inodes=0,
                    )
                )
        finally:
            os.environ["PATH"] = old_path

        assert out.status == "done"
        assert out.response_md == "fake codex final message\n"
        assert out.zip_path is not None
        assert Path(out.zip_path).exists()


def test_compute_finish_survives_nested_login_shell_path_reset() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        fake_codex = fake_bin / "codex"
        fake_codex.write_text(
            """#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  printf 'codex-cli 0.144.0\n'
  exit 0
fi
cat >/dev/null
printf '{"status":"done","summary":"via login shell"}' > "$HOME/finish-body.json"
/usr/bin/env -i HOME="$HOME" FINISH_DONE_PATH="$FINISH_DONE_PATH" /bin/bash -lc 'finish "$HOME/finish-body.json"'
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        fake_home = _fake_subscription_home(temp)

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            with (
                mock.patch(
                    "proofstack.agents.ac.compute.Path.home",
                    return_value=fake_home,
                ),
                mock.patch(
                    "proofstack.sandbox.subprocess._terminate_marked_processes",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                out = asyncio.run(
                    Compute(ctx)(
                        problem="P",
                        problem_id="prob-001",
                        round=1,
                        instructions="do the computation",
                        compute_workspace=temp / "compute",
                        sandbox_backend="subprocess",
                        codex_sandbox="workspace-write",
                        filesystem_min_free_bytes=0,
                        filesystem_min_free_inodes=0,
                    )
                )
        finally:
            os.environ["PATH"] = old_path

        assert out.status == "done"
        assert out.summary == "via login shell"
        assert out.response_md.endswith("via login shell")


def test_compute_setup_removes_stale_codex_last_message() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        fake_codex = fake_bin / "codex"
        fake_codex.write_text(
            """#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  printf 'codex-cli 0.144.0\n'
  exit 0
fi
cat >/dev/null
"$FINISH_BIN" '{"status":"done","summary":"no current result"}'
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        fake_home = _fake_subscription_home(temp)

        compute_root = temp / "compute"
        stale_path = compute_root / _CODEX_LAST_MESSAGE_REL
        stale_path.parent.mkdir(parents=True)
        stale_path.write_text("stale previous-round result", encoding="utf-8")

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            with (
                mock.patch(
                    "proofstack.agents.ac.compute.Path.home",
                    return_value=fake_home,
                ),
                mock.patch(
                    "proofstack.sandbox.subprocess._terminate_marked_processes",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                out = asyncio.run(
                    Compute(ctx)(
                        problem="P",
                        problem_id="prob-001",
                        round=2,
                        instructions="do the computation",
                        compute_workspace=compute_root,
                        sandbox_backend="subprocess",
                        codex_sandbox="workspace-write",
                        filesystem_min_free_bytes=0,
                        filesystem_min_free_inodes=0,
                    )
                )
        finally:
            os.environ["PATH"] = old_path

        assert out.status == "done"
        assert "stale previous-round result" not in out.response_md
        assert out.response_md == (
            "(Worker did not write responses/response_round_2.md; "
            "falling back to finish summary)\n\n"
            "no current result"
        )
        assert not stale_path.exists()
