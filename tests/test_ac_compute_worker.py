from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
)
from proofstack.context import RunContext  # noqa: E402
from proofstack.kinds.cli import CLIDoneRecord, measure_workspace_usage  # noqa: E402
from proofstack.sandbox.base import SandboxSpec  # noqa: E402
from proofstack.sandbox.subprocess import SubprocessSandbox  # noqa: E402


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
    assert inp.filesystem_min_free_inodes == COMPUTE_FILESYSTEM_MIN_FREE_INODES
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
        assert "workspace size" in done.summary
        assert stream.terminated is True
        assert any(kind == "cli.workspace_limit_exceeded" for kind, _ in events)


def test_filesystem_free_space_guard_stops_compute_worker() -> None:
    from unittest import mock

    class FakeStream:
        terminated = False

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


def test_filesystem_free_inode_guard_stops_compute_worker() -> None:
    from unittest import mock

    class FakeStream:
        terminated = False

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


def test_compute_run_captures_codex_last_message_on_clean_cli_exit() -> None:
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
            with mock.patch(
                "proofstack.agents.ac.compute.Path.home",
                return_value=fake_home,
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
            with mock.patch(
                "proofstack.agents.ac.compute.Path.home",
                return_value=fake_home,
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
            with mock.patch(
                "proofstack.agents.ac.compute.Path.home",
                return_value=fake_home,
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
            "(no done.json written)"
        )
        assert not stale_path.exists()
