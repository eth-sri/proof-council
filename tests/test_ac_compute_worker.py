from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.compute import (  # noqa: E402
    _CODEX_LAST_MESSAGE_REL,
    _ZIP_EXCLUDE_TOP,
    _build_codex_cmd,
    _zip_workspace,
    Compute,
    DEFAULT_COST_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
)
from proofstack.context import RunContext  # noqa: E402
from proofstack.kinds.cli import CLIDoneRecord  # noqa: E402
from proofstack.sandbox.base import SandboxSpec  # noqa: E402


class FakeSandbox(SimpleNamespace):
    async def write_file(self, relpath: str, content: str) -> Path:
        path = Path(self.root) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


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


def test_dockerfile_pins_and_smokes_codex_cli() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    pwc_text = (ROOT / "deploy" / "sandbox" / "Dockerfile.pwc").read_text(
        encoding="utf-8"
    )

    assert "@openai/codex@${OPENAI_CODEX_VERSION}" in text
    assert "OPENAI_CODEX_VERSION=" in text
    assert "codex exec --help | grep -q -- '--output-last-message'" in text
    assert "gmpy2 python-flint z3-solver cvxpy" in text
    assert "git file column time" in text
    assert "> /usr/local/bin/finish" in text
    assert "FINISH_DONE_PATH" in text
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


def test_compute_always_uses_scrubbed_codex_home() -> None:
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
        assert (root / "compute_utils.py").exists()
        assert (root / "sitecustomize.py").exists()

        (codex_home / "session.json").write_text("{}", encoding="utf-8")
        asyncio.run(agent.teardown(sandbox, inp))
        assert not codex_home.exists()


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

        import zipfile

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())

        assert "notes/result.txt" in names
        assert not any(name.startswith(".codex/") for name in names)
        assert not any(name.startswith(".codex-home/") for name in names)
        assert ".bash_profile" not in names


def test_compute_run_captures_codex_last_message_on_clean_cli_exit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        fake_codex = fake_bin / "codex"
        fake_codex.write_text(
            """#!/bin/sh
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

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            out = asyncio.run(
                Compute(ctx)(
                    problem="P",
                    problem_id="prob-001",
                    round=1,
                    instructions="do the computation",
                    compute_workspace=temp / "compute",
                    sandbox_backend="subprocess",
                    codex_sandbox="workspace-write",
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
cat >/dev/null
printf '{"status":"done","summary":"via login shell"}' > "$HOME/finish-body.json"
/usr/bin/env -i HOME="$HOME" FINISH_DONE_PATH="$FINISH_DONE_PATH" /bin/bash -lc 'finish "$HOME/finish-body.json"'
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            out = asyncio.run(
                Compute(ctx)(
                    problem="P",
                    problem_id="prob-001",
                    round=1,
                    instructions="do the computation",
                    compute_workspace=temp / "compute",
                    sandbox_backend="subprocess",
                    codex_sandbox="workspace-write",
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
cat >/dev/null
exit 0
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)

        compute_root = temp / "compute"
        stale_path = compute_root / _CODEX_LAST_MESSAGE_REL
        stale_path.parent.mkdir(parents=True)
        stale_path.write_text("stale previous-round result", encoding="utf-8")

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        try:
            ctx = RunContext.create(run_id="test", root_workdir=temp / "run", flat=True)
            out = asyncio.run(
                Compute(ctx)(
                    problem="P",
                    problem_id="prob-001",
                    round=2,
                    instructions="do the computation",
                    compute_workspace=compute_root,
                    sandbox_backend="subprocess",
                    codex_sandbox="workspace-write",
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
