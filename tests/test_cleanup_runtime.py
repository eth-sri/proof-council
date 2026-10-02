from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import pytest

from proofstack import cleanup_runtime as mod


@pytest.fixture
def probes(monkeypatch):
    seen = []
    replies = {
        ("claude", "--version"): "2.1.251 (Claude Code)",
        ("claude", "--help"): "\n".join(mod.CLAUDE_FLAGS),
        ("codex", "--version"): "codex-cli 0.154.0",
        ("codex", "exec", "--help"): "\n".join(mod.CODEX_FLAGS),
    }

    def run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(stdout=replies[tuple(command)], stderr="")

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-pass")
    return seen, replies


def test_version_and_options_checked_without_credentials(probes):
    seen, _ = probes
    result = mod.check_cleanup_runtime(claude_version="2.1.251", codex_version="0.154.0")
    assert result["claude"]["version"] == "2.1.251"
    assert len(seen) == 4
    for _, kwargs in seen:
        assert "API_KEY" not in repr(kwargs["env"])
        assert kwargs["env"]["DISABLE_AUTOUPDATER"] == "1"
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 20
        assert not Path(kwargs["cwd"]).exists()


@pytest.mark.parametrize("key,value,match", [
    (("claude", "--version"), "2.1.154 (Claude Code)", "requires claude"),
    (("claude", "--version"), "2.1.217 (Claude Code)", "requires claude"),
    (("claude", "--version"), "unknown", "cannot determine"),
    (("claude", "--help"), "--max-budget-usd-other", "missing required"),
    (("codex", "exec", "--help"), "--sandbox", "missing required"),
])
def test_incompatible_cli_fails_before_model_calls(probes, key, value, match):
    _, replies = probes
    replies[key] = value
    with pytest.raises(RuntimeError, match=match):
        mod.check_cleanup_runtime()


def test_build_rejects_unexpected_version_and_codex_is_optional(probes):
    seen, _ = probes
    with pytest.raises(RuntimeError, match="expected claude"):
        mod.check_cleanup_runtime(claude_version="2.1.218")
    seen.clear()
    mod.check_cleanup_runtime(require_codex=False)
    assert len(seen) == 2


def test_native_installer_does_not_replace_binary_on_bad_checksum(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/install_claude_native.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "uname": 'if [ "$1" = -s ]; then echo Linux; else echo x86_64; fi',
        "curl": 'while [ "$1" != --output ]; do shift; done; printf bad > "$2"',
    }.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    destination = tmp_path / "claude"
    destination.write_text("previous verified binary")
    result = subprocess.run(["sh", str(script), "2.1.251", str(destination)],
                            env={"PATH": f"{bin_dir}:{os.environ['PATH']}"}, capture_output=True)
    assert result.returncode != 0
    assert b"not found" not in result.stderr
    assert destination.read_text() == "previous verified binary"


def test_native_installer_requires_reviewed_pin(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/install_claude_native.sh"
    result = subprocess.run(["sh", str(script), "9.9.9", str(tmp_path / "claude")], capture_output=True)
    assert result.returncode != 0
    assert b"review and pin" in result.stderr
