"""Sandbox interface for CLI agents."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping


SandboxBackend = Literal["subprocess", "docker"]


# Standard env vars we always pass through; provider keys are added per call.
# HOME is intentionally NOT in the allowlist. ``build_env`` pins it to the
# sandbox root by default so a model-driven coding CLI cannot read or modify
# files under the orchestrator's real home directory unless a component
# explicitly overrides HOME through env_extra.
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "PYTHONPATH",
)


@dataclass
class SandboxSpec:
    """Resource and isolation knobs for a sandbox invocation."""

    cpu_limit: int = 2
    memory_gb: int = 4
    timeout_s: int = 900
    env_allowlist: tuple[str, ...] = DEFAULT_ENV_ALLOWLIST
    extra_env: Mapping[str, str] = field(default_factory=dict)
    # Provider-key env names to pass through *if present* in the parent
    # environment. The actual values are read from the parent at spawn
    # time and never written to disk.
    provider_keys: tuple[str, ...] = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GLM_API_KEY",
        "MOONSHOT_API_KEY",
        "TOGETHER_API_KEY",
        "STEPFUN_API_KEY",
    )

    # --- Backend selection -------------------------------------------------
    # "subprocess" — spawn the CLI as a child process on the host, with
    # setrlimit + env-allowlist isolation only. Useful inside trusted
    # containers or for tests.
    #
    # "docker"     — spawn the CLI inside an ephemeral `docker run --rm`
    # container with dropped capabilities, resource limits, and a
    # writable bind mount for the per-invocation workdir. This is the
    # default for CLI agents.
    backend: SandboxBackend = "docker"

    # --- Docker-backend-specific knobs -------------------------------------
    # Image to run. Must exist locally (we don't pull automatically).
    # Build with `docker build -t proofstack-sandbox:latest deploy/sandbox/`.
    docker_image: str = "proofstack-sandbox:latest"
    # "bridge" gives internet access (Codex needs OpenAI); "none" for
    # fully-offline smoke tests.
    docker_network: Literal["bridge", "none"] = "bridge"
    # Hard per-container pid cap. Prevents fork bombs.
    docker_pids_limit: int = 256
    # Some Docker/AppArmor installations reject every exec inside
    # node-based images when no-new-privileges is set. Keep it configurable
    # so Codex sandboxes can still run while retaining the other Docker caps.
    docker_no_new_privileges: bool = True
    # Extra flags passed verbatim to `docker run` (after the safety
    # defaults). For advanced tuning; keep empty by default.
    docker_extra_args: tuple[str, ...] = ()

    def build_env(self, *, sandbox_root: Path, extra_path: Iterable[Path] = ()) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in self.env_allowlist:
            if (val := os.environ.get(key)) is not None:
                env[key] = val
        for key in self.provider_keys:
            if (val := os.environ.get(key)) is not None:
                env[key] = val
        env.update(self.extra_env)
        # Prepend extra_path entries to PATH so our finish shim wins.
        path_parts = [str(p) for p in extra_path] + ([env["PATH"]] if "PATH" in env else [])
        if path_parts:
            env["PATH"] = os.pathsep.join(path_parts)
        # HOME and TMPDIR are pinned to the sandbox root at the base layer.
        # Call-site env_extra is merged later by sandbox backends and may
        # intentionally override HOME for trusted subscription-CLI workflows.
        env["HOME"] = str(sandbox_root)
        env["TMPDIR"] = str(sandbox_root)
        return env


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


class Sandbox:
    """Per-invocation isolated workdir for a CLI run.

    Subclasses (only ``SubprocessSandbox`` for now) provide the actual
    execution. The interface is async to fit the rest of the stack.
    """

    def __init__(self, spec: SandboxSpec, *, root: Path | None = None) -> None:
        self.spec = spec
        # Resolve to absolute at construction. Several env vars derived
        # from ``self.root`` (HOME, TMPDIR, CODEX_HOME, FINISH_DONE_PATH)
        # get exported into a child whose own cwd is the resolved
        # absolute path; if those env vars were relative, the child
        # would re-resolve them against its (already absolute) cwd
        # and look for a doubled-up path that does not exist. Codex
        # in particular bails with "Error finding codex home" in <1s.
        raw_root = root if root is not None else Path(tempfile.mkdtemp(prefix="proofstack-sbx-"))
        self.root = Path(raw_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._closed = False

    async def write_file(self, relpath: str, content: str | bytes) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return path

    async def read_file(self, relpath: str) -> str:
        return (self.root / relpath).read_text(encoding="utf-8")

    async def run_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
    ) -> CommandResult:
        raise NotImplementedError

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(shutil.rmtree, self.root, True)

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()


__all__ = [
    "CommandResult",
    "DEFAULT_ENV_ALLOWLIST",
    "Sandbox",
    "SandboxSpec",
]
