"""Compute — out-of-band codex CLI worker for the Author/Critic loop.

Invoked when the Author emits a ``<compute_agent>...</compute_agent>``
block. Fans out in parallel with the Critic (and Council, if also
requested). The reply lands in the *next* round's Author prompt — the
Author's API call has already returned by the time the worker spins
up, so its results cannot be folded into the same round.

Workspace shape (``ac_workspaces/<pid>/compute/``)::

    problem_documents_readonly/        # resynced every invocation
      problem.txt, answer.tex,
      research_notes.tex, references.bib
    responses/
      response_round_{N}.md            # worker's reply for round N
    code/  data/  papers/  notes/      # worker-owned, persistent
    ../.compute_codex_home/<id>/       # transient codex auth (scrubbed)
    .pwc/runtime/                      # framework state (done.json, WRAP_UP)

The worker is told never to write to ``problem_documents_readonly/``.
After it finishes, the workflow:
  1. reads ``responses/response_round_{N}.md`` to render the next
     Author prompt fragment;
  2. creates a bounded handoff archive from the persistent workspace
     (excluding readonly, credentials, framework state, and transient
     build files) for attachment to the next Author container call.

Same docker sandbox / soft-timeout / codex-jsonl usage accounting as
the PWC Worker. Model, effort, cost config, and timeout defaults can be
overridden per call through ``Inputs``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, Field, model_validator

from proofstack.cli_usage import (
    cost_for_codex_usage,
    load_cost_rates,
    parse_codex_jsonl,
)
from proofstack.codex_auth import (
    classify_codex_auth,
    extract_codex_auth_secrets,
    redact_codex_secrets,
)
from proofstack.kinds.cli import CLIAgent, CLIDoneRecord, measure_workspace_usage
from proofstack.sandbox import resolve_backend
from proofstack.sandbox.base import Sandbox, SandboxSpec


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_COST_CONFIG = "models/openai/gpt-56-sol-pro"
DEFAULT_SOFT_TIMEOUT_S = 7200
DEFAULT_HARD_TIMEOUT_S = 9000
DEFAULT_SANDBOX_BACKEND = "docker"
DEFAULT_DOCKER_IMAGE = "proofstack-pwc-sandbox:latest"
# ``auto`` resolves to ``--dangerously-bypass-approvals-and-sandbox``
# under docker and ``--sandbox workspace-write`` under subprocess.
DEFAULT_CODEX_SANDBOX = "auto"
MIN_CODEX_VERSION: Final[tuple[int, int, int]] = (0, 144, 0)
CODEX_AUTH_TIMEOUT_S: Final[int] = 30
COMPUTE_HANDOFF_MAX_FILES: Final[int] = 700
COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES: Final[int] = 200 * 1024 * 1024
COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES: Final[int] = 180 * 1024 * 1024
COMPUTE_HANDOFF_MAX_MEMBER_BYTES: Final[int] = 50 * 1024 * 1024
COMPUTE_HANDOFF_MANIFEST: Final[str] = "PROOFCOUNCIL_HANDOFF.txt"
COMPUTE_HANDOFF_LOCAL_ONLY_SUFFIX: Final[str] = ".local-only.json"
COMPUTE_WORKSPACE_SOFT_LIMIT_BYTES: Final[int] = 80 * 1024 * 1024 * 1024
COMPUTE_WORKSPACE_HARD_LIMIT_BYTES: Final[int] = 100 * 1024 * 1024 * 1024
COMPUTE_WORKSPACE_SOFT_LIMIT_ENTRIES: Final[int] = 80_000
COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES: Final[int] = 100_000
# Host capacity varies widely, so filesystem-wide headroom is an explicit run
# policy. Serious parallel runs should set both values on ACWorkflow.Inputs.
COMPUTE_FILESYSTEM_MIN_FREE_BYTES: Final[int] = 0
COMPUTE_FILESYSTEM_MIN_FREE_INODES: Final[int] = 100_000
COMPUTE_FILESYSTEM_RESERVATION_BYTES: Final[int] = 0
_COMPUTE_HANDOFF_LOCAL_ONLY_XATTR: Final[str] = "user.proofcouncil.local_only"

# Directories to exclude from the workspace zip handed back to the Author.
# - problem_documents_readonly/ is already what the Author has.
# - .codex-home/ is excluded for legacy runs; current CODEX_HOME is
#   outside this workspace and scrubbed in teardown.
# - .codex/ is excluded defensively in case a future Codex ignores CODEX_HOME.
# - .pwc/ is framework state (done.json, WRAP_UP sentinel, runtime bits).
# - shell startup files are framework shims so nested login shells can
#   still find ``finish``; they are not useful to the Author.
_ZIP_EXCLUDE_TOP = {
    "problem_documents_readonly",
    ".codex-home",
    ".codex",
    ".pwc",
    ".bash_profile",
    ".profile",
    ".bashrc",
    "scratch",
    "archive",
}
_ZIP_EXCLUDE_TOP_DIRS: Final[frozenset[str]] = frozenset({".local"})
_ZIP_EXCLUDE_DIRECTORY_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".cache",
        ".npm",
        ".sage",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".venv",
        "venv",
        "node_modules",
    }
)
_ZIP_EXCLUDE_DIRECTORY_PREFIXES: Final[tuple[str, ...]] = ("gaptempdir",)
_HANDOFF_LOCAL_ONLY_SUFFIXES: Final[tuple[str, ...]] = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".h5",
    ".hdf5",
    ".parquet",
    ".feather",
    ".arrow",
)
_HANDOFF_BULK_SUFFIX_MAX_BYTES: Final[dict[str, int]] = {
    ".csv": 5 * 1024 * 1024,
    ".tsv": 5 * 1024 * 1024,
    ".jsonl": 5 * 1024 * 1024,
    ".log": 5 * 1024 * 1024,
    ".out": 5 * 1024 * 1024,
    ".g": 5 * 1024 * 1024,
}
_CODEX_LAST_MESSAGE_REL: Final[str] = ".pwc/runtime/codex-last-message.md"
_DOCKER_CODEX_HOME: Final[str] = "/codex-home"
_COMPUTE_UTILS = """\
import json
from pathlib import Path


def safe_json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_json_dumps(obj, **kwargs):
    return json.dumps(obj, default=safe_json_default, **kwargs)


def safe_json_dump(obj, path, **kwargs):
    Path(path).write_text(safe_json_dumps(obj, **kwargs), encoding="utf-8")
"""
_SITECUSTOMIZE = """\
try:
    import numpy as np

    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
except Exception:
    pass
"""


COMPUTE_WORKER_PROMPT = """\
You are the Compute Worker for an Author/Critic mathematical research
loop. The Author has commissioned you for a focused computation,
code-development task, or deeper literature retrieval that is too
heavy or too slow for its own in-call code_interpreter sandbox.

This is round {round} of the loop.

## Workspace layout

Your sandbox is a persistent workspace shared across invocations
within this run; files you create now are visible to the next round's
worker call.

- ``problem_documents_readonly/`` — snapshot of the Author's current
  files. Refreshed at the start of every invocation. **Read-only.**
  Do not write here; the directory will be wiped and re-synced next
  round.
    * ``problem.txt``
    * ``answer.tex``
    * ``research_notes.tex``
    * ``references.bib``

- ``responses/`` — your reply to the Author. You **must** write
  ``responses/response_round_{round}.md`` before finishing. The Author
  will see this file's contents pasted verbatim into its next-turn
  prompt. Keep it focused (under ~4000 words). Include concrete
  outputs: tables of numbers, code excerpts, file pointers under your
  workspace, citations of papers you found. A bounded selection of the
  workspace is attached to the next Author call, so put every essential
  conclusion in this response and keep important artifacts in
  ``code/``, ``data/``, ``papers/``, or ``notes/``. Runtime caches and
  excess files are intentionally omitted from the handoff. In particular,
  ``.cache/``, ``.npm/``, ``.sage/``, virtual environments, Python caches,
  and the top-level ``.local/`` directory remain available to later worker
  rounds but are never attached to the Author.

- Everything else (``code/``, ``data/``, ``papers/``, ``notes/``, …)
  is yours to organize freely. Files persist across invocations, so
  later rounds can build on prior compute artifacts.

- ``scratch/`` — temporary CAS databases, expanded archives, failed
  attempts, checkpoints, and other reproducible intermediates. This
  directory is excluded from the Author handoff and deleted after the
  invocation. Promote only compact, necessary certificates into
  ``data/`` or ``notes/``. Do not create cumulative archives containing
  prior archives or keep failed ANUPQ/GAP databases in persistent dirs.

- ``archive/`` — retained local-only reproducibility material. Put raw
  databases, orbit dumps, compressed packages, and bulky logs here. It
  persists across invocations but is never attached to a model call.

The workspace has a hard storage limit. Prefer streaming/chunked
algorithms and check output sizes before launching a large enumeration.
Never inspect, print, copy, encode, or persist files under ``$CODEX_HOME``
or any authentication token. Credential material is not a research artifact.

## Tools

You have full network access (HTTPS — use it for arXiv / paper /
repository downloads), the standard scientific Python stack
(``sympy``, ``numpy``, ``scipy``, ``networkx``, ``mpmath``, ``pandas``,
``matplotlib``), TeX Live, and — depending on which sandbox image
this run uses — possibly a richer CAS toolchain (e.g. SageMath, GAP,
Singular, PARI/GP). Do not assume any specific CAS
is present; probe at the start of the round with e.g.
``command -v sage gap singular gp`` and adapt. When a CAS *is*
available, prefer it over hand-rolled ``sympy`` for the kind of
problem it was designed for (symbolic algebra, group theory,
commutative algebra, etc.).

Always run CAS binaries non-interactively (e.g. ``sage -c "..."``,
``gap -q -b`` with a script on stdin, ``Singular -b`` or
``singular -b`` with a script file) so the codex subprocess never
hangs on a prompt. Capture results to files under ``code/`` or
``data/`` so later rounds can reference them.

When searching TeX sources, use literal search mode (for example
``rg -F '\\begin{{theorem}}'``) or a small Python script for patterns
containing backslashes/braces; raw regex searches often reject LaTeX
syntax. The workspace includes ``compute_utils.py`` with
``safe_json_dumps`` / ``safe_json_dump`` helpers for NumPy and complex
values. It also installs a local ``sitecustomize.py`` shim so old
scripts using ``numpy.trapz`` continue to work under NumPy 2.x.

## The Author's instructions for this round

{instructions}

## Soft-timeout sentinel

If ``.pwc/runtime/WRAP_UP`` appears, stop new investigations
immediately, finalize ``responses/response_round_{round}.md``, and
call ``$FINISH_BIN``.

## Finishing

When done, invoke ``$FINISH_BIN`` (also installed as ``finish`` on PATH)
with a short JSON body, e.g.::

    "$FINISH_BIN" '{{"status": "done", "summary": "ran 3 experiments, found a counterexample to claim X"}}'

``status`` is one of:

  - ``done``    — task complete, response file written.
  - ``partial`` — ran out of time or hit a dead end; response file
                  still summarizes what you found.
  - ``error``   — could not start the task; explain in ``summary``.

Always ensure ``responses/response_round_{round}.md`` exists before
calling ``$FINISH_BIN`` — that file *is* your reply to the Author.
"""


@dataclass(frozen=True)
class ComputeHandoffInspection:
    attachable: bool
    reason: str
    file_count: int = 0
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    largest_member_bytes: int = 0


@dataclass(frozen=True)
class _WorkspaceFile:
    source: Path
    relative: Path
    size: int
    mtime_ns: int


class Compute(CLIAgent):
    """Codex CLI worker invoked by Author ``<compute_agent>`` blocks.

    Persistent workspace at ``ac_workspaces/<pid>/compute/`` is created
    by the workflow before the first invocation.
    """

    description: ClassVar[str] = (
        "Out-of-band codex CLI worker with persistent workspace for the AC loop."
    )
    execution_mode: ClassVar[str] = "agent"
    cache_enabled: ClassVar[bool] = False

    SANDBOX: ClassVar[SandboxSpec] = SandboxSpec(
        cpu_limit=4,
        memory_gb=8,
        timeout_s=DEFAULT_HARD_TIMEOUT_S,
        backend=DEFAULT_SANDBOX_BACKEND,
        docker_image=DEFAULT_DOCKER_IMAGE,
        docker_no_new_privileges=False,
    )
    SOFT_TIMEOUT_S: ClassVar[int] = DEFAULT_SOFT_TIMEOUT_S

    class Inputs(BaseModel):
        problem: str
        problem_id: str
        round: int
        instructions: str
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        compute_workspace: Path
        model: str = DEFAULT_MODEL
        reasoning_effort: str = DEFAULT_REASONING_EFFORT
        cost_config: str = DEFAULT_COST_CONFIG
        soft_timeout_s: int = Field(default=DEFAULT_SOFT_TIMEOUT_S, ge=0)
        hard_timeout_s: int = Field(default=DEFAULT_HARD_TIMEOUT_S, ge=1)
        workspace_soft_limit_bytes: int = Field(
            default=COMPUTE_WORKSPACE_SOFT_LIMIT_BYTES,
            ge=0,
        )
        workspace_hard_limit_bytes: int = Field(
            default=COMPUTE_WORKSPACE_HARD_LIMIT_BYTES,
            ge=1,
        )
        workspace_soft_limit_entries: int = Field(
            default=COMPUTE_WORKSPACE_SOFT_LIMIT_ENTRIES,
            ge=0,
        )
        workspace_hard_limit_entries: int = Field(
            default=COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES,
            ge=1,
            le=COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES,
        )
        filesystem_min_free_bytes: int = Field(
            default=COMPUTE_FILESYSTEM_MIN_FREE_BYTES,
            ge=0,
        )
        filesystem_min_free_inodes: int = Field(
            default=COMPUTE_FILESYSTEM_MIN_FREE_INODES,
            ge=0,
        )
        filesystem_reservation_bytes: int = Field(
            default=COMPUTE_FILESYSTEM_RESERVATION_BYTES,
            ge=0,
            description=(
                "Bytes this active Compute worker reserves cooperatively. Set "
                "this in the run input when parallel workers need guaranteed "
                "headroom; zero disables coordinated reservations."
            ),
        )
        filesystem_reservation_dir: Path | None = Field(
            default=None,
            description=(
                "Shared reservation registry for workers using the same storage "
                "filesystem. Defaults to the common outputs-root registry."
            ),
        )
        handoff_max_files: int = Field(
            default=COMPUTE_HANDOFF_MAX_FILES,
            ge=2,
            le=COMPUTE_HANDOFF_MAX_FILES,
        )
        handoff_max_compressed_bytes: int = Field(
            default=COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES,
            ge=1,
            le=COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES,
        )
        handoff_max_uncompressed_bytes: int = Field(
            default=COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES,
            gt=1024 * 1024,
            le=COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES,
        )
        handoff_max_member_bytes: int = Field(
            default=COMPUTE_HANDOFF_MAX_MEMBER_BYTES,
            ge=1,
            le=COMPUTE_HANDOFF_MAX_MEMBER_BYTES,
        )
        # ``docker`` (default, image=DEFAULT_DOCKER_IMAGE) or
        # ``subprocess`` (runs codex directly on the host; needed when
        # the pwc docker image is not available locally).
        sandbox_backend: str = DEFAULT_SANDBOX_BACKEND
        # Optional docker image override when ``sandbox_backend=docker``.
        docker_image: str = DEFAULT_DOCKER_IMAGE
        # Codex CLI sandbox flag: ``auto`` (default — bypass under
        # docker, workspace-write under subprocess), ``workspace-write``,
        # ``docker-bypass``, or ``none``.
        codex_sandbox: str = DEFAULT_CODEX_SANDBOX

        @model_validator(mode="after")
        def validate_timeouts(self) -> Self:
            if self.soft_timeout_s >= self.hard_timeout_s:
                raise ValueError("soft_timeout_s must be less than hard_timeout_s")
            if self.workspace_soft_limit_bytes >= self.workspace_hard_limit_bytes:
                raise ValueError(
                    "workspace_soft_limit_bytes must be less than "
                    "workspace_hard_limit_bytes"
                )
            if (
                self.workspace_soft_limit_entries
                >= self.workspace_hard_limit_entries
            ):
                raise ValueError(
                    "workspace_soft_limit_entries must be less than "
                    "workspace_hard_limit_entries"
                )
            if (
                self.handoff_max_member_bytes + 1024 * 1024
                > self.handoff_max_uncompressed_bytes
            ):
                raise ValueError(
                    "handoff_max_member_bytes must leave room for the handoff "
                    "manifest within handoff_max_uncompressed_bytes"
                )
            return self

    class Outputs(BaseModel):
        response_md: str = ""
        zip_path: Path | None = None
        status: str = ""
        summary: str = ""
        workspace: Path | None = None
        error: str | None = None
        handoff_stats: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, ctx: Any, **kw: Any) -> None:
        super().__init__(ctx, **kw)
        self._copied_codex_auth = False
        self._subscription_codex_auth = False
        self._host_codex_auth_text: str | None = None
        self._paid_codex_api_key: str | None = None
        self._last_model: str | None = None
        self._last_cost_config: str | None = None
        self._codex_home_host: Path | None = None
        self._codex_home_env: str | None = None
        self._handoff_secrets: tuple[str, ...] = ()

    # ----- framework hooks ----------------------------------------------------

    def sandbox_root_for(self, inp: BaseModel) -> Path | None:
        ws = Path(inp.compute_workspace)  # type: ignore[attr-defined]
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def _measure_workspace(self, root: Path) -> dict[str, Any]:
        usage = super()._measure_workspace(root)
        for item in usage.get("largest_files", []):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                item["path"] = redact_codex_secrets(
                    item["path"],
                    self._handoff_secrets,
                )
        return usage

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        self._last_model = inp.model  # type: ignore[attr-defined]
        self._last_cost_config = inp.cost_config  # type: ignore[attr-defined]
        soft_timeout_s = int(inp.soft_timeout_s)  # type: ignore[attr-defined]
        hard_timeout_s = int(inp.hard_timeout_s)  # type: ignore[attr-defined]
        self.SOFT_TIMEOUT_S = soft_timeout_s
        self.WORKSPACE_SOFT_LIMIT_BYTES = int(  # type: ignore[attr-defined]
            inp.workspace_soft_limit_bytes  # type: ignore[attr-defined]
        )
        self.WORKSPACE_HARD_LIMIT_BYTES = int(  # type: ignore[attr-defined]
            inp.workspace_hard_limit_bytes  # type: ignore[attr-defined]
        )
        self.WORKSPACE_SOFT_LIMIT_ENTRIES = int(  # type: ignore[attr-defined]
            inp.workspace_soft_limit_entries  # type: ignore[attr-defined]
        )
        self.WORKSPACE_HARD_LIMIT_ENTRIES = int(  # type: ignore[attr-defined]
            inp.workspace_hard_limit_entries  # type: ignore[attr-defined]
        )
        self.WORKSPACE_MIN_FREE_BYTES = int(  # type: ignore[attr-defined]
            inp.filesystem_min_free_bytes  # type: ignore[attr-defined]
        )
        self.WORKSPACE_MIN_FREE_INODES = int(  # type: ignore[attr-defined]
            inp.filesystem_min_free_inodes  # type: ignore[attr-defined]
        )
        self.WORKSPACE_RESERVATION_BYTES = int(  # type: ignore[attr-defined]
            inp.filesystem_reservation_bytes  # type: ignore[attr-defined]
        )
        configured_reservation_dir = inp.filesystem_reservation_dir  # type: ignore[attr-defined]
        self.WORKSPACE_RESERVATION_DIR = (
            Path(configured_reservation_dir)
            if configured_reservation_dir is not None
            else None
        )
        codex_home_host, codex_home_env, docker_extra_args = self._codex_home_paths(inp)
        self._codex_home_host = codex_home_host
        self._codex_home_env = codex_home_env
        # Build per-call SandboxSpec so callers can switch between
        # docker and subprocess without subclassing.
        # ONE read + classification drives BOTH the sandbox env decision and
        # the billing decision. Re-reading the file later would reopen the
        # absent->present race in which ChatGPT auth and a paid env key
        # reach codex simultaneously while accounting says subscription/$0.
        try:
            self._host_codex_auth_text = (
                Path.home() / ".codex" / "auth.json"
            ).read_text(encoding="utf-8")
        except OSError:
            self._host_codex_auth_text = None
        auth_class = classify_codex_auth(self._host_codex_auth_text)
        if auth_class == "unknown":
            raise RuntimeError(
                "Codex authentication is invalid or uses an unrecognized schema: "
                f"{Path.home() / '.codex' / 'auth.json'}. Run `codex login` again "
                "or remove that file to use OPENAI_API_KEY."
            )
        if auth_class == "absent":
            self._paid_codex_api_key = os.environ.get("OPENAI_API_KEY") or None
            if self._paid_codex_api_key is None:
                raise RuntimeError(
                    "Codex authentication is unavailable: no ChatGPT/API-key "
                    "auth.json and OPENAI_API_KEY is not set."
                )
        self._handoff_secrets = extract_codex_auth_secrets(
            self._host_codex_auth_text,
            additional=(self._paid_codex_api_key or "",),
        )
        self.SANDBOX = SandboxSpec(
            cpu_limit=4,
            memory_gb=8,
            timeout_s=hard_timeout_s,
            backend=str(inp.sandbox_backend or DEFAULT_SANDBOX_BACKEND),  # type: ignore[attr-defined,arg-type]
            docker_image=str(inp.docker_image or DEFAULT_DOCKER_IMAGE),  # type: ignore[attr-defined]
            docker_no_new_privileges=False,
            docker_extra_args=docker_extra_args,
            # Codex 0.144 does not authenticate from OPENAI_API_KEY merely
            # being present in its environment. setup() either copies an
            # existing login or pipes the paid key through `codex login` into
            # the transient CODEX_HOME. Never expose provider keys to the
            # model-driven `codex exec` process itself.
            provider_keys=(),
        )
        self.CLI_CMD = _build_codex_cmd(
            model=inp.model,  # type: ignore[attr-defined]
            reasoning_effort=inp.reasoning_effort,  # type: ignore[attr-defined]
            sandbox=self.SANDBOX,
            codex_sandbox=str(inp.codex_sandbox or DEFAULT_CODEX_SANDBOX),  # type: ignore[attr-defined]
        )
        try:
            return await super().run(inp)
        finally:
            self._paid_codex_api_key = None
            self._host_codex_auth_text = None
            self._handoff_secrets = ()

    async def setup(self, sandbox: Sandbox, inp: BaseModel) -> None:
        codex_home = self._ensure_codex_home(inp)
        codex_home.mkdir(parents=True, exist_ok=True)
        await _require_codex_cli_version(sandbox)
        await self._prepare_codex_auth(sandbox, codex_home)
        root = sandbox.root
        # ``scratch`` is explicitly ephemeral. Clear leftovers before the
        # parent class performs its pre-spawn quota check so a cancelled or
        # killed invocation cannot permanently prevent this workspace from
        # resuming.
        await self._reset_scratch(root, inp, phase="setup", strict=True)
        ro = root / "problem_documents_readonly"
        # Wipe & resync: the Author's canonical files may have changed
        # since the last invocation; the worker must see the *current*
        # snapshot. Worker-created code/data/notes/papers in the rest of
        # the workspace are NOT touched.
        if ro.exists():
            shutil.rmtree(ro, ignore_errors=True)
        ro.mkdir(parents=True, exist_ok=True)
        (ro / "problem.txt").write_text(inp.problem or "", encoding="utf-8")  # type: ignore[attr-defined]
        (ro / "answer.tex").write_text(inp.answer_tex or "", encoding="utf-8")  # type: ignore[attr-defined]
        (ro / "research_notes.tex").write_text(
            inp.research_notes_tex or "", encoding="utf-8"  # type: ignore[attr-defined]
        )
        (ro / "references.bib").write_text(
            inp.references_bib or "", encoding="utf-8"  # type: ignore[attr-defined]
        )
        # Ensure standard worker dirs exist. ``code`` is made a package
        # so helper modules there beat the stdlib ``code`` module during
        # local imports.
        for dirname in (
            "responses",
            "code",
            "data",
            "papers",
            "notes",
            "scratch",
            "archive",
        ):
            (root / dirname).mkdir(parents=True, exist_ok=True)
        (root / "code" / "__init__.py").touch()
        _write_helper_if_missing(root / "compute_utils.py", _COMPUTE_UTILS)
        _write_helper_if_missing(root / "sitecustomize.py", _SITECUSTOMIZE)
        shutil.rmtree(root / ".codex-home", ignore_errors=True)
        try:
            (root / _CODEX_LAST_MESSAGE_REL).unlink()
        except FileNotFoundError:
            pass

    async def teardown(self, sandbox: Sandbox, inp: BaseModel) -> None:
        await self._reset_scratch(
            sandbox.root,
            inp,
            phase="teardown",
            strict=False,
        )
        if self._codex_home_host is not None:
            shutil.rmtree(self._codex_home_host, ignore_errors=True)
        shutil.rmtree(sandbox.root / ".codex-home", ignore_errors=True)
        self._codex_home_host = None
        self._codex_home_env = None
        self._copied_codex_auth = False
        self._subscription_codex_auth = False
        self._host_codex_auth_text = None
        self._paid_codex_api_key = None
        self._handoff_secrets = ()

    def extra_env(self, sandbox: Sandbox, inp: BaseModel) -> dict[str, str]:
        self._ensure_codex_home(inp)
        return {"CODEX_HOME": self._codex_home_env or str(self._codex_home_host)}

    def sanitize_cli_output(self, text: str) -> str:
        return redact_codex_secrets(text, self._handoff_secrets)

    def _codex_home_paths(self, inp: BaseModel) -> tuple[Path, str, tuple[str, ...]]:
        name = _safe_codex_home_name(
            f"{getattr(inp, 'problem_id', 'problem')}-r{getattr(inp, 'round', 0)}"
        )
        host = (self.ctx.root_workdir / ".compute_codex_home" / name).resolve()
        backend = str(getattr(inp, "sandbox_backend", DEFAULT_SANDBOX_BACKEND) or DEFAULT_SANDBOX_BACKEND)
        if backend == "docker":
            return host, _DOCKER_CODEX_HOME, ("-v", f"{host}:{_DOCKER_CODEX_HOME}")
        return host, str(host), ()

    def _ensure_codex_home(self, inp: BaseModel) -> Path:
        if self._codex_home_host is None or self._codex_home_env is None:
            host, env, _docker_extra_args = self._codex_home_paths(inp)
            self._codex_home_host = host
            self._codex_home_env = env
        self._codex_home_host.mkdir(parents=True, exist_ok=True)
        return self._codex_home_host

    async def _prepare_codex_auth(self, sandbox: Sandbox, codex_home: Path) -> None:
        """Populate and verify the transient CODEX_HOME without leaking keys."""
        auth_path = codex_home / "auth.json"
        auth_text = self._host_codex_auth_text
        auth_class = classify_codex_auth(auth_text)
        env = {"CODEX_HOME": self._codex_home_env or str(codex_home)}

        if auth_text is not None:
            if auth_class not in {"subscription", "api_key"}:
                raise RuntimeError(
                    "Codex authentication is invalid or uses an unrecognized schema."
                )
            # Uses the text captured and classified once in run(); never
            # re-read the host file and reopen an auth/accounting race.
            auth_path.write_text(auth_text, encoding="utf-8")
        else:
            api_key = self._paid_codex_api_key
            if not api_key:
                raise RuntimeError(
                    "Codex API-key authentication was not prepared before setup."
                )
            try:
                result = await sandbox.run_command(
                    ["codex", "login", "--with-api-key"],
                    timeout_s=CODEX_AUTH_TIMEOUT_S,
                    env_extra=env,
                    input_data=api_key,
                )
            finally:
                # The credential now belongs only in the transient auth file.
                self._paid_codex_api_key = None
            if result.returncode != 0:
                detail = _redact_auth_output(result.stdout, result.stderr, api_key)
                raise RuntimeError(
                    "Codex API-key login failed "
                    f"(exit {result.returncode}): {detail}"
                )
            try:
                auth_text = auth_path.read_text(encoding="utf-8")
            except OSError as e:
                raise RuntimeError(
                    "Codex API-key login succeeded but did not create "
                    f"{auth_path}: {e}"
                ) from e
            auth_class = classify_codex_auth(auth_text)
            if auth_class != "api_key":
                raise RuntimeError(
                    "Codex API-key login created an unrecognized authentication file."
                )

        self._handoff_secrets = extract_codex_auth_secrets(
            auth_text,
            additional=self._handoff_secrets,
        )

        try:
            auth_path.chmod(0o600)
        except OSError:
            pass
        self._copied_codex_auth = True
        # Only a verified ChatGPT login is subscription-covered. API-key auth
        # remains paid and is charged against the ProofCouncil budget.
        self._subscription_codex_auth = auth_class == "subscription"

        status = await sandbox.run_command(
            ["codex", "login", "status"],
            timeout_s=CODEX_AUTH_TIMEOUT_S,
            env_extra=env,
        )
        if status.returncode != 0:
            detail = redact_codex_secrets(
                _redact_auth_output(status.stdout, status.stderr),
                self._handoff_secrets,
            )
            raise RuntimeError(
                "Codex authentication preflight failed "
                f"(exit {status.returncode}): {detail}"
            )

    async def _reset_scratch(
        self,
        root: Path,
        inp: BaseModel,
        *,
        phase: str,
        strict: bool,
    ) -> None:
        scratch = root / "scratch"
        stats: dict[str, Any] = {
            "bytes": 0,
            "allocated_bytes": 0,
            "files": 0,
            "errors": 0,
        }
        try:
            if scratch.is_symlink() or (scratch.exists() and not scratch.is_dir()):
                stat_result = scratch.lstat()
                stats.update(
                    {
                        "bytes": max(0, stat_result.st_size),
                        "allocated_bytes": (
                            max(0, getattr(stat_result, "st_blocks", 0)) * 512
                        ),
                        "files": 1,
                    }
                )
                scratch.unlink()
            elif scratch.is_dir():
                stats = await asyncio.to_thread(self._measure_workspace, scratch)
                await asyncio.to_thread(shutil.rmtree, scratch)
            scratch.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            await self.events.emit(
                "ac.compute.scratch_cleanup_failed",
                {
                    "round": getattr(inp, "round", 0),
                    "phase": phase,
                    "type": type(e).__name__,
                    "msg": str(e),
                    **stats,
                },
            )
            if strict:
                raise RuntimeError(
                    f"could not reset Compute scratch directory before start: {e}"
                ) from e
            return
        if stats.get("files") or stats.get("bytes"):
            await self.events.emit(
                "ac.compute.scratch_cleaned",
                {
                    "round": getattr(inp, "round", 0),
                    "phase": phase,
                    **stats,
                },
            )

    def cli_input(self, inp: BaseModel) -> str:
        text = COMPUTE_WORKER_PROMPT.format(
            round=inp.round,  # type: ignore[attr-defined]
            instructions=inp.instructions or "(no instructions provided)",  # type: ignore[attr-defined]
        )
        if not text.endswith("\n"):
            text += "\n"
        return text

    async def collect(
        self,
        sandbox: Sandbox,
        inp: BaseModel,
        done: CLIDoneRecord,
    ) -> BaseModel:
        root = sandbox.root
        response_path = root / "responses" / f"response_round_{inp.round}.md"  # type: ignore[attr-defined]
        response_md = ""
        if response_path.exists():
            try:
                response_md = response_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                response_md = ""
        if not response_md.strip():
            try:
                response_md = (root / _CODEX_LAST_MESSAGE_REL).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                response_md = ""
        if not response_md.strip() and done.summary:
            response_md = (
                "(Worker did not write responses/response_round_"
                f"{inp.round}.md; falling back to finish summary)\n\n"  # type: ignore[attr-defined]
                f"{done.summary}"
            )
        response_md = self.sanitize_cli_output(response_md)
        summary = self.sanitize_cli_output(done.summary or "")
        await self._reset_scratch(root, inp, phase="collect", strict=False)

        zip_path: Path | None = (
            self.workdir / f"compute_workspace_round_{inp.round}.zip"  # type: ignore[attr-defined]
        )
        workspace_stats = await asyncio.to_thread(self._measure_workspace, root)
        usage_snapshot_path = root / ".pwc" / "runtime" / "compute-workspace-usage.json"
        previous_usage = _read_json_object(usage_snapshot_path)
        handoff_stats: dict[str, Any] = {
            f"workspace_{key}": value for key, value in workspace_stats.items()
        }
        if previous_usage:
            handoff_stats["workspace_growth_bytes"] = workspace_stats["bytes"] - int(
                previous_usage.get("bytes", 0) or 0
            )
            handoff_stats["workspace_growth_files"] = workspace_stats["files"] - int(
                previous_usage.get("files", 0) or 0
            )
            handoff_stats["workspace_growth_allocated_bytes"] = workspace_stats[
                "allocated_bytes"
            ] - int(previous_usage.get("allocated_bytes", 0) or 0)
        else:
            handoff_stats["workspace_growth_bytes"] = None
            handoff_stats["workspace_growth_files"] = None
            handoff_stats["workspace_growth_allocated_bytes"] = None
        _write_json_object(
            usage_snapshot_path,
            {
                "round": inp.round,  # type: ignore[attr-defined]
                "bytes": workspace_stats["bytes"],
                "allocated_bytes": workspace_stats["allocated_bytes"],
                "files": workspace_stats["files"],
            },
        )
        try:
            handoff_stats.update(
                _zip_workspace(
                    root,
                    zip_path,
                    exclude_top=_ZIP_EXCLUDE_TOP,
                    max_files=inp.handoff_max_files,  # type: ignore[attr-defined]
                    max_compressed_bytes=inp.handoff_max_compressed_bytes,  # type: ignore[attr-defined]
                    max_uncompressed_bytes=inp.handoff_max_uncompressed_bytes,  # type: ignore[attr-defined]
                    max_member_bytes=inp.handoff_max_member_bytes,  # type: ignore[attr-defined]
                    max_workspace_entries=inp.workspace_hard_limit_entries,  # type: ignore[attr-defined]
                    secret_values=self._handoff_secrets,
                )
            )
            await self.events.emit("ac.compute.handoff_created", handoff_stats)
            utilization = max(
                handoff_stats["archive_compressed_bytes"]
                / inp.handoff_max_compressed_bytes,  # type: ignore[attr-defined]
                handoff_stats["archive_uncompressed_bytes"]
                / inp.handoff_max_uncompressed_bytes,  # type: ignore[attr-defined]
            )
            if utilization >= 0.5:
                threshold = 0.9 if utilization >= 0.9 else 0.75 if utilization >= 0.75 else 0.5
                await self.events.emit(
                    "ac.compute.handoff_limit_warning",
                    {
                        **handoff_stats,
                        "utilization": utilization,
                        "threshold": threshold,
                    },
                )
            if handoff_stats["omitted_files"]:
                await self.events.emit(
                    "ac.compute.handoff_truncated",
                    handoff_stats,
                )
            if not handoff_stats["attachable"]:
                await self.events.emit(
                    "ac.compute.handoff_rejected",
                    handoff_stats,
                )
                try:
                    zip_path.unlink()
                except FileNotFoundError:
                    pass
                zip_path = None
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            await self.events.emit(
                "ac.compute.zip_failed",
                {"type": type(e).__name__, "msg": str(e), **handoff_stats},
            )
            if zip_path is not None:
                try:
                    zip_path.unlink()
                except FileNotFoundError:
                    pass
            zip_path = None
        return self.Outputs(
            response_md=response_md,
            zip_path=zip_path,
            status=done.status,
            summary=summary,
            workspace=root,
            handoff_stats=handoff_stats,
        )

    async def record_cli_usage(
        self,
        stdout_text: str,
        stderr_text: str,
        done: CLIDoneRecord,
    ) -> None:
        usage = parse_codex_jsonl(stdout_text)
        if usage.n_turns == 0:
            return
        cost = 0.0
        nominal: float | None = None
        billing_unknown = False
        subscription = self._subscription_codex_auth
        cfg_ref: str | None = self._last_cost_config or DEFAULT_COST_CONFIG
        try:
            rates = load_cost_rates(cfg_ref)
        except (KeyError, FileNotFoundError, ValueError) as e:
            await self.events.emit(
                "cli.cost_lookup_failed",
                {"config_ref": cfg_ref, "error": f"{type(e).__name__}: {e}"},
            )
            billing_unknown = not subscription
            cfg_ref = None
        else:
            nominal = cost_for_codex_usage(usage, **rates)
            if not subscription:
                cost = nominal
                self.tracker.add_usd(cost)
        self.tracker.add_tokens(usage.input_tokens + usage.output_tokens)
        await self.events.emit(
            "model.call",
            {
                "model": self._last_model or DEFAULT_MODEL,
                "in_tokens": usage.input_tokens,
                "cached_in_tokens": usage.cached_input_tokens,
                "cache_write_in_tokens": usage.cache_write_input_tokens,
                "out_tokens": usage.output_tokens,
                "reasoning_out_tokens": usage.reasoning_output_tokens,
                "cost_usd": cost,
                "api_equivalent_usd": nominal,
                "subscription": subscription,
                "billing_unknown": billing_unknown,
                "n_turns": usage.n_turns,
                "via": "codex_exec_json",
                "cost_config": cfg_ref,
                "role": "ac_compute_worker",
            },
        )

# --- helpers ----------------------------------------------------------------


def _redact_auth_output(stdout: str, stderr: str, secret: str | None = None) -> str:
    text = f"{stderr}\n{stdout}".strip()
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"\bsk-[A-Za-z0-9_.*-]{4,}", "[redacted]", text)
    return text[:400] or "no output"


async def _require_codex_cli_version(sandbox: Sandbox) -> None:
    result = await sandbox.run_command(["codex", "--version"], timeout_s=10)
    output = f"{result.stdout}\n{result.stderr}".strip()
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"Could not determine Codex CLI version: {output or 'no output'}")
    version = tuple(int(part) for part in match.groups())
    if version < MIN_CODEX_VERSION:
        minimum = ".".join(str(part) for part in MIN_CODEX_VERSION)
        raise RuntimeError(
            f"Codex CLI {minimum} or newer is required for gpt-5.6-sol with max effort; found {match.group(0)}"
        )


def _build_codex_cmd(
    *,
    model: str,
    reasoning_effort: str,
    sandbox: SandboxSpec,
    codex_sandbox: str = DEFAULT_CODEX_SANDBOX,
) -> list[str]:
    backend = resolve_backend(sandbox)
    mode = (codex_sandbox or "auto").strip().lower()
    if mode == "auto":
        mode = "docker-bypass" if backend == "docker" else "workspace-write"
    if mode in {"docker-bypass", "bypass"}:
        sandbox_flag = ["--dangerously-bypass-approvals-and-sandbox"]
    elif mode in {"workspace-write", "workspace"}:
        sandbox_flag = ["--sandbox", "workspace-write"]
    elif mode == "none":
        sandbox_flag = []
    else:
        raise ValueError(f"unsupported codex_sandbox mode: {codex_sandbox!r}")
    return [
        "codex",
        "exec",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        _CODEX_LAST_MESSAGE_REL,
        *sandbox_flag,
        "-",
    ]


def _handoff_file_is_local_only(relative: Path, size: int) -> bool:
    name = relative.name.lower()
    if any(name.endswith(suffix) for suffix in _HANDOFF_LOCAL_ONLY_SUFFIXES):
        return True
    bulk_limit = _HANDOFF_BULK_SUFFIX_MAX_BYTES.get(relative.suffix.lower())
    return bulk_limit is not None and size > bulk_limit


def _handoff_path_is_credential_sensitive(
    relative: Path,
    secret_values: tuple[str, ...],
) -> bool:
    """Reject conventional credential files and secret-bearing filenames."""
    name = relative.name.lower()
    if (
        name in {
            ".env",
            ".netrc",
            "auth.json",
            "credentials.json",
            "secrets.env",
        }
        or name.startswith(".env.")
        or name in {"id_rsa", "id_ed25519"}
    ):
        return True
    rendered = relative.as_posix()
    return any(secret in rendered for secret in secret_values)


def _file_contains_secret(path: Path, secret_values: tuple[str, ...]) -> bool:
    """Scan a bounded candidate file without loading it fully into memory."""
    needles = tuple(
        value.encode("utf-8")
        for value in secret_values
        if len(value) >= 8
    )
    if not needles:
        return False
    overlap = max(len(needle) for needle in needles) - 1
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            window = carry + chunk
            if any(needle in window for needle in needles):
                return True
            carry = window[-overlap:] if overlap > 0 else b""
    return False


def _zip_workspace(
    root: Path,
    out_zip: Path,
    *,
    exclude_top: set[str],
    max_files: int = COMPUTE_HANDOFF_MAX_FILES,
    max_compressed_bytes: int = COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES,
    max_uncompressed_bytes: int = COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES,
    max_member_bytes: int = COMPUTE_HANDOFF_MAX_MEMBER_BYTES,
    max_workspace_entries: int = COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES,
    secret_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a bounded, useful projection of the Compute workspace.

    Responses, notes, and root-level files are retained first. Remaining
    capacity is shared across code, data, and papers before logs and other
    files. Within each group, newer files win. Transient dependency/build
    trees are omitted entirely. Cache exclusions apply to directories and
    ancestors, so a regular output file named ``.cache`` remains eligible.

    Symlinks are dropped unless their resolved target stays inside ``root``
    and outside every excluded path.

    Why: the worker can create symlinks inside its workspace. Without
    this check, a path like ``notes/auth.json -> ../.codex-home/auth.json``
    would exfiltrate the host's Codex credentials past
    ``_ZIP_EXCLUDE_TOP`` because ``zipfile.ZipFile.write`` follows
    symlinks by default and stores the resolved file's contents.
    """
    if max_files < 2:
        raise ValueError("max_files must leave room for a handoff manifest")
    if max_compressed_bytes < 1:
        raise ValueError("max_compressed_bytes must be positive")
    if max_uncompressed_bytes <= 1024 * 1024:
        raise ValueError("max_uncompressed_bytes must leave room for the manifest")
    if max_member_bytes < 1:
        raise ValueError("max_member_bytes must be positive")
    if max_workspace_entries < 1:
        raise ValueError("max_workspace_entries must be positive")

    secret_values = tuple(
        sorted(
            {value for value in secret_values if isinstance(value, str) and len(value) >= 8},
            key=lambda value: (-len(value), value),
        )
    )

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        _compute_handoff_local_only_marker(out_zip).unlink()
    except OSError:
        pass
    if out_zip.exists():
        try:
            out_zip.unlink()
        except OSError:
            pass
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    candidates: list[_WorkspaceFile] = []
    excluded_files = 0
    excluded_bytes = 0
    local_only_files = 0
    local_only_bytes = 0
    local_only_paths: list[str] = []
    credential_files = 0
    credential_bytes = 0
    credential_scan_bytes = 0

    def excluded(rel: Path, *, is_directory: bool) -> bool:
        if not rel.parts:
            return True
        if rel.parts[0] in exclude_top:
            return True
        directory_parts = rel.parts if is_directory else rel.parts[:-1]
        if directory_parts and directory_parts[0] in _ZIP_EXCLUDE_TOP_DIRS:
            return True
        return any(
            part in _ZIP_EXCLUDE_DIRECTORY_PARTS
            or any(
                part.startswith(prefix)
                for prefix in _ZIP_EXCLUDE_DIRECTORY_PREFIXES
            )
            for part in directory_parts
        )

    walk_errors = 0
    walked_entries = 0
    directories = [Path(root)]
    while directories:
        current_path = directories.pop()
        try:
            current_rel = current_path.relative_to(root)
        except ValueError:
            continue
        try:
            entries = os.scandir(current_path)
        except OSError:
            walk_errors += 1
            continue
        try:
            with entries:
                for entry in entries:
                    walked_entries += 1
                    if walked_entries >= max_workspace_entries:
                        raise ValueError(
                            "workspace entry count exceeded handoff scan limit "
                            f"{max_workspace_entries}"
                        )
                    path = Path(entry.path)
                    rel = current_rel / entry.name
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        walk_errors += 1
                        continue
                    if is_dir:
                        if excluded(rel, is_directory=True) or entry.is_symlink():
                            excluded_files += 1
                            continue
                        directories.append(path)
                        continue
                    if excluded(rel, is_directory=False):
                        excluded_files += 1
                        try:
                            excluded_bytes += max(0, path.stat().st_size)
                        except OSError:
                            pass
                        continue
                    source = path
                    if path.is_symlink():
                        try:
                            source = path.resolve(strict=True)
                            target_rel = source.relative_to(root_resolved)
                        except (OSError, RuntimeError, ValueError):
                            excluded_files += 1
                            continue
                        if not source.is_file() or excluded(
                            target_rel,
                            is_directory=False,
                        ):
                            excluded_files += 1
                            continue
                    try:
                        stat_result = source.stat()
                    except OSError:
                        excluded_files += 1
                        continue
                    size = max(0, stat_result.st_size)
                    if _handoff_path_is_credential_sensitive(rel, secret_values):
                        credential_files += 1
                        credential_bytes += size
                        continue
                    if _handoff_file_is_local_only(rel, size):
                        local_only_files += 1
                        local_only_bytes += size
                        if len(local_only_paths) < 100:
                            local_only_paths.append(rel.as_posix())
                        continue
                    candidates.append(
                        _WorkspaceFile(
                            source=source,
                            relative=rel,
                            size=size,
                            mtime_ns=stat_result.st_mtime_ns,
                        )
                    )
        except OSError:
            walk_errors += 1
            continue

    def newest_first(item: _WorkspaceFile) -> tuple[int, str]:
        return (-item.mtime_ns, item.relative.as_posix())

    priority: list[_WorkspaceFile] = []
    pools: dict[str, list[_WorkspaceFile]] = {
        "code": [],
        "data": [],
        "papers": [],
    }
    other: list[_WorkspaceFile] = []
    for item in candidates:
        rel = item.relative
        top = rel.parts[0]
        if top in {"responses", "notes"} or len(rel.parts) == 1:
            priority.append(item)
        elif top in pools:
            pools[top].append(item)
        else:
            other.append(item)

    priority.sort(key=newest_first)
    for pool in pools.values():
        pool.sort(key=newest_first)
    other.sort(key=newest_first)

    ordered = list(priority)
    indexes = {name: 0 for name in pools}
    while True:
        added = False
        for name, pool in pools.items():
            idx = indexes[name]
            if idx < len(pool):
                ordered.append(pool[idx])
                indexes[name] = idx + 1
                added = True
        if not added:
            break
    ordered.extend(other)

    capacity = max_files - 1
    payload_budget = max_uncompressed_bytes - 1024 * 1024
    selected: list[_WorkspaceFile] = []
    selected_bytes = 0
    omitted_reasons: dict[str, str] = {}
    oversized_files = 0
    oversized_bytes = 0
    byte_limited_files = 0
    byte_limited_bytes = 0
    file_limited_files = 0
    file_limited_bytes = 0
    write_failed_files = 0
    write_failed_bytes = 0
    for item in ordered:
        name = item.relative.as_posix()
        if _handoff_path_is_credential_sensitive(item.relative, secret_values):
            credential_files += 1
            credential_bytes += item.size
            continue
        if item.size > max_member_bytes:
            oversized_files += 1
            oversized_bytes += item.size
            omitted_reasons[name] = "member-byte-limit"
            continue
        if selected_bytes + item.size > payload_budget:
            byte_limited_files += 1
            byte_limited_bytes += item.size
            omitted_reasons[name] = "total-byte-limit"
            continue
        if len(selected) >= capacity:
            file_limited_files += 1
            file_limited_bytes += item.size
            omitted_reasons[name] = "file-count-limit"
            continue
        if secret_values:
            if credential_scan_bytes + item.size > max_uncompressed_bytes:
                byte_limited_files += 1
                byte_limited_bytes += item.size
                omitted_reasons[name] = "credential-scan-limit"
                continue
            credential_scan_bytes += item.size
            try:
                contains_secret = _file_contains_secret(item.source, secret_values)
            except OSError:
                write_failed_files += 1
                write_failed_bytes += item.size
                omitted_reasons[name] = "credential-scan-failed"
                continue
            if contains_secret:
                credential_files += 1
                credential_bytes += item.size
                continue
        selected.append(item)
        selected_bytes += item.size

    written = 0
    written_bytes = 0
    with zipfile.ZipFile(
        out_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
        strict_timestamps=False,
    ) as zf:
        for item in selected:
            try:
                zf.write(item.source, arcname=item.relative.as_posix())
                written += 1
                written_bytes += item.size
            except (OSError, PermissionError, ValueError):
                write_failed_files += 1
                write_failed_bytes += item.size
                omitted_reasons[item.relative.as_posix()] = "write-failed"
                continue
        omitted = sorted(omitted_reasons)
        omitted_preview = "\n".join(f"  - {name}" for name in omitted[:200])
        if len(omitted) > 200:
            omitted_preview += f"\n  - ... and {len(omitted) - 200} more"
        manifest = (
            "ProofCouncil Compute handoff\n"
            "============================\n"
            "This is a bounded, read-only projection of the persistent Compute "
            "workspace. Inspect it with Python's zipfile.namelist()/read() and "
            "extract only files needed for the current task; do not extract the "
            "whole archive.\n\n"
            f"Included workspace files: {written}\n"
            f"Included uncompressed bytes: {written_bytes}\n"
            f"Omitted by handoff cap: {len(omitted)}\n"
            f"Omitted above member-byte limit: {oversized_files}\n"
            f"Omitted above total-byte limit: {byte_limited_files}\n"
            f"Omitted above file-count limit: {file_limited_files}\n"
            f"Omitted after read/write failure: {write_failed_files}\n"
            f"Excluded runtime/transient files: {excluded_files}\n"
            f"Excluded local-only artifact files: {local_only_files}\n"
            f"Excluded credential-bearing files: {credential_files}\n"
            f"Maximum archive entries (including this manifest): {max_files}\n"
            f"Maximum compressed bytes: {max_compressed_bytes}\n"
            f"Maximum uncompressed bytes: {max_uncompressed_bytes}\n"
            f"Maximum individual member bytes: {max_member_bytes}\n"
        )
        if omitted_preview:
            manifest += f"\nOmitted paths (first 200):\n{omitted_preview}\n"
        zf.writestr(COMPUTE_HANDOFF_MANIFEST, manifest)

    inspection = inspect_compute_handoff(
        out_zip,
        max_files=max_files,
        max_compressed_bytes=max_compressed_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_member_bytes=max_member_bytes,
    )
    eligible_bytes = sum(item.size for item in candidates)
    largest_selected_files = [
        {"path": item.relative.as_posix(), "bytes": item.size}
        for item in sorted(selected, key=lambda item: item.size, reverse=True)[:10]
    ]
    return {
        "attachable": inspection.attachable,
        "rejection_reason": inspection.reason,
        "included_files": written,
        "included_uncompressed_bytes": inspection.uncompressed_bytes,
        "omitted_files": len(candidates) - written,
        "omitted_bytes": max(0, eligible_bytes - written_bytes),
        "oversized_files": oversized_files,
        "oversized_bytes": oversized_bytes,
        "byte_limited_files": byte_limited_files,
        "byte_limited_bytes": byte_limited_bytes,
        "file_limited_files": file_limited_files,
        "file_limited_bytes": file_limited_bytes,
        "write_failed_files": write_failed_files,
        "write_failed_bytes": write_failed_bytes,
        "write_failed_paths": sorted(
            path
            for path, reason in omitted_reasons.items()
            if reason == "write-failed"
        )[:100],
        "omitted_paths": [
            {"path": path, "reason": omitted_reasons[path]}
            for path in sorted(omitted_reasons)[:100]
        ],
        "excluded_files": excluded_files,
        "excluded_bytes": excluded_bytes,
        "local_only_files": local_only_files,
        "local_only_bytes": local_only_bytes,
        "local_only_paths": sorted(local_only_paths),
        "credential_files": credential_files,
        "credential_bytes": credential_bytes,
        "credential_scan_bytes": credential_scan_bytes,
        "walk_errors": walk_errors,
        "walked_entries": walked_entries,
        "eligible_files": len(candidates),
        "eligible_bytes": eligible_bytes,
        "largest_selected_files": largest_selected_files,
        "archive_entries": inspection.file_count,
        "archive_compressed_bytes": inspection.compressed_bytes,
        "archive_uncompressed_bytes": inspection.uncompressed_bytes,
        "largest_member_bytes": inspection.largest_member_bytes,
        "compression_ratio": (
            inspection.uncompressed_bytes / inspection.compressed_bytes
            if inspection.compressed_bytes
            else None
        ),
        "sha256": _sha256_file(out_zip),
        "max_files": max_files,
        "max_compressed_bytes": max_compressed_bytes,
        "max_uncompressed_bytes": max_uncompressed_bytes,
        "max_member_bytes": max_member_bytes,
    }


def _write_helper_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_object(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_compute_handoff(
    path: Path,
    *,
    max_files: int = COMPUTE_HANDOFF_MAX_FILES,
    max_compressed_bytes: int = COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES,
    max_uncompressed_bytes: int = COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES,
    max_member_bytes: int = COMPUTE_HANDOFF_MAX_MEMBER_BYTES,
) -> ComputeHandoffInspection:
    """Inspect all provider-relevant handoff limits without extracting it."""
    local_only_reason = _compute_handoff_local_only_reason(path)
    if local_only_reason is not None:
        return ComputeHandoffInspection(
            False,
            f"marked local-only after provider rejection: {local_only_reason}",
        )
    if not path.is_file():
        return ComputeHandoffInspection(False, "file is missing")
    try:
        compressed_bytes = path.stat().st_size
    except OSError as e:
        return ComputeHandoffInspection(False, f"cannot stat file: {type(e).__name__}")
    if compressed_bytes > max_compressed_bytes:
        return ComputeHandoffInspection(
            False,
            f"compressed size {compressed_bytes} exceeds {max_compressed_bytes}",
            compressed_bytes=compressed_bytes,
        )
    try:
        with zipfile.ZipFile(path) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        return ComputeHandoffInspection(
            False,
            f"invalid ZIP: {type(e).__name__}",
            compressed_bytes=compressed_bytes,
        )
    file_count = len(members)
    uncompressed_bytes = sum(max(0, info.file_size) for info in members)
    largest_member_bytes = max((max(0, info.file_size) for info in members), default=0)
    details = {
        "file_count": file_count,
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": uncompressed_bytes,
        "largest_member_bytes": largest_member_bytes,
    }
    if file_count > max_files:
        return ComputeHandoffInspection(
            False,
            f"file count {file_count} exceeds {max_files}",
            **details,
        )
    if largest_member_bytes > max_member_bytes:
        return ComputeHandoffInspection(
            False,
            f"largest member {largest_member_bytes} exceeds {max_member_bytes}",
            **details,
        )
    if uncompressed_bytes > max_uncompressed_bytes:
        return ComputeHandoffInspection(
            False,
            f"uncompressed size {uncompressed_bytes} exceeds {max_uncompressed_bytes}",
            **details,
        )
    return ComputeHandoffInspection(True, "", **details)


def compute_handoff_is_attachable(path: Path) -> bool:
    """Reject malformed handoffs or any archive above a byte/count limit."""
    return inspect_compute_handoff(path).attachable


def mark_compute_handoff_local_only(path: Path, *, reason: str) -> None:
    """Persistently quarantine an optional handoff rejected by a provider."""
    path = Path(path)
    reason = str(reason).strip()[:1000] or "provider rejected attachment"
    _write_compute_handoff_local_only_marker(path, reason)
    source = _compute_handoff_reference_source(path)
    if source is not None:
        _write_compute_handoff_local_only_marker(source, reason)


def _write_compute_handoff_local_only_marker(path: Path, reason: str) -> None:
    try:
        os.setxattr(path, _COMPUTE_HANDOFF_LOCAL_ONLY_XATTR, reason.encode("utf-8"))
    except (AttributeError, OSError):
        pass
    marker = _compute_handoff_local_only_marker(path)
    temporary = marker.with_name(marker.name + ".tmp")
    payload = {"version": 1, "artifact": str(path), "reason": reason}
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(marker)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def _compute_handoff_local_only_reason(path: Path) -> str | None:
    path = Path(path)
    reason = _direct_compute_handoff_local_only_reason(path)
    if reason is not None:
        return reason

    # Resume snapshots retain only a hardlink/symlink plus this metadata.
    # Follow its relocatable source reference to the source's adjacent marker;
    # do not write into the ZIP itself, because that invalidates recorded size
    # and SHA-256 metadata.
    candidate = _compute_handoff_reference_source(path)
    if candidate is None:
        return None
    return _direct_compute_handoff_local_only_reason(candidate)


def _compute_handoff_reference_source(path: Path) -> Path | None:
    metadata_path = Path(path).parent / "compute_handoff_ref.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    source_relative = payload.get("source_relative")
    source = payload.get("source")
    if isinstance(source_relative, str) and source_relative:
        candidate = Path(path).parent / source_relative
    elif isinstance(source, str) and source:
        candidate = Path(source)
    else:
        return None
    try:
        if candidate.resolve(strict=False) == Path(path).resolve(strict=False):
            return None
    except OSError:
        pass
    return candidate


def _direct_compute_handoff_local_only_reason(path: Path) -> str | None:
    marker = _compute_handoff_local_only_marker(path)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        reason = payload.get("reason") if isinstance(payload, dict) else None
        if reason:
            return str(reason)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        value = os.getxattr(path, _COMPUTE_HANDOFF_LOCAL_ONLY_XATTR)
    except (AttributeError, OSError):
        pass
    else:
        return value.decode("utf-8", errors="replace") or "provider rejected attachment"
    return None


def _compute_handoff_local_only_marker(path: Path) -> Path:
    return Path(str(path) + COMPUTE_HANDOFF_LOCAL_ONLY_SUFFIX)


def _safe_codex_home_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip(".-")
    return cleaned or "compute"


def render_compute_reply_for_author(compute_out: Compute.Outputs) -> str:
    """Format the compute worker's reply for inclusion in the next
    Author turn's user prompt.

    The full body is the worker's ``responses/response_round_N.md``;
    a small header carries the worker's ``finish`` status and a
    pointer to the bounded read-only zip attachment the Author can inspect
    selectively via code_interpreter on its next turn.
    """
    if compute_out is None:  # type: ignore[unreachable]
        return "(no compute reply)"
    status_line = f"status: {compute_out.status or '(unknown)'}"
    if compute_out.error:
        status_line += f" — error: {compute_out.error}"
    zip_path = Path(compute_out.zip_path) if compute_out.zip_path is not None else None
    if zip_path is None:
        zip_line = "workspace zip: (none)"
    else:
        inspection = inspect_compute_handoff(zip_path)
        if inspection.attachable:
            zip_line = f"workspace zip attached as: {zip_path.name}"
        else:
            zip_line = (
                "workspace zip: omitted by the handoff safety contract "
                f"({inspection.reason})"
            )
    body = compute_out.response_md or "(empty response)"
    return (
        f"### Compute worker reply ###\n"
        f"{status_line}\n"
        f"{zip_line}\n\n"
        f"{body}"
    )


__all__ = [
    "Compute",
    "COMPUTE_WORKER_PROMPT",
    "COMPUTE_HANDOFF_MAX_FILES",
    "COMPUTE_HANDOFF_MAX_COMPRESSED_BYTES",
    "COMPUTE_HANDOFF_MAX_UNCOMPRESSED_BYTES",
    "COMPUTE_HANDOFF_MAX_MEMBER_BYTES",
    "COMPUTE_HANDOFF_MANIFEST",
    "COMPUTE_HANDOFF_LOCAL_ONLY_SUFFIX",
    "COMPUTE_WORKSPACE_SOFT_LIMIT_BYTES",
    "COMPUTE_WORKSPACE_HARD_LIMIT_BYTES",
    "COMPUTE_WORKSPACE_SOFT_LIMIT_ENTRIES",
    "COMPUTE_WORKSPACE_HARD_LIMIT_ENTRIES",
    "COMPUTE_FILESYSTEM_MIN_FREE_BYTES",
    "COMPUTE_FILESYSTEM_MIN_FREE_INODES",
    "COMPUTE_FILESYSTEM_RESERVATION_BYTES",
    "ComputeHandoffInspection",
    "compute_handoff_is_attachable",
    "inspect_compute_handoff",
    "mark_compute_handoff_local_only",
    "render_compute_reply_for_author",
]
