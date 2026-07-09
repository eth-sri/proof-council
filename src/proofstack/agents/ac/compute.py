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
  2. zips the persistent workspace (excluding readonly + codex auth +
     framework state) into the agent workdir for attachment to the
     next Author container call.

Same docker sandbox / soft-timeout / codex-jsonl usage accounting as
the PWC Worker. Model, effort, cost config, and timeout defaults can be
overridden per call through ``Inputs``.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, Field

from proofstack.cli_usage import (
    cost_for_codex_usage,
    load_cost_rates,
    parse_codex_jsonl,
)
from proofstack.kinds.cli import CLIAgent, CLIDoneRecord
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
  workspace, citations of papers you found. The whole workspace is
  also zipped and attached to the next Author call, so you can refer
  to ``code/``, ``data/``, ``papers/``, etc. paths and the Author can
  inspect them via its own code_interpreter.

- Everything else (``code/``, ``data/``, ``papers/``, ``notes/``, …)
  is yours to organize freely. Files persist across invocations, so
  later rounds can build on prior compute artifacts.

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

    class Outputs(BaseModel):
        response_md: str = ""
        zip_path: Path | None = None
        status: str = ""
        summary: str = ""
        workspace: Path | None = None
        error: str | None = None

    def __init__(self, ctx: Any, **kw: Any) -> None:
        super().__init__(ctx, **kw)
        self._copied_codex_auth = False
        self._last_model: str | None = None
        self._last_cost_config: str | None = None
        self._codex_home_host: Path | None = None
        self._codex_home_env: str | None = None

    # ----- framework hooks ----------------------------------------------------

    def sandbox_root_for(self, inp: BaseModel) -> Path | None:
        ws = Path(inp.compute_workspace)  # type: ignore[attr-defined]
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        self._last_model = inp.model  # type: ignore[attr-defined]
        self._last_cost_config = inp.cost_config  # type: ignore[attr-defined]
        soft_timeout_s = int(inp.soft_timeout_s)  # type: ignore[attr-defined]
        hard_timeout_s = int(inp.hard_timeout_s)  # type: ignore[attr-defined]
        if soft_timeout_s >= hard_timeout_s:
            raise ValueError("soft_timeout_s must be less than hard_timeout_s")
        self.SOFT_TIMEOUT_S = soft_timeout_s
        codex_home_host, codex_home_env, docker_extra_args = self._codex_home_paths(inp)
        self._codex_home_host = codex_home_host
        self._codex_home_env = codex_home_env
        # Build per-call SandboxSpec so callers can switch between
        # docker and subprocess without subclassing.
        self.SANDBOX = SandboxSpec(
            cpu_limit=4,
            memory_gb=8,
            timeout_s=hard_timeout_s,
            backend=str(inp.sandbox_backend or DEFAULT_SANDBOX_BACKEND),  # type: ignore[attr-defined,arg-type]
            docker_image=str(inp.docker_image or DEFAULT_DOCKER_IMAGE),  # type: ignore[attr-defined]
            docker_no_new_privileges=False,
            docker_extra_args=docker_extra_args,
        )
        self.CLI_CMD = _build_codex_cmd(
            model=inp.model,  # type: ignore[attr-defined]
            reasoning_effort=inp.reasoning_effort,  # type: ignore[attr-defined]
            sandbox=self.SANDBOX,
            codex_sandbox=str(inp.codex_sandbox or DEFAULT_CODEX_SANDBOX),  # type: ignore[attr-defined]
        )
        return await super().run(inp)

    async def setup(self, sandbox: Sandbox, inp: BaseModel) -> None:
        root = sandbox.root
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
        for dirname in ("responses", "code", "data", "papers", "notes"):
            (root / dirname).mkdir(parents=True, exist_ok=True)
        (root / "code" / "__init__.py").touch()
        _write_helper_if_missing(root / "compute_utils.py", _COMPUTE_UTILS)
        _write_helper_if_missing(root / "sitecustomize.py", _SITECUSTOMIZE)
        codex_home = self._ensure_codex_home(inp)
        codex_home.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(root / ".codex-home", ignore_errors=True)
        try:
            (root / _CODEX_LAST_MESSAGE_REL).unlink()
        except FileNotFoundError:
            pass

        # Copy codex auth — same approach as PWC worker's
        # ``copy_codex_auth``. Scrubbed in teardown.
        host_auth = Path.home() / ".codex" / "auth.json"
        if host_auth.exists():
            auth_path = codex_home / "auth.json"
            auth_path.write_text(host_auth.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                auth_path.chmod(0o600)
            except OSError:
                pass
            self._copied_codex_auth = True

    async def teardown(self, sandbox: Sandbox, inp: BaseModel) -> None:
        if self._codex_home_host is not None:
            shutil.rmtree(self._codex_home_host, ignore_errors=True)
        shutil.rmtree(sandbox.root / ".codex-home", ignore_errors=True)
        self._codex_home_host = None
        self._codex_home_env = None
        self._copied_codex_auth = False

    def extra_env(self, sandbox: Sandbox, inp: BaseModel) -> dict[str, str]:
        self._ensure_codex_home(inp)
        return {"CODEX_HOME": self._codex_home_env or str(self._codex_home_host)}

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
        zip_path = self.workdir / f"compute_workspace_round_{inp.round}.zip"  # type: ignore[attr-defined]
        try:
            _zip_workspace(root, zip_path, exclude_top=_ZIP_EXCLUDE_TOP)
        except OSError as e:
            await self.events.emit(
                "ac.compute.zip_failed",
                {"type": type(e).__name__, "msg": str(e)},
            )
            zip_path = None  # type: ignore[assignment]
        return self.Outputs(
            response_md=response_md,
            zip_path=zip_path,
            status=done.status,
            summary=done.summary or "",
            workspace=root,
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
        cfg_ref = self._last_cost_config or DEFAULT_COST_CONFIG
        try:
            rates = load_cost_rates(cfg_ref)
        except (KeyError, FileNotFoundError, ValueError) as e:
            await self.events.emit(
                "cli.cost_lookup_failed",
                {"config_ref": cfg_ref, "error": f"{type(e).__name__}: {e}"},
            )
            return
        cost = cost_for_codex_usage(usage, **rates)
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
                "n_turns": usage.n_turns,
                "via": "codex_exec_json",
                "cost_config": cfg_ref,
                "role": "ac_compute_worker",
            },
        )

# --- helpers ----------------------------------------------------------------


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


def _zip_workspace(root: Path, out_zip: Path, *, exclude_top: set[str]) -> None:
    """Zip everything under ``root`` except top-level directories in
    ``exclude_top``. Symlinks are dropped unless their resolved target
    stays inside ``root`` and outside any excluded top-level directory.

    Why: the worker can create symlinks inside its workspace. Without
    this check, a path like ``notes/auth.json -> ../.codex-home/auth.json``
    would exfiltrate the host's Codex credentials past
    ``_ZIP_EXCLUDE_TOP`` because ``zipfile.ZipFile.write`` follows
    symlinks by default and stores the resolved file's contents.
    """
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        try:
            out_zip.unlink()
        except OSError:
            pass
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            top = rel.parts[0] if rel.parts else ""
            if top in exclude_top:
                continue
            if path.is_symlink():
                # Resolve the target and require it to stay inside
                # ``root`` and outside any excluded top-level dir.
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                try:
                    target_rel = target.relative_to(root_resolved)
                except ValueError:
                    continue
                target_top = target_rel.parts[0] if target_rel.parts else ""
                if target_top in exclude_top:
                    continue
                if not target.is_file():
                    continue
                try:
                    zf.write(target, arcname=rel.as_posix())
                except (OSError, PermissionError):
                    continue
                continue
            if path.is_dir():
                continue
            try:
                zf.write(path, arcname=rel.as_posix())
            except (OSError, PermissionError):
                # Skip files we cannot read (e.g. transient lock files);
                # the zip is best-effort.
                continue


def _write_helper_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def _safe_codex_home_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip(".-")
    return cleaned or "compute"


def render_compute_reply_for_author(compute_out: Compute.Outputs) -> str:
    """Format the compute worker's reply for inclusion in the next
    Author turn's user prompt.

    The full body is the worker's ``responses/response_round_N.md``;
    a small header carries the worker's ``finish`` status and a
    pointer to the read-only zip attachment the Author can ``unzip``
    via code_interpreter on its next turn.
    """
    if compute_out is None:  # type: ignore[unreachable]
        return "(no compute reply)"
    status_line = f"status: {compute_out.status or '(unknown)'}"
    if compute_out.error:
        status_line += f" — error: {compute_out.error}"
    zip_line = (
        f"workspace zip attached as: {Path(compute_out.zip_path).name}"
        if compute_out.zip_path is not None
        else "workspace zip: (none)"
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
    "render_compute_reply_for_author",
]
