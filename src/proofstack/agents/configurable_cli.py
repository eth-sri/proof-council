"""YAML-configurable CLI agent.

Use this when a node is:

  inputs -> prompt/files in a sandbox workspace -> external CLI -> files/done.json outputs.

The goal is to make Codex/Claude-style workers configurable from workflow
YAML instead of requiring one Python subclass per worker role.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from proofstack.cli_usage import (
    cost_for_codex_usage,
    load_cost_rates,
    parse_claude_json,
    parse_codex_jsonl,
)
from proofstack.kinds.cli import CLIAgent, CLIDoneRecord
from proofstack.sandbox import resolve_backend
from proofstack.sandbox.base import Sandbox, SandboxSpec


class ConfigurableCLIAgent(CLIAgent):
    """Generic CLI component configured through ``components:`` YAML."""

    description: ClassVar[str] = "YAML-defined CLI worker with a workspace."
    SANDBOX: ClassVar[SandboxSpec] = SandboxSpec()
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        workspace: str | Path | None = Field(default=None, description="Optional persistent workspace path.")

    class Outputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        workspace: str = Field(description="Workspace path used by the CLI run.")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._raw_cmd = self.component_config.get("cmd") or []
        self._copied_codex_auth = False
        self._active_workspace_root: Path | None = None

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        self.CLI_CMD = self._command_for(inp)
        if "soft_timeout_s" in self.component_config:
            self.SOFT_TIMEOUT_S = int(self.component_config["soft_timeout_s"] or 0)
        return await super().run(inp)

    def sandbox_root_for(self, inp: BaseModel) -> Path | None:
        fields = self._fields(inp)
        workspace_field = str(self.component_config.get("workspace_input") or "workspace")
        raw_workspace = fields.get(workspace_field)
        if raw_workspace:
            return self._workspace_path(str(raw_workspace))

        raw_root = self.component_config.get("workspace_root")
        if isinstance(raw_root, str) and raw_root.strip():
            return self._workspace_path(_format_template(raw_root, fields))
        return super().sandbox_root_for(inp)

    async def setup(self, sandbox: Sandbox, inp: BaseModel) -> None:
        self._active_workspace_root = sandbox.root
        fields = self._fields(inp, workspace=sandbox.root)

        await self._write_file_group(sandbox, fields, "bootstrap_files", overwrite_default=False)
        await self._write_file_group(sandbox, fields, "input_files", overwrite_default=True)

        if self._copy_codex_auth_enabled():
            (sandbox.root / ".codex-home").mkdir(parents=True, exist_ok=True)
            host_auth = Path.home() / ".codex" / "auth.json"
            if host_auth.exists():
                await sandbox.write_file(".codex-home/auth.json", host_auth.read_text(encoding="utf-8"))
                try:
                    (sandbox.root / ".codex-home" / "auth.json").chmod(0o600)
                except OSError:
                    pass
                self._copied_codex_auth = True

    async def teardown(self, sandbox: Sandbox, inp: BaseModel) -> None:
        if self._copy_codex_auth_enabled():
            shutil.rmtree(sandbox.root / ".codex-home", ignore_errors=True)
            self._copied_codex_auth = False

    def cli_input(self, inp: BaseModel) -> str:
        fields = self._fields(inp, workspace=self._active_workspace_root)
        raw = self.component_config.get("prompt") or ""
        text = _format_template(str(raw), fields)
        if self.component_config.get("append_prompt_newline", True) and not text.endswith("\n"):
            text += "\n"
        return text

    def extra_env(self, sandbox: Sandbox, inp: BaseModel) -> dict[str, str]:
        fields = self._fields(inp, workspace=sandbox.root)
        env: dict[str, str] = {}
        raw_env = self.component_config.get("env") or {}
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                env[str(key)] = _format_template(str(value), fields)
        if self._copy_codex_auth_enabled():
            env.setdefault("CODEX_HOME", str(sandbox.root / ".codex-home"))
        return env

    async def collect(
        self,
        sandbox: Sandbox,
        inp: BaseModel,
        done: CLIDoneRecord,
    ) -> BaseModel:
        data: dict[str, Any] = {"workspace": str(sandbox.root)}
        data.update(self._constant_outputs(inp, sandbox))
        data.update(self._done_outputs(done))
        await self._collect_file_outputs(sandbox, data)
        return self.Outputs.model_validate(data)

    async def record_cli_usage(
        self,
        stdout_text: str,
        stderr_text: str,
        done: CLIDoneRecord,
    ) -> None:
        usage_cfg = self.component_config.get("usage") or {}
        if not isinstance(usage_cfg, dict):
            return
        kind = usage_cfg.get("type")
        if kind == "claude_json":
            await self._record_claude_usage(stdout_text)
            return
        if kind != "codex_jsonl":
            return
        usage = parse_codex_jsonl(stdout_text)
        if usage.n_turns == 0:
            return
        cfg_ref = str(usage_cfg.get("cost_config") or "models/openai/gpt-54-mini")
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
                "model": str(
                    self.component_config.get("model")
                    or usage_cfg.get("model")
                    or _model_from_cmd(self.CLI_CMD)
                    or "codex"
                ),
                "in_tokens": usage.input_tokens,
                "cached_in_tokens": usage.cached_input_tokens,
                "out_tokens": usage.output_tokens,
                "reasoning_out_tokens": usage.reasoning_output_tokens,
                "cost_usd": cost,
                "n_turns": usage.n_turns,
                "via": "codex_exec_json",
                "cost_config": cfg_ref,
            },
        )

    async def _record_claude_usage(self, stdout_text: str) -> None:
        usage = parse_claude_json(stdout_text)
        if not usage.found:
            return
        # Charge tokens (gated by max_tokens) but NOT usd: a subscription run has
        # no API spend, and add_usd would trip the max_usd: 0.0 gate. Tokens are
        # the real subscription limit (Anthropic's rolling token window). We meter
        # the full throughput (input + cache create + cache read + output): cache
        # reads dominate an agentic loop and counting only input+output undercounts
        # ~40x. All categories are recorded below so the weighting stays visible.
        self.tracker.add_tokens(usage.metered_tokens)
        await self.events.emit(
            "model.call",
            {
                "model": str(
                    self.component_config.get("model")
                    or _model_from_cmd(self.CLI_CMD)
                    or "claude"
                ),
                "in_tokens": usage.input_tokens,
                "cache_creation_in_tokens": usage.cache_creation_input_tokens,
                "cached_in_tokens": usage.cache_read_input_tokens,
                "out_tokens": usage.output_tokens,
                "metered_tokens": usage.metered_tokens,
                "cost_usd": usage.total_cost_usd,
                "n_turns": usage.num_turns,
                "via": "claude_exec_json",
            },
        )

    def _command_for(self, inp: BaseModel) -> list[str]:
        fields = self._fields(inp)
        cmd = _coerce_cmd(self._raw_cmd, fields)
        if _is_codex_exec_cmd(cmd):
            model = str(self.component_config.get("model") or "").strip()
            if model:
                cmd = _with_codex_model(cmd, model)
            reasoning_effort = str(self.component_config.get("model_reasoning_effort") or "").strip()
            if reasoning_effort:
                cmd = _with_codex_reasoning_effort(cmd, reasoning_effort)
        else:
            # Per-node model override for non-codex CLIs (e.g. claude): a node can
            # use a stronger model than the global {claude_model} default.
            model = str(self.component_config.get("model") or "").strip()
            if model:
                cmd = _with_claude_model(cmd, model)
        if self.component_config.get("prompt") and _is_codex_exec_cmd(cmd) and _codex_prompt_arg_index(cmd) is None:
            cmd = [*cmd, "-"]
        codex_sandbox = str(self.component_config.get("codex_sandbox") or "").strip()
        if codex_sandbox and codex_sandbox.lower() != "none":
            cmd = _with_codex_sandbox_flag(cmd, codex_sandbox, resolve_backend(self.SANDBOX))
        return cmd

    def _workspace_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.ctx.root_workdir / path
        path = path.resolve()
        root = self.ctx.root_workdir.resolve()
        allow_outside = bool(self.component_config.get("allow_workspace_outside_run"))
        if not allow_outside:
            try:
                path.relative_to(root)
            except ValueError as e:
                raise ValueError(f"workspace path escapes run directory: {path}") from e
        return path

    def _fields(self, inp: BaseModel, *, workspace: Path | str | None = None) -> dict[str, Any]:
        fields = inp.model_dump(mode="json")
        if workspace is not None:
            fields["workspace"] = str(workspace)
        return fields

    def _copy_codex_auth_enabled(self) -> bool:
        return bool(self.component_config.get("copy_codex_auth"))

    async def _write_file_group(
        self,
        sandbox: Sandbox,
        fields: dict[str, Any],
        key: str,
        *,
        overwrite_default: bool,
    ) -> None:
        files = self.component_config.get(key) or {}
        if not isinstance(files, dict):
            return
        for relpath, spec in files.items():
            rel = _safe_relpath(str(relpath))
            overwrite = overwrite_default
            if isinstance(spec, dict):
                overwrite = bool(spec.get("overwrite", overwrite_default))
            target = sandbox.root / rel
            if target.exists() and not overwrite:
                continue
            content = _file_content(spec, fields)
            await sandbox.write_file(rel, content)

    def _constant_outputs(self, inp: BaseModel, sandbox: Sandbox) -> dict[str, Any]:
        fields = self._fields(inp, workspace=sandbox.root)
        raw = self.component_config.get("constant_outputs") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for field, value in raw.items():
            if isinstance(value, str):
                _set_nested(out, str(field), _format_template(value, fields))
            else:
                _set_nested(out, str(field), value)
        return out

    def _done_outputs(self, done: CLIDoneRecord) -> dict[str, Any]:
        raw = self.component_config.get("done_outputs")
        if raw is None:
            raw = {
                key: key
                for key in _configured_output_fields(self.component_config)
                if key in {"status", "summary", "diff_summary", "open_questions", "artifacts"}
            }
        if not isinstance(raw, dict):
            return {}
        source = done.model_dump(mode="json")
        out: dict[str, Any] = {}
        for field, spec in raw.items():
            done_field = str(spec.get("field") if isinstance(spec, dict) else spec)
            value = source.get(done_field)
            if isinstance(spec, dict) and spec.get("join") and isinstance(value, list):
                value = str(spec.get("sep", "\n")).join(str(item) for item in value)
            _set_nested(out, str(field), value)
        return out

    async def _collect_file_outputs(self, sandbox: Sandbox, data: dict[str, Any]) -> None:
        raw = self.component_config.get("output_files") or {}
        if not isinstance(raw, dict):
            return
        for field, spec in raw.items():
            relpath, kind, default = _output_file_spec(spec)
            rel = _safe_relpath(relpath)
            path = sandbox.root / rel
            if kind == "path":
                value: Any = str(path)
            elif kind == "exists":
                value = path.exists()
            elif kind == "listing":
                value = _workspace_listing(path if path.exists() else sandbox.root)
            elif not path.exists():
                value = default
            elif kind == "json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    value = default
            elif kind == "int":
                try:
                    value = int(path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    value = default
            elif kind == "float":
                try:
                    value = float(path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    value = default
            else:
                value = path.read_text(encoding="utf-8")
            _set_nested(data, str(field), value)


def _coerce_cmd(raw: Any, fields: dict[str, Any]) -> list[str]:
    if isinstance(raw, str):
        raw = _format_template(raw, fields)
        return shlex.split(raw)
    if isinstance(raw, (list, tuple)):
        return [_format_template(str(part), fields) for part in raw]
    return []


def _with_codex_sandbox_flag(cmd: list[str], mode: str, backend: str) -> list[str]:
    if not cmd or Path(cmd[0]).name != "codex" or _has_codex_sandbox_flag(cmd):
        return cmd
    mode = mode.lower()
    if mode == "auto":
        mode = "docker-bypass" if backend == "docker" else "workspace-write"
    flag: list[str]
    if mode in {"docker-bypass", "bypass"}:
        flag = ["--dangerously-bypass-approvals-and-sandbox"]
    elif mode in {"workspace-write", "workspace"}:
        flag = ["--sandbox", "workspace-write"]
    else:
        raise ValueError(f"unsupported codex_sandbox mode: {mode!r}")
    prompt_idx = _codex_prompt_arg_index(cmd)
    if prompt_idx is not None:
        return [*cmd[:prompt_idx], *flag, *cmd[prompt_idx:]]
    return [*cmd, *flag]


def _with_codex_model(cmd: list[str], model: str) -> list[str]:
    cmd = _without_codex_model(cmd)
    return _insert_codex_exec_options(cmd, ["-m", model])


def _with_claude_model(cmd: list[str], model: str) -> list[str]:
    """Override the model of a non-codex (e.g. claude) command in place. Rewrites
    the existing ``--model``/``-m`` value, or appends one if absent."""
    out = list(cmd)
    for i, part in enumerate(out):
        if part in ("--model", "-m") and i + 1 < len(out):
            out[i + 1] = model
            return out
        if part.startswith("--model="):
            out[i] = f"--model={model}"
            return out
    return [*out, "--model", model]


def _without_codex_model(cmd: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        part = cmd[i]
        if part in {"-m", "--model"}:
            i += 2
            continue
        if part.startswith("--model="):
            i += 1
            continue
        out.append(part)
        i += 1
    return out


def _with_codex_reasoning_effort(cmd: list[str], effort: str) -> list[str]:
    cmd = _without_codex_reasoning_effort(cmd)
    return _insert_codex_exec_options(cmd, ["-c", f'model_reasoning_effort="{effort}"'])


def _without_codex_reasoning_effort(cmd: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        part = cmd[i]
        if part in {"-c", "--config"} and i + 1 < len(cmd):
            if _is_reasoning_effort_config(cmd[i + 1]):
                i += 2
                continue
        if part.startswith("--config=") and _is_reasoning_effort_config(part.split("=", 1)[1]):
            i += 1
            continue
        out.append(part)
        i += 1
    return out


def _is_reasoning_effort_config(value: str) -> bool:
    return value.strip().startswith("model_reasoning_effort")


def _insert_codex_exec_options(cmd: list[str], options: list[str]) -> list[str]:
    try:
        idx = cmd.index("exec") + 1
    except ValueError:
        return cmd
    return [*cmd[:idx], *options, *cmd[idx:]]


def _is_codex_exec_cmd(cmd: list[str]) -> bool:
    return bool(cmd and Path(cmd[0]).name == "codex" and "exec" in cmd)


def _codex_prompt_arg_index(cmd: list[str]) -> int | None:
    if not _is_codex_exec_cmd(cmd):
        return None
    try:
        i = cmd.index("exec") + 1
    except ValueError:
        return None
    value_options = {
        "-c",
        "--config",
        "-i",
        "--image",
        "-m",
        "--model",
        "--local-provider",
        "-p",
        "--profile",
        "-s",
        "--sandbox",
        "-C",
        "--cd",
        "--add-dir",
        "--output-schema",
        "--color",
        "-o",
        "--output-last-message",
    }
    value_options_with_equals = {
        "--config",
        "--image",
        "--model",
        "--local-provider",
        "--profile",
        "--sandbox",
        "--cd",
        "--add-dir",
        "--output-schema",
        "--color",
        "--output-last-message",
    }
    while i < len(cmd):
        part = cmd[i]
        if part == "--":
            return i + 1 if i + 1 < len(cmd) else None
        if part in value_options:
            i += 2
            continue
        if any(part.startswith(f"{opt}=") for opt in value_options_with_equals):
            i += 1
            continue
        if part == "-" or not part.startswith("-"):
            return i
        i += 1
    return None


def _has_codex_sandbox_flag(cmd: list[str]) -> bool:
    return any(
        part in {"--dangerously-bypass-approvals-and-sandbox", "--sandbox"}
        for part in cmd
    )


def _safe_relpath(raw: str) -> str:
    path = Path(raw)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"workspace file path must be relative and stay inside workspace: {raw!r}")
    return path.as_posix()


def _file_content(spec: Any, fields: dict[str, Any]) -> str:
    if isinstance(spec, dict):
        if "from_path_input" in spec:
            path = fields.get(str(spec["from_path_input"]))
            if not path:
                return str(spec.get("default", ""))
            try:
                return Path(str(path)).read_text(encoding="utf-8", errors="replace")
            except (OSError, FileNotFoundError):
                return str(spec.get("default", ""))
        if "from_input" in spec:
            value = fields.get(str(spec["from_input"]), "")
            return "" if value is None else str(value)
        raw = spec.get("content", spec.get("template", ""))
    else:
        raw = spec
    if raw is None:
        return ""
    return _format_template(str(raw), fields)


def _format_template(template: str, fields: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            return match.group(0)
        value = fields.get(key, "")
        return "" if value is None else str(value)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)


def _output_file_spec(spec: Any) -> tuple[str, str, Any]:
    if isinstance(spec, dict):
        return (
            str(spec.get("path") or ""),
            str(spec.get("type") or "text"),
            spec.get("default", ""),
        )
    return (str(spec), "text", "")


def _set_nested(out: dict[str, Any], field: str, value: Any) -> None:
    parts = [part for part in field.split(".") if part]
    if not parts:
        return
    cur = out
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def _workspace_listing(root: Path) -> str:
    if root.is_file():
        return root.name
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if ".codex-home" in path.parts:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{rel.as_posix()}{suffix}")
        if len(lines) >= 200:
            lines.append("...")
            break
    return "\n".join(lines)


def _configured_output_fields(config: dict[str, Any]) -> set[str]:
    fields: set[str] = {"workspace"}
    raw_schema = config.get("output_schema")
    if isinstance(raw_schema, dict):
        fields.update(str(key) for key in raw_schema)
    raw_files = config.get("output_files") or {}
    if isinstance(raw_files, dict):
        fields.update(str(key).split(".", 1)[0] for key in raw_files)
    return fields


def _model_from_cmd(cmd: list[str]) -> str | None:
    for idx, part in enumerate(cmd[:-1]):
        if part in {"-m", "--model"}:
            return cmd[idx + 1]
        if part.startswith("--model="):
            return part.split("=", 1)[1]
    return None


__all__ = ["ConfigurableCLIAgent"]
