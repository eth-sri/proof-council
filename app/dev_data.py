"""Data layer for the dev dashboard.

Pure functions over the on-disk run layout (``outputs/<run-id>/`` per
SPEC §5) and the workflow preset library (``configs/workflows/``).
Kept separate from ``app/dev.py`` so the Flask route layer stays
short and the parsing logic is easy to unit-test if we ever want to.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import pkgutil
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from mathagents.config_loader import CONFIGS_ROOT
from proofstack.dag_schema import DAGReport, build_dag_report
from proofstack.registry import (
    DEFAULT_PRESET_ROOT,
    PresetError,
    WorkflowPreset,
    list_presets,
    load_preset,
)

DEFAULT_TOOL_ROOT = CONFIGS_ROOT / "tools"
WORKFLOW_OUTPUT_NODE_ID = "__workflow_outputs"
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REPEAT_KINDS = {"repeat"}
REPEAT_BODY_MARKER = "::body::"
REPEAT_INPUT_SUFFIX = "::repeat_input"
REPEAT_OUTPUT_SUFFIX = "::repeat_output"
REPEAT_RUNTIME_FIELDS = {"history", "iteration", "iterations", "next_iteration", "reason"}
REMOVED_NODE_TEMPLATES = {"solver", "validator", "improver", "map_chain", "join"}


def _is_repeat_node(node: dict[str, Any] | None) -> bool:
    return bool(node and str(node.get("kind")) in REPEAT_KINDS)


# ---------------------------------------------------------------------------
# Agent introspection (UI-1, moved from dev.py)
# ---------------------------------------------------------------------------

INTROSPECTABLE_CLASS_ATTRS = [
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "MODEL",
    "SANDBOX",
]

@dataclass
class AgentInfo:
    qualname: str
    module: str
    kind: str
    description: str
    execution_mode: str
    inputs_schema: dict[str, Any]
    outputs_schema: dict[str, Any]
    class_attrs: dict[str, Any]
    palette: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    name: str
    yaml_text: str
    python_text: str
    description: str = ""
    descriptor: dict[str, Any] = field(default_factory=dict)
    function_name: str = ""
    error: str | None = None


@dataclass(frozen=True)
class EditorNodeRef:
    kind: str
    node_id: str
    node: dict[str, Any] | None = None
    loop_id: str = ""
    loop: dict[str, Any] | None = None
    body_node_id: str = ""


def discover_agents() -> list[AgentInfo]:
    from proofstack.agent import Agent
    import proofstack.agents as agents_pkg

    for mod_info in pkgutil.walk_packages(agents_pkg.__path__, f"{agents_pkg.__name__}."):
        try:
            importlib.import_module(mod_info.name)
        except Exception:
            continue

    discovered: dict[str, AgentInfo] = {}
    stack: list[type] = [Agent]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            stack.append(sub)
            if sub.__module__.startswith("proofstack.agents."):
                info = _class_to_info(sub)
                discovered[info.qualname] = info
    return sorted(discovered.values(), key=lambda i: (i.execution_mode, i.qualname))


def discover_agent_palette_items() -> list[dict[str, Any]]:
    items = []
    for agent in discover_agents():
        if not agent.palette:
            continue
        agent_path = f"{agent.module}.{agent.qualname}"
        item_id = str(agent.palette.get("id") or _clean_id(agent.qualname) or "python_agent")
        items.append(
            {
                "template": "python_agent",
                "agent": agent_path,
                "node_id": item_id,
                "label": str(agent.palette.get("label") or agent.qualname),
                "group": str(agent.palette.get("group") or "Custom"),
                "description": str(agent.palette.get("description") or agent.description or "Runs this Python agent node."),
                "keywords": str(agent.palette.get("keywords") or f"python agent {agent.qualname}"),
            }
        )
    return sorted(items, key=lambda item: (str(item["group"]).lower(), str(item["label"]).lower()))


def find_agent(qualname: str) -> AgentInfo | None:
    return next((a for a in discover_agents() if a.qualname == qualname), None)


def discover_tool_definitions(root: Path | None = None) -> list[ToolDefinition]:
    root = root or DEFAULT_TOOL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    out: list[ToolDefinition] = []
    for yaml_path in sorted(root.glob("*.yaml")):
        out.append(_read_tool_definition(yaml_path))
    return out


def find_tool_definition(name: str, root: Path | None = None) -> ToolDefinition | None:
    safe = normalize_tool_name(name)
    return next((tool for tool in discover_tool_definitions(root) if tool.name == safe), None)


def create_tool_definition(root: Path | None = None) -> ToolDefinition:
    root = root or DEFAULT_TOOL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    existing = {tool.name for tool in discover_tool_definitions(root)}
    name = "new_tool"
    idx = 2
    while name in existing:
        name = f"new_tool_{idx}"
        idx += 1
    yaml_text = _starter_tool_yaml(name)
    python_text = _starter_tool_python(name)
    _tool_yaml_path(name, root).write_text(yaml_text, encoding="utf-8")
    _tool_python_path(name, root).write_text(python_text, encoding="utf-8")
    return _read_tool_definition(_tool_yaml_path(name, root))


def save_tool_definition(
    old_name: str,
    new_name: str,
    yaml_text: str,
    python_text: str,
    root: Path | None = None,
) -> ToolDefinition:
    root = root or DEFAULT_TOOL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    old_safe = normalize_tool_name(old_name)
    new_safe = _require_tool_name(new_name)
    _validate_tool_yaml_names(yaml_text)
    old_yaml = _tool_yaml_path(old_safe, root)
    old_py = _tool_python_path(old_safe, root)
    new_yaml = _tool_yaml_path(new_safe, root)
    new_py = _tool_python_path(new_safe, root)
    if old_safe != new_safe:
        if new_yaml.exists() or new_py.exists():
            raise PresetError(f"tool already exists: {new_safe}")
        if old_yaml.exists():
            old_yaml.rename(new_yaml)
        if old_py.exists():
            old_py.rename(new_py)
    new_yaml.write_text(yaml_text, encoding="utf-8")
    new_py.write_text(python_text, encoding="utf-8")
    return _read_tool_definition(new_yaml)


def tool_definition_to_dict(tool: ToolDefinition) -> dict[str, Any]:
    return asdict(tool)


def normalize_tool_name(value: str) -> str:
    return _clean_tool_name(value) or "tool"


def _clean_tool_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"^[^A-Za-z_]+", "", cleaned)
    return cleaned


def _require_tool_name(value: str) -> str:
    raw = str(value or "").strip()
    cleaned = _clean_tool_name(raw)
    if not cleaned or raw != cleaned or not TOOL_NAME_RE.fullmatch(cleaned):
        raise PresetError("tool names must use letters, numbers, and underscores, and must start with a letter or underscore")
    return cleaned


def _validate_tool_yaml_names(yaml_text: str) -> None:
    try:
        raw = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        raise PresetError(f"tool YAML parse error: {e}") from e
    if not isinstance(raw, dict):
        raise PresetError("tool YAML must be a mapping")
    candidates = [
        ("YAML tool name", raw.get("name")),
        ("Python function", raw.get("python_function")),
    ]
    descriptor = raw.get("tool") or raw.get("descriptor") or {}
    if isinstance(descriptor, dict):
        fn = descriptor.get("function")
        if isinstance(fn, dict):
            candidates.append(("descriptor function name", fn.get("name")))
    for label, value in candidates:
        if value in (None, ""):
            continue
        name = str(value).strip()
        if not TOOL_NAME_RE.fullmatch(name):
            raise PresetError(f"{label} must use letters, numbers, and underscores, and must start with a letter or underscore")


def _read_tool_definition(yaml_path: Path) -> ToolDefinition:
    name = yaml_path.stem
    yaml_text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    python_path = yaml_path.with_suffix(".py")
    python_text = python_path.read_text(encoding="utf-8") if python_path.exists() else ""
    descriptor: dict[str, Any] = {}
    description = ""
    function_name = ""
    error = None
    try:
        raw = yaml.safe_load(yaml_text) or {}
        if not isinstance(raw, dict):
            raise PresetError("tool YAML must be a mapping")
        description = str(raw.get("description") or "")
        descriptor = raw.get("tool") or raw.get("descriptor") or {}
        if not isinstance(descriptor, dict):
            raise PresetError("tool descriptor must be a mapping")
        function_name = str(raw.get("python_function") or "")
        fn = descriptor.get("function")
        if not function_name and isinstance(fn, dict):
            function_name = str(fn.get("name") or "")
    except Exception as e:
        error = str(e)
        name = yaml_path.stem
    return ToolDefinition(
        name=name,
        yaml_text=yaml_text,
        python_text=python_text,
        description=description,
        descriptor=descriptor,
        function_name=function_name,
        error=error,
    )


def _tool_yaml_path(name: str, root: Path) -> Path:
    return root / f"{normalize_tool_name(name)}.yaml"


def _tool_python_path(name: str, root: Path) -> Path:
    return root / f"{normalize_tool_name(name)}.py"


def _starter_tool_yaml(name: str) -> str:
    return f"""name: {name}
description: Describe what this tool lets the model do.
python_function: {name}
tool:
  type: function
  function:
    name: {name}
    description: Describe this tool for the model.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: Input for the tool.
      required: [query]
"""


def _starter_tool_python(name: str) -> str:
    return f'''def {name}(query: str) -> str:
    """Implement the tool."""
    return query
'''


@dataclass
class RenderedMessages:
    """Best-effort reproduction of what an APICallAgent sent to the model."""
    system: str | None = None
    user: str | None = None
    turns: list[dict[str, str]] = field(default_factory=list)
    source: str = "reconstructed"
    error: str | None = None       # populated if rendering failed


def render_recorded_messages(messages: list[dict[str, Any]] | None) -> RenderedMessages | None:
    """Render the exact messages persisted by ``APICallAgent.run``."""
    if not messages:
        return None
    system_parts: list[str] = []
    user_parts: list[str] = []
    turns: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for msg in messages:
        role = str(msg.get("role") or "message")
        content = _message_text(msg.get("content"))
        if not content:
            continue
        display_role = "system" if role in ("system", "developer") else role
        counts[display_role] = int(counts.get(display_role, 0) or 0) + 1
        turns.append({
            "role": display_role,
            "label": f"{display_role} #{counts[display_role]}",
            "text": content,
        })
        if role in ("system", "developer"):
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    return RenderedMessages(
        system="\n\n---\n\n".join(system_parts) or None,
        user="\n\n---\n\n".join(user_parts) or None,
        turns=turns,
        source="recorded",
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _class_to_info(cls: type) -> AgentInfo:
    inputs = getattr(cls, "Inputs", None)
    outputs = getattr(cls, "Outputs", None)
    try:
        inputs_schema = inputs.model_json_schema() if inputs else {}
    except Exception as e:
        inputs_schema = {"error": str(e)}
    try:
        outputs_schema = outputs.model_json_schema() if outputs else {}
    except Exception as e:
        outputs_schema = {"error": str(e)}

    class_attrs: dict[str, Any] = {}
    for attr in INTROSPECTABLE_CLASS_ATTRS:
        if hasattr(cls, attr):
            raw = getattr(cls, attr)
            if callable(raw):
                continue
            class_attrs[attr] = _format_value(raw)
    raw_palette = getattr(cls, "PALETTE", None)
    palette = dict(raw_palette) if isinstance(raw_palette, dict) else {}

    kind = "Agent"
    for base in cls.__mro__[1:]:
        if base.__name__ in {"APICallAgent", "MultiTurnAgent", "CLIAgent"}:
            kind = base.__name__
            break

    return AgentInfo(
        qualname=cls.__qualname__,
        module=cls.__module__,
        kind=kind,
        description=getattr(cls, "description", "") or "",
        execution_mode=str(getattr(cls, "execution_mode", "agent")),
        inputs_schema=inputs_schema,
        outputs_schema=outputs_schema,
        class_attrs=class_attrs,
        palette=palette,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return ", ".join(str(v) for v in value) or "(empty)"
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return repr(value)


# ---------------------------------------------------------------------------
# Run discovery (UI-0)
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    run_id: str           # filesystem dir name; URL key
    path: Path            # absolute path to the run dir
    display_name: str = ""
    started_at: str | None = None
    status: str | None = None
    preset: str | None = None
    problem_summary: str | None = None
    cost_usd: float | None = None
    wallclock_s: float | None = None
    n_problems: int | None = None
    has_events: bool = False
    monitor_enabled: bool = False
    problems: dict[str, Any] = field(default_factory=dict)


def discover_runs(roots: Iterable[Path], *, include_batch_children: bool = False) -> list[RunInfo]:
    """Walk one or more run roots, returning RunInfo entries newest-first.

    A root is treated as a *single run dir* if it directly contains
    ``events.jsonl`` or ``run-metadata.json``; otherwise its immediate
    subdirectories are scanned for run dirs. Run-id collisions across
    roots are resolved first-seen-wins.
    """
    seen: dict[str, RunInfo] = {}
    for root in roots:
        if not root.exists():
            continue
        candidates: list[Path]
        if _looks_like_run(root):
            candidates = [root]
        else:
            candidates = [p for p in root.iterdir() if p.is_dir() and _looks_like_run(p)]
        for path in candidates:
            run_id = path.name
            if run_id in seen:
                continue
            seen[run_id] = _read_run_info(path)
    _aggregate_batch_runs(seen)
    if not include_batch_children:
        for run_id in _batch_child_run_ids(seen.values()):
            seen.pop(run_id, None)
    return sorted(seen.values(), key=lambda r: r.path.stat().st_mtime, reverse=True)


def _looks_like_run(path: Path) -> bool:
    return (path / "events.jsonl").exists() or (path / "run-metadata.json").exists()


def _read_run_info(path: Path) -> RunInfo:
    info = RunInfo(
        run_id=path.name,
        path=path,
        has_events=(path / "events.jsonl").exists(),
    )
    meta_path = path / "run-metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        info.display_name = str(meta.get("display_name") or meta.get("run_name") or "").strip()
        info.status = _normalize_run_status(meta.get("status"))
        info.preset = str(meta.get("preset") or "").strip() or None
        monitor = meta.get("monitor") or {}
        if not isinstance(monitor, dict):
            monitor = {}
        config_snapshot = meta.get("config_snapshot") or {}
        if not monitor and isinstance(config_snapshot, dict):
            snapshot_monitor = config_snapshot.get("monitor") or {}
            if isinstance(snapshot_monitor, dict):
                monitor = snapshot_monitor
        info.monitor_enabled = bool(monitor.get("enabled"))
        info.cost_usd = meta.get("cost_usd")
        info.wallclock_s = meta.get("wallclock_s")
        config_snapshot = meta.get("config_snapshot") or {}
        if isinstance(config_snapshot, dict):
            info.preset = info.preset or str(config_snapshot.get("preset") or "").strip() or None
            problem_id = str(config_snapshot.get("problem_id") or "").strip()
            if problem_id and not info.problem_summary:
                info.problem_summary = _humanize_problem_id(problem_id)
                info.n_problems = info.n_problems or 1
        manifest = meta.get("manifest") or {}
        if not isinstance(manifest, dict):
            manifest = {}
        info.started_at = manifest.get("started_at") or meta.get("started_at")
        if isinstance(manifest.get("totals"), dict):
            totals = manifest["totals"]
            info.cost_usd = totals.get("cost_usd", info.cost_usd)
            info.wallclock_s = totals.get("wallclock_s", info.wallclock_s)
        if isinstance(manifest, dict):
            info.preset = info.preset or str(manifest.get("preset") or "").strip() or None
        finished_at = manifest.get("finished_at") or meta.get("finished_at")
        if info.wallclock_s is None and info.started_at and finished_at:
            info.wallclock_s = _duration(info.started_at, str(finished_at))
        problems = manifest.get("problems") or {}
        if isinstance(problems, dict) and problems:
            info.problems = problems
            info.n_problems = len(problems)
            statuses = [p.get("status") for p in problems.values() if isinstance(p, dict)]
            problem_status = _status_from_problem_statuses(statuses)
            if problem_status in {"error", "running"} or info.status is None:
                info.status = problem_status
            info.problem_summary = _problem_summary(problems)
        _enrich_from_dashboard_subprocess_log(info, finished_at=finished_at)
    # Fallback / fill-in from events.jsonl. ``run_workflow.py`` runs do
    # not produce a manifest block, so totals + timestamps come straight
    # from the event log. Cheap O(n) scan, ~kB-sized files.
    if info.has_events:
        _enrich_from_events(info)
    _finalize_run_info(info)
    return info


def _batch_child_run_ids(infos: Iterable[RunInfo]) -> set[str]:
    child_ids: set[str] = set()
    for info in infos:
        for problem in info.problems.values():
            if isinstance(problem, dict):
                run_id = str(problem.get("run_id") or "").strip()
                if run_id:
                    child_ids.add(run_id)
    return child_ids


def _aggregate_batch_runs(seen: dict[str, RunInfo]) -> None:
    for info in seen.values():
        if not info.problems:
            continue
        total_cost = 0.0
        saw_cost = False
        for problem in info.problems.values():
            if not isinstance(problem, dict):
                continue
            child = seen.get(str(problem.get("run_id") or ""))
            problem["has_run"] = child is not None
            if child is None:
                continue
            if child.cost_usd is not None:
                total_cost += float(child.cost_usd)
                saw_cost = True
            if not problem.get("started_at") and child.started_at:
                problem["started_at"] = child.started_at
            if child.status:
                problem["status"] = child.status
        if saw_cost:
            info.cost_usd = total_cost


def _enrich_from_dashboard_subprocess_log(info: RunInfo, *, finished_at: Any) -> None:
    if finished_at or info.status != "running":
        return
    message = _dashboard_subprocess_failure_message(info.path / "dashboard-subprocess.log")
    if not message:
        return
    info.status = "error"
    for problem in info.problems.values():
        if not isinstance(problem, dict):
            continue
        if _normalize_run_status(problem.get("status")) == "running":
            problem["status"] = "error"
            problem["error"] = message


def _dashboard_subprocess_failure_message(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    lower = text.lower()
    markers = (
        "traceback (most recent call last):",
        "modulenotfounderror",
        "syntaxerror",
        "systemexit",
        "empty latex",
        "empty problem text",
    )
    if not any(marker in lower for marker in markers):
        return ""
    return lines[-1][:500]


def _enrich_from_events(info: RunInfo) -> None:
    cost = 0.0
    first_ts: str | None = None
    last_ts: str | None = None
    saw_workflow_failure = False
    try:
        with (info.path / "events.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = e.get("ts")
                if ts:
                    first_ts = first_ts or ts
                    last_ts = ts
                if e.get("kind") == "run.start":
                    if info.status is None:
                        info.status = "running"
                    payload = e.get("payload") or {}
                    if isinstance(payload, dict):
                        info.preset = info.preset or str(payload.get("preset") or "").strip() or None
                        display_name = str(payload.get("display_name") or "").strip()
                        if display_name and not info.display_name:
                            info.display_name = display_name
                        problem_id = str(payload.get("problem_id") or "").strip()
                        inputs = payload.get("inputs") or {}
                        if not problem_id and isinstance(inputs, dict):
                            problem_id = str(inputs.get("problem_id") or "").strip()
                        if problem_id and not info.problem_summary:
                            info.problem_summary = _humanize_problem_id(problem_id)
                            info.n_problems = info.n_problems or 1
                if e.get("kind") == "run.end":
                    payload = e.get("payload") or {}
                    if isinstance(payload, dict):
                        info.status = _normalize_run_status(payload.get("status")) or info.status
                if e.get("kind") == "workflow.last_gasp":
                    saw_workflow_failure = True
                if e.get("kind") == "model.call":
                    p = e.get("payload") or {}
                    cost += float(p.get("cost_usd") or 0.0)
    except OSError:
        return
    if info.started_at is None and first_ts is not None:
        info.started_at = first_ts
    if info.cost_usd is None and cost:
        info.cost_usd = cost
    if info.wallclock_s is None and first_ts and last_ts:
        info.wallclock_s = _duration(first_ts, last_ts)
    if saw_workflow_failure:
        info.status = "error"


def _finalize_run_info(info: RunInfo) -> None:
    if info.status is None:
        info.status = "running" if info.has_events else "finished"
    if info.display_name:
        info.display_name = _dedupe_display_name(info.display_name)
        return
    preset_label = _preset_label(info.preset)
    if preset_label and info.problem_summary:
        info.display_name = f"{preset_label} · {info.problem_summary}"
    elif preset_label:
        info.display_name = preset_label
    elif info.started_at:
        info.display_name = f"ProofCouncil Run · {_short_local_time(info.started_at)}"
    else:
        info.display_name = _humanize_problem_id(info.run_id)
    info.display_name = _dedupe_display_name(info.display_name)


def _dedupe_display_name(value: str) -> str:
    parts = [part.strip() for part in str(value or "").split("·")]
    if len(parts) < 2:
        return str(value or "").strip()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if out and out[-1] == part:
            continue
        out.append(part)
    return " · ".join(out)


def _problem_summary(problems: dict[str, Any]) -> str:
    if len(problems) == 1:
        problem = next(iter(problems.values()))
        if isinstance(problem, dict):
            return str(
                problem.get("display_name")
                or problem.get("title")
                or _humanize_problem_id(str(problem.get("problem_id") or "problem"))
            )
        return _humanize_problem_id(str(next(iter(problems.keys()))))
    return f"{len(problems)} problems"


def _normalize_run_status(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if raw in {"ok", "finished", "success", "succeeded", "complete", "completed", "done"}:
        return "finished"
    if raw in {"error", "failed", "failure", "cancelled", "canceled"}:
        return "error"
    if raw in {"running", "starting", "started", "queued", "pending"}:
        return "running"
    return None


def _status_from_problem_statuses(statuses: list[Any]) -> str | None:
    normalized = [_normalize_run_status(status) for status in statuses if status]
    if not normalized:
        return None
    if "error" in normalized:
        return "error"
    if "running" in normalized:
        return "running"
    if all(status == "finished" for status in normalized):
        return "finished"
    return None


def _preset_label(name: str | None) -> str:
    if not name:
        return ""
    path = DEFAULT_PRESET_ROOT / f"{name}.yaml"
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        if isinstance(raw, dict):
            label = _preset_display_label(raw, name)
            if label:
                return label
    return _humanize_problem_id(name)


def _humanize_problem_id(value: str) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "").strip()).strip()
    return text.title() if text else "Problem"


def _short_local_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    return f"{dt:%b} {dt.day} {dt:%H:%M}"


def find_run(roots: Iterable[Path], run_id: str) -> RunInfo | None:
    for r in discover_runs(roots, include_batch_children=True):
        if r.run_id == run_id:
            return r
    return None


def load_pending_human_tasks(run_path: Path) -> list[dict[str, Any]]:
    """Human-in-the-loop tasks awaiting a response in this run.

    A task is pending when a ``human.waiting`` event has no matching
    ``human.submitted``/``human.timeout`` and its response file hasn't been
    written yet. Each task carries the rendered prompt, the node's inputs,
    the expected output fields, and the response filename to write back to.
    """
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []
    waiting: dict[str, dict[str, Any]] = {}
    resolved: set[str] = set()
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = e.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                response_path = str(payload.get("response_path") or "")
                if not response_path:
                    continue
                kind = e.get("kind")
                if kind == "human.waiting":
                    waiting[response_path] = {**payload, "agent": e.get("agent")}
                elif kind in ("human.submitted", "human.timeout"):
                    resolved.add(response_path)
    except OSError:
        return []

    inbox = run_path / "human_inbox"
    tasks: list[dict[str, Any]] = []
    for response_path, payload in waiting.items():
        if response_path in resolved:
            continue
        filename = Path(response_path).name
        if (inbox / filename).exists():
            continue  # response already dropped; run will resume
        # Read display fields from the on-disk task.json, NOT the event payload:
        # a large prompt (e.g. one embedding an AI proof) gets offloaded to a blob
        # in events.jsonl and shows up as {"$ref": ...}, whereas HumanAgent writes
        # the full prompt straight to task.json.
        full: dict[str, Any] = {}
        task_path = payload.get("task_path")
        if task_path and Path(task_path).exists():
            try:
                loaded = json.loads(Path(task_path).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    full = loaded
            except (json.JSONDecodeError, OSError):
                full = {}
        prompt = full.get("prompt")
        if not isinstance(prompt, str):
            prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else ""
        output_fields = full.get("output_fields") or payload.get("output_fields")
        if not isinstance(output_fields, dict) or not output_fields:
            output_fields = {"response": "string"}
        tasks.append(
            {
                "agent": full.get("agent") or payload.get("agent"),
                "prompt": prompt,
                "output_fields": output_fields,
                "inputs": full.get("inputs") or {},
                "response_filename": filename,
            }
        )
    return tasks


def find_runs_awaiting_human(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Every run currently blocked on human input, across all run roots.

    Used by the global "waiting on you" nav indicator. A run can have more
    than one pending ask (loops), so each entry carries a count. Only
    non-terminal runs are scanned (a finished/errored run cannot be waiting),
    which keeps this cheap enough to call on every page render. Newest first.
    """
    out: list[dict[str, Any]] = []
    for run in discover_runs(roots, include_batch_children=True):
        if (run.status or "running") in {"finished", "error"}:
            continue
        tasks = load_pending_human_tasks(run.path)
        if not tasks:
            continue
        out.append(
            {
                "run_id": run.run_id,
                "display_name": run.display_name or run.run_id,
                "count": len(tasks),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Event tree (UI-0)
# ---------------------------------------------------------------------------


@dataclass
class CallNode:
    """One agent invocation reconstructed from agent.start..agent.end events."""

    call_id: str
    display_ref: str = ""
    display_label: str | None = None
    agent: str | None = None
    agent_path: str | None = None
    parent_call_id: str | None = None
    execution_mode: str | None = None
    status: str = "running"     # running | ok | error | unknown
    start_ts: str | None = None
    end_ts: str | None = None
    duration_s: float | None = None
    input: Any = None
    output: Any = None
    error: dict[str, Any] | None = None
    cache_hit: bool = False
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    children: list["CallNode"] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.display_label:
            return self.display_label
        if self.agent_path:
            return self.agent_path
        return self.agent or "(anonymous)"


@dataclass
class RunEventTree:
    run_id: str
    roots: list[CallNode]
    by_id: dict[str, CallNode]
    by_ref: dict[str, CallNode]


@dataclass
class ExecutionGraphNode:
    """One workflow DAG node plus its observed runtime status."""

    node_id: str
    raw_id: str
    label: str
    subtitle: str = ""
    kind: str = "agent"
    status: str = "pending"
    start_ts: str | None = None
    end_ts: str | None = None
    duration_s: float | None = None
    call_id: str | None = None
    call_ref: str | None = None
    cost_usd: float = 0.0
    reason: str = ""
    execution_index: int = 1
    parent_node_id: str = ""
    children: list["ExecutionGraphNode"] = field(default_factory=list)


@dataclass
class ExecutionGraph:
    run_id: str
    roots: list[ExecutionGraphNode]
    by_id: dict[str, ExecutionGraphNode]
    preset: str | None = None


RUN_LEVEL_EVENT_KINDS: frozenset[str] = frozenset({
    "run.start",
    "run.end",
    "run.budget.warn",
    "run.config",
})


def load_event_tree(run_path: Path) -> RunEventTree:
    """Stream events.jsonl and reconstruct the call tree."""
    events_path = run_path / "events.jsonl"
    by_id: dict[str, CallNode] = {}
    run_status = ""
    run_end_ts: str | None = None
    if not events_path.exists():
        return RunEventTree(
            run_id=run_path.name,
            roots=[],
            by_id=by_id,
            by_ref={},
        )

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = evt.get("kind")
            call_id = evt.get("call_id")
            parent_id = evt.get("parent_call_id")

            if kind == "run.end":
                payload = evt.get("payload") or {}
                if isinstance(payload, dict):
                    run_status = _normalize_run_status(payload.get("status")) or str(payload.get("status") or "")
                run_end_ts = evt.get("ts")

            if kind in RUN_LEVEL_EVENT_KINDS:
                continue

            if kind == "agent.start" and call_id:
                node = by_id.setdefault(call_id, CallNode(call_id=call_id))
                node.agent = evt.get("agent")
                node.agent_path = evt.get("agent_path")
                node.parent_call_id = parent_id
                node.execution_mode = evt.get("execution_mode")
                node.start_ts = evt.get("ts")
                payload = evt.get("payload") or {}
                node.input = payload.get("input")
            elif kind == "agent.end" and call_id:
                node = by_id.setdefault(call_id, CallNode(call_id=call_id))
                node.end_ts = evt.get("ts")
                payload = evt.get("payload") or {}
                node.output = payload.get("output")
                output_status = _call_status_from_output(node.output)
                if output_status == "error":
                    node.status = "error"
                    node.error = _call_error_from_output(node.output)
                else:
                    node.status = "ok" if node.status == "running" else node.status
                node.duration_s = _duration(node.start_ts, node.end_ts)
            elif kind == "agent.error" and call_id:
                node = by_id.setdefault(call_id, CallNode(call_id=call_id))
                node.status = "error"
                node.end_ts = evt.get("ts")
                node.duration_s = _duration(node.start_ts, node.end_ts)
                node.error = evt.get("payload") or {}
            elif kind == "workflow.last_gasp":
                target = parent_id or call_id
                if target:
                    node = by_id.setdefault(str(target), CallNode(call_id=str(target)))
                    node.status = "error"
                    node.end_ts = evt.get("ts")
                    node.duration_s = _duration(node.start_ts, node.end_ts)
                    node.error = evt.get("payload") or {}
            elif kind == "agent.cache_hit" and call_id:
                # Cache hits skip agent.start/agent.end (see Agent.__call__),
                # so this is the only event carrying the call's identity.
                # Pull all the parenting/identity fields from it so the
                # cached call still slots under its real parent in the tree
                # rather than orphaning as an anonymous "running" root.
                node = by_id.setdefault(call_id, CallNode(call_id=call_id))
                node.cache_hit = True
                node.agent = node.agent or evt.get("agent")
                node.agent_path = node.agent_path or evt.get("agent_path")
                node.parent_call_id = node.parent_call_id or parent_id
                node.execution_mode = node.execution_mode or evt.get("execution_mode")
                node.start_ts = node.start_ts or evt.get("ts")
                node.end_ts = node.end_ts or evt.get("ts")
                if node.status == "running":
                    node.status = "ok"
            elif kind == "model.call":
                # parent_call_id of a model.call points at the calling agent
                target = parent_id or call_id
                if target:
                    node = by_id.setdefault(target, CallNode(call_id=target))
                    payload = evt.get("payload") or {}
                    node.model_calls.append({"ts": evt.get("ts"), **payload})
                    node.cost_usd += float(payload.get("cost_usd") or 0.0)
                    node.in_tokens += int(payload.get("in_tokens") or 0)
                    node.out_tokens += int(payload.get("out_tokens") or 0)
            elif kind == "model.empty_response":
                target = parent_id or call_id
                if target:
                    node = by_id.setdefault(target, CallNode(call_id=target))
                    node.status = "error"
                    node.error = evt.get("payload") or {}
            elif kind in ("tool.call", "tool.result"):
                target = parent_id or call_id
                if target:
                    node = by_id.setdefault(target, CallNode(call_id=target))
                    node.tool_calls.append({"ts": evt.get("ts"), "kind": kind, **(evt.get("payload") or {})})
            elif kind in ("model.call.start",):
                # Span-marker; ignore for tree building
                continue

    if run_status == "error":
        for node in by_id.values():
            if node.status != "running":
                continue
            node.status = "error"
            node.end_ts = node.end_ts or run_end_ts
            node.duration_s = _duration(node.start_ts, node.end_ts)
            node.error = node.error or {
                "type": "IncompleteCall",
                "msg": "Run ended with an error before this call emitted a completion event.",
            }

    # Build parent->children links (sorted by start_ts so the tree is stable)
    roots: list[CallNode] = []
    for node in by_id.values():
        if node.parent_call_id and node.parent_call_id in by_id:
            by_id[node.parent_call_id].children.append(node)
        else:
            roots.append(node)
    for node in by_id.values():
        node.children.sort(key=lambda c: c.start_ts or "")

    for root in roots:
        _rollup_cost(root)

    roots.sort(key=lambda c: c.start_ts or "")
    by_ref = _assign_call_refs(roots)
    return RunEventTree(
        run_id=run_path.name,
        roots=roots,
        by_id=by_id,
        by_ref=by_ref,
    )


def load_monitor_summaries(run_path: Path, *, graph: ExecutionGraph | None = None) -> list[dict[str, Any]]:
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []
    labels_by_call_id = _monitor_labels_by_call_id(graph)
    summaries: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "monitor.summary":
                continue
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            call_id = payload.get("call_id") or event.get("parent_call_id")
            summaries.append(
                {
                    "ts": event.get("ts"),
                    "agent": payload.get("agent"),
                    "display_label": (
                        payload.get("display_label")
                        or labels_by_call_id.get(str(call_id or ""))
                        or _monitor_fallback_label(payload.get("agent"), payload.get("agent_path"))
                    ),
                    "call_id": call_id,
                    "status": payload.get("status"),
                    "summary": payload.get("summary"),
                }
            )
    return summaries


def _monitor_labels_by_call_id(graph: ExecutionGraph | None) -> dict[str, str]:
    if graph is None:
        return {}
    labels: dict[str, str] = {}
    for node in graph.by_id.values():
        if node.call_id and node.label:
            labels[str(node.call_id)] = node.label
    return labels


def _monitor_fallback_label(agent: Any, agent_path: Any) -> str:
    if str(agent or "") == "DAGWorkflow" or str(agent_path or "").endswith("DAGWorkflow"):
        return "Workflow"
    raw = str(agent or agent_path or "Monitor update").rsplit(".", 1)[-1]
    return _humanize_id(raw)


def workflow_output_from_tree(tree: RunEventTree) -> Any:
    roots = [node for node in tree.roots if node.output is not None]
    if not roots:
        return None
    workflow_roots = [
        node
        for node in roots
        if node.execution_mode == "workflow"
        or node.agent == "DAGWorkflow"
        or str(node.agent_path or "").endswith("DAGWorkflow")
    ]
    node = (workflow_roots or roots)[-1]
    return _clean_display_output(node.output)


def workflow_input_from_tree(tree: RunEventTree) -> Any:
    roots = [node for node in tree.roots if node.input is not None]
    if not roots:
        return None
    workflow_roots = [
        node
        for node in roots
        if node.execution_mode == "workflow"
        or node.agent == "DAGWorkflow"
        or str(node.agent_path or "").endswith("DAGWorkflow")
    ]
    node = (workflow_roots or roots)[0]
    return _clean_display_output(node.input)


def _clean_display_output(value: Any) -> Any:
    return _clean_display_output_value(value)


def _clean_display_output_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for child_key, item in value.items():
            if child_key == "raw_text":
                continue
            clean_item = _clean_display_output_value(item, key=str(child_key))
            if _is_display_empty(clean_item):
                continue
            cleaned[child_key] = clean_item
        return cleaned
    if isinstance(value, list):
        return [
            item
            for item in (_clean_display_output_value(item) for item in value)
            if not _is_display_empty(item)
        ]
    if isinstance(value, str) and _is_internal_path(value) and not _is_display_path_field(key):
        return None
    return value


def _is_display_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _is_internal_path(value: str) -> bool:
    return value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _is_display_path_field(key: str | None) -> bool:
    if not key:
        return False
    normalized = key.lower()
    return normalized == "path" or normalized.endswith("_path") or normalized in {
        "answer_tex",
        "solution_tex",
    }


def _assign_call_refs(roots: list[CallNode]) -> dict[str, CallNode]:
    by_ref: dict[str, CallNode] = {}
    next_ref = 1

    def visit(node: CallNode) -> None:
        nonlocal next_ref
        ref = str(next_ref)
        next_ref += 1
        node.display_ref = ref
        by_ref[ref] = node
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return by_ref


def load_execution_graph(
    run_path: Path,
    *,
    tree: RunEventTree | None = None,
    preset_root: Path | None = None,
) -> ExecutionGraph:
    events = _read_jsonl_events(run_path / "events.jsonl")
    preset_name = _run_preset_name(events)
    roots, by_id, aliases = _execution_nodes_from_preset(preset_name, preset_root)
    active_by_parent: dict[str, list[ExecutionGraphNode]] = {}
    active_events: dict[tuple[str, str], list[ExecutionGraphNode]] = {}
    call_nodes: dict[str, ExecutionGraphNode] = {}
    recent_completed_nodes: dict[tuple[str, str], list[ExecutionGraphNode]] = {}
    workflow_call_nodes: dict[str, ExecutionGraphNode] = {}
    execution_counts: dict[tuple[str, str], int] = {}
    repeat_child_offsets: dict[tuple[str, str], int] = {}
    run_status = None

    for evt in events:
        kind = evt.get("kind")
        payload = evt.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        if kind == "run.end":
            run_status = str(payload.get("status") or "")
        if not str(kind or "").startswith("dag.node_"):
            if kind == "agent.start":
                parent_id = evt.get("parent_call_id")
                if parent_id:
                    attached = _attach_call_to_active_node(
                        active_by_parent.get(str(parent_id), []),
                        str(evt.get("call_id") or ""),
                    )
                    call_id = str(evt.get("call_id") or "")
                    if attached and call_id:
                        call_nodes[call_id] = attached
                        if evt.get("execution_mode") == "workflow":
                            workflow_call_nodes[call_id] = attached
            elif kind in {"agent.end", "agent.error"}:
                call_id = str(evt.get("call_id") or "")
                parent_id = str(evt.get("parent_call_id") or "")
                node = call_nodes.get(call_id)
                if node is not None:
                    _apply_agent_terminal_event_to_execution_graph_node(node, kind, payload, evt.get("ts"))
                if node is not None and parent_id:
                    recent_completed_nodes.setdefault((parent_id, node.raw_id), []).append(node)
            continue

        ts = evt.get("ts")
        parent_id = str(evt.get("parent_call_id") or "")
        if kind == "dag.node_started":
            parent_node = workflow_call_nodes.get(parent_id) or _repeat_parent_for_body_event(
                active_by_parent,
                parent_id,
                payload,
                repeat_child_offsets,
            )
            node = _execution_node_for_payload(
                payload,
                roots,
                by_id,
                aliases,
                parent_node=parent_node,
                for_start=True,
                execution_counts=execution_counts,
            )
        else:
            node = _pop_recent_completed_node(
                recent_completed_nodes,
                active_events,
                parent_id,
                payload,
            )
            if node is None:
                node = _pop_active_event_node(active_events, parent_id, payload)
            if node is None:
                node = _execution_node_for_payload(
                    payload,
                    roots,
                    by_id,
                    aliases,
                    parent_node=workflow_call_nodes.get(parent_id),
                )
        if node is None:
            continue
        _apply_runtime_node_metadata(node, payload)
        if kind == "dag.node_started":
            node.status = "running"
            node.start_ts = node.start_ts or ts
            if parent_id:
                active_by_parent.setdefault(parent_id, []).append(node)
                active_events.setdefault(_active_event_key(parent_id, payload), []).append(node)
        elif kind == "dag.node_done":
            if node.status != "error":
                node.status = "ok"
            node.end_ts = ts
            node.duration_s = _duration(node.start_ts, node.end_ts)
            _drop_active_node(active_by_parent.get(parent_id, []), node)
        elif kind == "dag.node_skipped":
            node.status = "skipped"
            node.end_ts = ts
            _drop_active_node(active_by_parent.get(parent_id, []), node)
        elif kind == "dag.node_pruned":
            node.status = "skipped"
            node.end_ts = ts
            node.reason = str(payload.get("reason") or "Skipped by branch")
            _drop_active_node(active_by_parent.get(parent_id, []), node)
        elif kind in {"dag.node_error", "dag.node_budget_exhausted"}:
            node.status = "error"
            node.end_ts = ts
            node.reason = str(payload.get("msg") or payload.get("kind") or "Node failed")
            _drop_active_node(active_by_parent.get(parent_id, []), node)

    _add_nested_workflow_progress(
        events,
        roots,
        by_id,
        workflow_call_nodes=workflow_call_nodes,
        call_nodes=call_nodes,
    )

    if run_status == "error":
        for node in by_id.values():
            if node.status == "running":
                node.status = "error"
                node.reason = node.reason or "Run ended before this node finished."

    if tree is not None:
        for node in by_id.values():
            if not node.call_id:
                continue
            call = tree.by_id.get(node.call_id)
            if call is None:
                continue
            node.call_ref = call.display_ref or node.call_id
            call.display_label = node.label
            node.cost_usd = float(getattr(call, "cost_usd_subtree", call.cost_usd) or 0.0)
            if node.duration_s is None:
                node.duration_s = call.duration_s
            if call.status == "error":
                node.status = "error"
                node.end_ts = node.end_ts or call.end_ts
                node.reason = node.reason or _call_error_message(call)

    return ExecutionGraph(
        run_id=run_path.name,
        roots=roots,
        by_id=by_id,
        preset=preset_name,
    )


def _add_nested_workflow_progress(
    events: list[dict[str, Any]],
    roots: list[ExecutionGraphNode],
    by_id: dict[str, ExecutionGraphNode],
    *,
    workflow_call_nodes: dict[str, ExecutionGraphNode],
    call_nodes: dict[str, ExecutionGraphNode],
) -> None:
    active: dict[str, ExecutionGraphNode] = {}
    current_round: dict[str, ExecutionGraphNode] = {}
    for evt in events:
        kind = str(evt.get("kind") or "")
        parent_id = str(evt.get("parent_call_id") or "")
        payload = evt.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        ts = evt.get("ts")
        workflow_node = workflow_call_nodes.get(parent_id)
        if kind == "ac.round_start" and workflow_node is not None:
            previous = current_round.get(parent_id)
            if previous is not None and previous.status == "running":
                previous.status = "ok"
                previous.end_ts = ts
                previous.duration_s = _duration(previous.start_ts, previous.end_ts)
            round_value = payload.get("round")
            label = f"Round {round_value}" if round_value is not None else "Round"
            node = ExecutionGraphNode(
                node_id=_unique_runtime_node_id(f"ac_round_{round_value}", by_id, parent_id=workflow_node.node_id),
                raw_id=f"round_{round_value}",
                label=label,
                kind="ac_round",
                status="running",
                start_ts=ts,
                parent_node_id=workflow_node.node_id,
            )
            _attach_execution_graph_node(node, roots, by_id, workflow_node)
            current_round[parent_id] = node
            continue
        if kind.startswith("ac.") and workflow_node is not None:
            milestone = _ac_milestone_node(kind, payload, ts, workflow_node, current_round.get(parent_id), roots, by_id)
            if kind in {"ac.final_compile", "ac.last_gasp", "ac.resume_noop"}:
                round_node = current_round.get(parent_id)
                if round_node is not None and round_node.status == "running":
                    round_node.status = "ok" if kind != "ac.last_gasp" else "error"
                    round_node.end_ts = ts
                    round_node.duration_s = _duration(round_node.start_ts, round_node.end_ts)
            if milestone is not None:
                continue
        if kind == "agent.start" and workflow_node is not None:
            call_id = str(evt.get("call_id") or "")
            if not call_id or call_id in call_nodes:
                continue
            parent = _ac_figure_parent(evt, by_id) or current_round.get(parent_id) or workflow_node
            if _is_ac_figure_block(parent):
                _mark_ac_figure_block_started(parent, ts)
            node = ExecutionGraphNode(
                node_id=_unique_runtime_node_id(_nested_agent_node_id(evt), by_id, parent_id=parent.node_id),
                raw_id=_nested_agent_raw_id(evt),
                label=_nested_agent_label(evt),
                kind=str(evt.get("execution_mode") or "agent"),
                status="running",
                start_ts=ts,
                call_id=call_id,
                parent_node_id=parent.node_id,
            )
            _attach_execution_graph_node(node, roots, by_id, parent)
            active[call_id] = node
            call_nodes[call_id] = node
        elif kind in {"agent.end", "agent.error"}:
            call_id = str(evt.get("call_id") or "")
            if call_id in workflow_call_nodes:
                status, reason = _agent_terminal_status_and_reason(kind, payload)
                _finish_current_ac_round(current_round, call_id, ts, status=status, reason=reason)
            node = active.get(call_id)
            if node is None:
                continue
            _apply_agent_terminal_event_to_execution_graph_node(node, kind, payload, ts)
            parent_node = by_id.get(node.parent_node_id)
            if _is_ac_figure_block(parent_node):
                _mark_ac_figure_block_finished(parent_node, node, ts)


def _ac_milestone_node(
    kind: str,
    payload: dict[str, Any],
    ts: str | None,
    workflow_node: ExecutionGraphNode,
    round_node: ExecutionGraphNode | None,
    roots: list[ExecutionGraphNode],
    by_id: dict[str, ExecutionGraphNode],
) -> ExecutionGraphNode | None:
    labels = {
        "ac.resume": "Resume checkpoint",
        "ac.resume_noop": "Resume complete",
        "ac.resume_missing_review_start": "Resume missing review",
        "ac.terminal_auxiliary_suppressed": "Terminal auxiliary suppressed",
        "ac.early_stop_deferred_for_compute": "Early stop deferred for compute",
        "ac.early_stop_deferred_for_terminal_auxiliary": "Early stop deferred",
        "ac.forced_fresh_review": "Forced-fresh critic",
        "ac.early_stop_agreed": "Early stop agreed",
        "ac.deterministic_gate_blocked": "Deterministic gate blocked",
        "ac.final_compile": "Final compile",
        "ac.final_critic_review": "Final critic",
        "ac.final_critic_skipped": "Final critic skipped",
        "ac.final_critic_failed": "Final critic failed",
        "ac.last_gasp": "Last-gasp recovery",
    }
    label = labels.get(kind)
    if label is None:
        return None
    parent = round_node if kind not in {"ac.final_compile", "ac.final_critic_review", "ac.final_critic_skipped", "ac.final_critic_failed", "ac.last_gasp", "ac.resume", "ac.resume_noop"} and round_node is not None else workflow_node
    status = "ok"
    if kind in {"ac.final_critic_skipped"}:
        status = "skipped"
    if kind in {"ac.final_critic_failed", "ac.last_gasp"}:
        status = "error"
    reason = _ac_milestone_reason(kind, payload)
    node = ExecutionGraphNode(
        node_id=_unique_runtime_node_id(kind.replace(".", "_"), by_id, parent_id=parent.node_id),
        raw_id=kind.replace(".", "_"),
        label=label,
        kind="ac_event",
        status=status,
        start_ts=ts,
        end_ts=ts,
        duration_s=0.0 if ts else None,
        reason=reason,
        parent_node_id=parent.node_id,
    )
    _attach_execution_graph_node(node, roots, by_id, parent)
    return node


def _ac_milestone_reason(kind: str, payload: dict[str, Any]) -> str:
    if kind == "ac.final_compile":
        if "pages" in payload:
            return f"pages: {payload.get('pages')} / {payload.get('page_limit')}"
        return ""
    if kind == "ac.last_gasp":
        return str(payload.get("msg") or payload.get("type") or "Workflow recovered after an error")
    if kind == "ac.final_critic_failed":
        return str(payload.get("msg") or payload.get("type") or "Final critic failed")
    if kind == "ac.final_critic_skipped":
        return str(payload.get("reason") or "Skipped")
    if kind == "ac.deterministic_gate_blocked":
        reasons = payload.get("reasons")
        if isinstance(reasons, list):
            return ", ".join(str(item) for item in reasons)
    return ""


_AC_FIGURE_NODE_IDS = {
    "author",
    "stateful_critic",
    "fresh_critic",
    "llm_council",
    "compute_node",
}


def _ac_figure_parent(
    evt: dict[str, Any],
    by_id: dict[str, ExecutionGraphNode],
) -> ExecutionGraphNode | None:
    raw = str(evt.get("agent") or evt.get("agent_path") or "").rsplit(".", 1)[-1]
    node_id = ""
    if raw == "Author":
        node_id = "author"
    elif raw == "ACCritic":
        payload = evt.get("payload") if isinstance(evt.get("payload"), dict) else {}
        call_input = payload.get("input") if isinstance(payload, dict) else {}
        if not isinstance(call_input, dict):
            call_input = {}
        mode = str(call_input.get("mode") or "").lower()
        omit_thinking = bool(call_input.get("omit_author_thinking"))
        node_id = "fresh_critic" if mode == "fresh" or omit_thinking else "stateful_critic"
    elif raw == "Council":
        node_id = "llm_council"
    elif raw == "Compute":
        node_id = "compute_node"
    if not node_id:
        return None
    return by_id.get(node_id)


def _is_ac_figure_block(node: ExecutionGraphNode | None) -> bool:
    return node is not None and node.raw_id in _AC_FIGURE_NODE_IDS


def _mark_ac_figure_block_started(node: ExecutionGraphNode, ts: str | None) -> None:
    node.status = "running"
    node.start_ts = node.start_ts or ts
    node.end_ts = None
    node.duration_s = None
    node.reason = ""


def _mark_ac_figure_block_finished(
    parent: ExecutionGraphNode,
    child: ExecutionGraphNode,
    ts: str | None,
) -> None:
    parent.end_ts = ts
    parent.duration_s = _duration(parent.start_ts, parent.end_ts)
    if child.status == "error":
        parent.status = "error"
        parent.reason = child.reason
        return
    if any(item.status == "running" for item in parent.children):
        parent.status = "running"
        return
    if not any(item.status == "error" for item in parent.children):
        parent.status = "ok"


def _finish_current_ac_round(
    current_round: dict[str, ExecutionGraphNode],
    parent_id: str,
    ts: str | None,
    *,
    status: str,
    reason: str = "",
) -> None:
    round_node = current_round.get(parent_id)
    if round_node is None or round_node.status != "running":
        return
    round_node.status = status
    round_node.end_ts = ts
    round_node.duration_s = _duration(round_node.start_ts, round_node.end_ts)
    if status == "error":
        round_node.reason = reason or "Agent failed"


def _nested_agent_raw_id(evt: dict[str, Any]) -> str:
    agent = str(evt.get("agent") or evt.get("agent_path") or "agent")
    return _clean_runtime_node_id(agent.rsplit(".", 1)[-1])


def _nested_agent_node_id(evt: dict[str, Any]) -> str:
    call_id = str(evt.get("call_id") or "")
    return f"{_nested_agent_raw_id(evt)}_{call_id}" if call_id else _nested_agent_raw_id(evt)


def _nested_agent_label(evt: dict[str, Any]) -> str:
    raw = str(evt.get("agent") or evt.get("agent_path") or "Agent").rsplit(".", 1)[-1]
    labels = {
        "ACCritic": "Critic",
        "Author": "Author",
        "Council": "Council",
        "Compute": "Compute Worker",
    }
    return labels.get(raw, _humanize_id(raw))


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(evt, dict):
                events.append(evt)
    return events


def _run_preset_name(events: list[dict[str, Any]]) -> str | None:
    for evt in events:
        if evt.get("kind") != "run.start":
            continue
        payload = evt.get("payload") or {}
        if isinstance(payload, dict) and payload.get("preset"):
            return str(payload["preset"])
    return None


def _execution_nodes_from_preset(
    preset_name: str | None,
    preset_root: Path | None,
) -> tuple[list[ExecutionGraphNode], dict[str, ExecutionGraphNode], dict[str, list[str]]]:
    if not preset_name:
        return [], {}, {}
    preset_path = (preset_root or DEFAULT_PRESET_ROOT) / f"{preset_name}.yaml"
    if not preset_path.exists():
        return [], {}, {}
    try:
        raw = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return [], {}, {}
    if not isinstance(raw, dict):
        return [], {}, {}
    dag = raw.get("dag") if isinstance(raw.get("dag"), dict) else {}
    raw_nodes = dag.get("nodes") if isinstance(dag, dict) else []
    if not isinstance(raw_nodes, list):
        return [], {}, {}

    by_id: dict[str, ExecutionGraphNode] = {}
    aliases: dict[str, list[str]] = {}
    roots = [
        node
        for item in raw_nodes
        if isinstance(item, dict)
        for node in [_execution_node_from_dag(item, by_id, aliases)]
    ]
    return roots, by_id, aliases


def _execution_node_from_dag(
    raw: dict[str, Any],
    by_id: dict[str, ExecutionGraphNode],
    aliases: dict[str, list[str]],
    *,
    parent_visual_id: str = "",
) -> ExecutionGraphNode:
    raw_id = str(raw.get("id") or "node")
    visual_id = f"{parent_visual_id}{REPEAT_BODY_MARKER}{raw_id}" if parent_visual_id else raw_id
    ui = raw.get("ui") if isinstance(raw.get("ui"), dict) else {}
    node = ExecutionGraphNode(
        node_id=visual_id,
        raw_id=raw_id,
        label=str(ui.get("label") or _humanize_id(raw_id)),
        subtitle=str(ui.get("subtitle") or _runtime_node_subtitle(raw)),
        kind=str(raw.get("kind") or "agent"),
        parent_node_id=parent_visual_id,
    )
    by_id[visual_id] = node
    aliases.setdefault(raw_id, []).append(visual_id)
    body = raw.get("body")
    body_nodes = body.get("nodes") if isinstance(body, dict) else None
    if isinstance(body_nodes, list):
        node.children = [
            child
            for item in body_nodes
            if isinstance(item, dict)
            for child in [
                _execution_node_from_dag(
                    item,
                    by_id,
                    aliases,
                    parent_visual_id=visual_id,
                )
            ]
        ]
    return node


def _execution_node_for_payload(
    payload: dict[str, Any],
    roots: list[ExecutionGraphNode],
    by_id: dict[str, ExecutionGraphNode],
    aliases: dict[str, list[str]],
    *,
    parent_node: ExecutionGraphNode | None = None,
    for_start: bool = False,
    execution_counts: dict[tuple[str, str], int] | None = None,
) -> ExecutionGraphNode | None:
    node_path = str(payload.get("node_path") or payload.get("editor_node") or "")
    if node_path and node_path in by_id:
        node = by_id[node_path]
        return _runtime_execution_node(
            node,
            roots,
            by_id,
            parent_node,
            for_start=for_start,
            execution_counts=execution_counts,
        )
    raw_id = str(payload.get("node") or "")
    if not raw_id:
        return None
    node = _child_node_by_raw_id(parent_node, raw_id) if parent_node else None
    if parent_node is None:
        if node is None and raw_id in by_id:
            node = by_id[raw_id]
        if node is None:
            alias_ids = aliases.get(raw_id) or []
            if alias_ids:
                node = by_id.get(alias_ids[0])
    if node is not None:
        return _runtime_execution_node(
            node,
            roots,
            by_id,
            parent_node,
            for_start=for_start,
            execution_counts=execution_counts,
        )
    parent_id = parent_node.node_id if parent_node else ""
    node = ExecutionGraphNode(
        node_id=_unique_runtime_node_id(raw_id, by_id, parent_id=parent_id),
        raw_id=raw_id,
        label=str(payload.get("label") or _humanize_id(raw_id)),
        subtitle=str(payload.get("subtitle") or ""),
        kind=str(payload.get("kind") or "agent"),
        parent_node_id=parent_id,
    )
    _attach_execution_graph_node(node, roots, by_id, parent_node)
    aliases.setdefault(raw_id, []).append(node.node_id)
    return _runtime_execution_node(
        node,
        roots,
        by_id,
        parent_node,
        for_start=for_start,
        execution_counts=execution_counts,
    )


def _runtime_execution_node(
    node: ExecutionGraphNode,
    roots: list[ExecutionGraphNode],
    by_id: dict[str, ExecutionGraphNode],
    parent_node: ExecutionGraphNode | None,
    *,
    for_start: bool,
    execution_counts: dict[tuple[str, str], int] | None,
) -> ExecutionGraphNode:
    if not for_start:
        return node
    key = ((parent_node.node_id if parent_node else node.parent_node_id), node.node_id)
    if execution_counts is None:
        return node
    execution_counts[key] = int(execution_counts.get(key, 0) or 0) + 1
    index = execution_counts[key]
    if index == 1 and node.status == "pending" and not node.call_id:
        node.execution_index = 1
        return node
    instance = ExecutionGraphNode(
        node_id=_unique_runtime_node_id(node.node_id, by_id, suffix=f"run{index}", parent_id=parent_node.node_id if parent_node else node.parent_node_id),
        raw_id=node.raw_id,
        label=node.label,
        subtitle=node.subtitle,
        kind=node.kind,
        execution_index=index,
        parent_node_id=parent_node.node_id if parent_node else node.parent_node_id,
    )
    _attach_execution_graph_node(instance, roots, by_id, parent_node or by_id.get(node.parent_node_id))
    return instance


def _child_node_by_raw_id(parent: ExecutionGraphNode | None, raw_id: str) -> ExecutionGraphNode | None:
    if parent is None:
        return None
    for child in parent.children:
        if child.raw_id == raw_id:
            return child
    return None


def _repeat_parent_for_body_event(
    active_by_parent: dict[str, list[ExecutionGraphNode]],
    parent_id: str,
    payload: dict[str, Any],
    repeat_child_offsets: dict[tuple[str, str], int],
) -> ExecutionGraphNode | None:
    if payload.get("node_path") or payload.get("editor_node"):
        return None
    raw_id = str(payload.get("node") or "")
    if not raw_id:
        return None
    candidates = [
        node
        for node in active_by_parent.get(parent_id, [])
        if node.kind in REPEAT_KINDS and _child_node_by_raw_id(node, raw_id) is not None
    ]
    if not candidates:
        return None
    key = (parent_id, raw_id)
    index = int(repeat_child_offsets.get(key, 0) or 0) % len(candidates)
    repeat_child_offsets[key] = int(repeat_child_offsets.get(key, 0) or 0) + 1
    return candidates[index]


def _attach_execution_graph_node(
    node: ExecutionGraphNode,
    roots: list[ExecutionGraphNode],
    by_id: dict[str, ExecutionGraphNode],
    parent: ExecutionGraphNode | None,
) -> None:
    by_id[node.node_id] = node
    if parent is not None:
        if node not in parent.children:
            parent.children.append(node)
        return
    if node not in roots:
        roots.append(node)


def _unique_runtime_node_id(
    base: str,
    by_id: dict[str, ExecutionGraphNode],
    *,
    suffix: str = "",
    parent_id: str = "",
) -> str:
    clean = _clean_runtime_node_id(base)
    if parent_id:
        clean = f"{_clean_runtime_node_id(parent_id)}::runtime::{clean}"
    if suffix:
        clean = f"{clean}::{suffix}"
    candidate = clean
    i = 2
    while candidate in by_id:
        candidate = f"{clean}::{i}"
        i += 1
    return candidate


def _clean_runtime_node_id(value: str) -> str:
    return str(value or "node").replace(" ", "_")


def _active_event_key(parent_id: str, payload: dict[str, Any]) -> tuple[str, str]:
    node_path = str(payload.get("node_path") or payload.get("editor_node") or "")
    raw_id = str(payload.get("node") or "")
    return (parent_id, node_path or raw_id)


def _pop_active_event_node(
    active_events: dict[tuple[str, str], list[ExecutionGraphNode]],
    parent_id: str,
    payload: dict[str, Any],
) -> ExecutionGraphNode | None:
    key = _active_event_key(parent_id, payload)
    nodes = active_events.get(key) or []
    if not nodes:
        return None
    node = nodes.pop()
    if not nodes:
        active_events.pop(key, None)
    return node


def _pop_recent_completed_node(
    recent_completed_nodes: dict[tuple[str, str], list[ExecutionGraphNode]],
    active_events: dict[tuple[str, str], list[ExecutionGraphNode]],
    parent_id: str,
    payload: dict[str, Any],
) -> ExecutionGraphNode | None:
    raw_id = str(payload.get("node") or "")
    if not raw_id:
        return None
    key = (parent_id, raw_id)
    nodes = recent_completed_nodes.get(key) or []
    if not nodes:
        return None
    node = nodes.pop(0)
    if not nodes:
        recent_completed_nodes.pop(key, None)
    _drop_active_event_node(active_events, parent_id, payload, node)
    return node


def _drop_active_event_node(
    active_events: dict[tuple[str, str], list[ExecutionGraphNode]],
    parent_id: str,
    payload: dict[str, Any],
    node: ExecutionGraphNode,
) -> None:
    key = _active_event_key(parent_id, payload)
    nodes = active_events.get(key) or []
    try:
        nodes.remove(node)
    except ValueError:
        return
    if not nodes:
        active_events.pop(key, None)


def _apply_runtime_node_metadata(node: ExecutionGraphNode, payload: dict[str, Any]) -> None:
    if payload.get("kind"):
        node.kind = str(payload["kind"])
    if payload.get("label") and node.label == _humanize_id(node.raw_id):
        node.label = str(payload["label"])
    if payload.get("subtitle") and not node.subtitle:
        node.subtitle = str(payload["subtitle"])


def _attach_call_to_active_node(active: list[ExecutionGraphNode], call_id: str) -> ExecutionGraphNode | None:
    if not call_id:
        return None
    for node in active:
        if node.call_id is None and node.kind not in REPEAT_KINDS and node.kind != "if_else":
            node.call_id = call_id
            return node
    return None


def _drop_active_node(active: list[ExecutionGraphNode], node: ExecutionGraphNode) -> None:
    try:
        active.remove(node)
    except ValueError:
        pass


def _agent_terminal_status_and_reason(kind: str, payload: dict[str, Any]) -> tuple[str, str]:
    if kind == "agent.error":
        return "error", str(payload.get("msg") or payload.get("type") or "Agent failed")
    output = payload.get("output") if isinstance(payload, dict) else None
    if _call_status_from_output(output) == "error":
        err = _call_error_from_output(output)
        return "error", str(err.get("msg") or err.get("type") or "Node output reported an error")
    return "ok", ""


def _apply_agent_terminal_event_to_execution_graph_node(
    node: ExecutionGraphNode,
    kind: str,
    payload: dict[str, Any],
    ts: str | None,
) -> None:
    node.end_ts = ts
    node.duration_s = _duration(node.start_ts, node.end_ts)
    status, reason = _agent_terminal_status_and_reason(kind, payload)
    if status == "error":
        node.status = "error"
        node.reason = reason
        return
    if node.status != "error":
        node.status = "ok"


def _call_error_message(call: CallNode) -> str:
    if isinstance(call.error, dict):
        msg = str(call.error.get("msg") or "").strip()
        if msg:
            return msg
        err_type = str(call.error.get("type") or "").strip()
        if err_type:
            return err_type
    return "Node failed"


def _call_status_from_output(output: Any) -> str | None:
    if not isinstance(output, dict):
        return None
    status = str(output.get("status") or "").strip().lower()
    if status in {"error", "blocked", "timeout"}:
        return "error"
    return None


def _call_error_from_output(output: Any) -> dict[str, str]:
    if not isinstance(output, dict):
        return {"type": "OutputStatus", "msg": "Node output reported an error"}
    status = str(output.get("status") or "error").strip() or "error"
    summary = str(output.get("summary") or "").strip()
    msg = summary or f"Node output status is {status!r}"
    return {"type": "OutputStatus", "msg": msg}


def _runtime_node_subtitle(node: dict[str, Any]) -> str:
    kind = str(node.get("kind") or "agent")
    if kind in REPEAT_KINDS:
        return "Repeats a subgraph until its stop condition is reached."
    if kind == "if_else":
        return "Routes execution based on a condition."
    if kind == "workflow_ref":
        return "Runs another workflow."
    if kind == "map_chain":
        return "Runs a chain over a list of items."
    return ""


def _humanize_id(value: str) -> str:
    return str(value or "node").replace("_", " ").strip().title()


def _duration(start_ts: str | None, end_ts: str | None) -> float | None:
    if not start_ts or not end_ts:
        return None
    try:
        from datetime import datetime
        s = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        return (e - s).total_seconds()
    except Exception:
        return None


def _rollup_cost(node: CallNode) -> tuple[float, int, int]:
    """Recursively roll up cost/tokens from descendants into a node.
    Returns the (cost, in_tokens, out_tokens) including this subtree."""
    sub_cost = node.cost_usd
    sub_in = node.in_tokens
    sub_out = node.out_tokens
    for child in node.children:
        c, ci, co = _rollup_cost(child)
        sub_cost += c
        sub_in += ci
        sub_out += co
    # Stash subtree totals as separate attributes so the template can show both
    node.cost_usd_subtree = sub_cost  # type: ignore[attr-defined]
    node.in_tokens_subtree = sub_in   # type: ignore[attr-defined]
    node.out_tokens_subtree = sub_out  # type: ignore[attr-defined]
    return sub_cost, sub_in, sub_out


# ---------------------------------------------------------------------------
# Per-call detail (UI-0)
# ---------------------------------------------------------------------------


@dataclass
class CallDetail:
    node: CallNode
    input_json: Any = None
    output_json: Any = None
    messages_json: list[dict[str, Any]] | None = None


def load_call_detail(run_path: Path, node: CallNode) -> CallDetail:
    workdir = _resolve_workdir(run_path, node)
    input_json = None
    output_json = None
    messages_json = None
    if workdir is not None and workdir.exists():
        if (workdir / "input.json").exists():
            try:
                input_json = json.loads((workdir / "input.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if (workdir / "output.json").exists():
            try:
                output_json = json.loads((workdir / "output.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if (workdir / "messages.json").exists():
            try:
                raw_messages = json.loads((workdir / "messages.json").read_text(encoding="utf-8"))
                if isinstance(raw_messages, list):
                    messages_json = [m for m in raw_messages if isinstance(m, dict)]
            except (OSError, json.JSONDecodeError):
                pass
    return CallDetail(
        node=node,
        input_json=input_json,
        output_json=output_json,
        messages_json=messages_json,
    )


def _resolve_workdir(run_path: Path, node: CallNode) -> Path | None:
    """Find the agent artifact directory written by the current runner."""
    agents_dir = run_path / "agents"
    if agents_dir.exists():
        match = next(
            (p for p in agents_dir.iterdir() if p.is_dir() and p.name.endswith(f"-{node.call_id}")),
            None,
        )
        if match is not None:
            return match
    return None


def safe_blob_path(run_path: Path, ref: str) -> Path:
    """Resolve a ``$ref`` payload to an absolute path under ``run_path``.

    Raises ``ValueError`` if the resolved path escapes the run dir.
    """
    if not ref:
        raise ValueError("empty $ref")
    resolved = (run_path / ref).resolve()
    run_root = run_path.resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as e:
        raise ValueError(f"$ref escapes run dir: {ref!r}") from e
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"$ref does not point at a file: {ref!r}")
    return resolved


# ---------------------------------------------------------------------------
# Workflow presets (UI-2)
# ---------------------------------------------------------------------------


CONFIGURABLE_PROMPT_AGENT = "proofstack.agents.configurable_prompt.ConfigurablePromptAgent"
CONFIGURABLE_CLI_AGENT = "proofstack.agents.configurable_cli.ConfigurableCLIAgent"
DEFAULT_MODEL_ROOT = CONFIGS_ROOT / "models"


@dataclass
class PresetInfo:
    name: str
    label: str
    workflow_qualname: str
    description: str
    inputs: dict[str, Any]
    model_overrides: dict[str, Any]
    component_configs: dict[str, Any]
    budget: dict[str, Any] | None
    raw_yaml: str
    file_version: str
    error: str | None = None


def discover_presets(root: Path | None = None) -> list[PresetInfo]:
    paths = list_presets(root or DEFAULT_PRESET_ROOT)
    out: list[PresetInfo] = []
    for path in paths:
        out.append(_preset_to_info(path))
    return out


def presets_registry_version(root: Path | None = None) -> str:
    root = root or DEFAULT_PRESET_ROOT
    if not root.exists():
        return "empty"
    parts: list[str] = []
    for path in sorted(root.glob("*.yaml")):
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def discover_model_options(root: Path | None = None) -> list[str]:
    """Return model config refs that can be used in prompt components."""
    root = (root or DEFAULT_MODEL_ROOT).resolve()
    if not root.exists():
        return []

    refs: list[str] = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict) or data.get("type") == "agent":
            continue
        rel = path.relative_to(root).with_suffix("")
        refs.append(f"models/{rel.as_posix()}")
    return refs


def discover_exported_presets(root: Path | None = None) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for preset in discover_presets(root):
        if preset.error:
            continue
        try:
            raw = yaml.safe_load(preset.raw_yaml) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict):
            continue
        export = raw.get("export")
        if not isinstance(export, dict) or not export.get("visible_as_node"):
            continue
        exported.append(
            {
                "name": preset.name,
                "label": _preset_display_label(raw, preset.name),
                "description": str(export.get("description") or preset.description or ""),
                "params": export.get("params") if isinstance(export.get("params"), dict) else {},
                "inputs": _workflow_input_fields(raw),
                "outputs": _workflow_output_fields(raw),
            }
        )
    return sorted(exported, key=lambda item: str(item["label"]).lower())


def find_preset(name: str, root: Path | None = None) -> PresetInfo | None:
    for p in discover_presets(root):
        if p.name == name:
            return p
    return None


def preset_dag_report(name: str, root: Path | None = None) -> DAGReport:
    preset = find_preset(name, root)
    if preset is None:
        raise PresetError(f"preset not found: {name}")
    try:
        raw = yaml.safe_load(preset.raw_yaml) or {}
    except yaml.YAMLError as e:
        return DAGReport(ok=False, errors=[f"YAML parse error: {e}"], nodes=[], edges=[])
    return _dag_report_from_raw(raw, workflow_qualname=preset.workflow_qualname)


def validate_preset_yaml(raw_yaml: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as e:
        return {
            "ok": False,
            "errors": [f"YAML parse error: {e}"],
            "nodes": [],
            "edges": [],
        }
    if not isinstance(raw, dict):
        return {"ok": False, "errors": ["top-level YAML must be a mapping"], "nodes": [], "edges": []}
    preset_errors = _preset_validation_errors(raw)
    report = _dag_report_from_raw(raw, workflow_qualname=_workflow_qualname_from_raw(raw))
    out = report.to_dict()
    if preset_errors:
        out["ok"] = False
        out["errors"] = [*preset_errors, *out.get("errors", [])]
    return out


def _preset_validation_errors(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    workflow_path = raw.get("workflow")
    workflow_cls = _workflow_cls_from_raw(raw)
    if not isinstance(workflow_path, str) or not workflow_path:
        errors.append("missing or non-string 'workflow' key")
    elif "." not in workflow_path:
        errors.append(f"workflow path must be dotted: {workflow_path!r}")
    elif workflow_cls is None:
        try:
            module_name, _, class_name = workflow_path.rpartition(".")
            cls = getattr(importlib.import_module(module_name), class_name)
            from proofstack.agent import Agent

            if not issubclass(cls, Agent):
                errors.append(f"'workflow' {workflow_path} is not an Agent subclass")
        except Exception as e:
            errors.append(f"cannot import workflow {workflow_path!r}: {e}")

    for key in ("inputs", "model_overrides", "budget"):
        if key in raw and raw[key] is not None and not isinstance(raw[key], dict):
            errors.append(f"'{key}' must be a mapping")
    if isinstance(raw.get("budget"), dict):
        try:
            from proofstack.budget import BudgetSpec

            BudgetSpec(**raw["budget"])
        except Exception as e:
            errors.append(f"invalid budget: {e}")

    components = raw.get("components") or {}
    if not isinstance(components, dict):
        errors.append("'components' must be a mapping")
    else:
        for name, cfg in components.items():
            if not isinstance(name, str) or not isinstance(cfg, dict):
                errors.append("each component config must map a string name to a mapping")
                break

    if workflow_cls is not None and not hasattr(workflow_cls, "Inputs"):
        errors.append(f"workflow {workflow_cls.__name__} does not declare Inputs")
    return errors


def _workflow_cls_from_raw(raw: dict[str, Any]) -> type | None:
    workflow_path = raw.get("workflow")
    if not isinstance(workflow_path, str) or "." not in workflow_path:
        return None
    try:
        module_name, _, class_name = workflow_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        from proofstack.agent import Agent

        if not issubclass(cls, Agent):
            return None
        return cls
    except Exception:
        return None


def _is_dag_workflow_cls(cls: type | None) -> bool:
    if cls is None:
        return False
    try:
        from proofstack.agents.dag_workflow import DAGWorkflow

        return issubclass(cls, DAGWorkflow)
    except Exception:
        return False


def _workflow_qualname_from_raw(raw: dict[str, Any]) -> str | None:
    cls = _workflow_cls_from_raw(raw)
    if cls is not None:
        return str(getattr(cls, "__qualname__", "") or cls.__name__)
    return None


def _dag_report_from_raw(raw: dict[str, Any], workflow_qualname: str | None = None) -> DAGReport:
    components = raw.get("components") or {}
    if not isinstance(components, dict):
        components = {}
    dag = raw.get("dag")
    if dag is None:
        workflow_cfg = _workflow_component_config(raw, components, workflow_qualname)
        dag = workflow_cfg.get("dag") if isinstance(workflow_cfg, dict) else None
    workflow_cls = _workflow_cls_from_raw(raw)
    workflow_inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
    workflow_budget = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    workflow_inputs = {
        key: value
        for key, value in workflow_inputs.items()
        if key not in workflow_budget
    }
    if dag is None and workflow_cls is not None and not _is_dag_workflow_cls(workflow_cls):
        return DAGReport(
            ok=True,
            errors=[],
            nodes=[],
            edges=[],
            warnings=[
                f"{workflow_cls.__qualname__} is a custom workflow, so the visual DAG editor is unavailable. Edit the YAML directly."
            ],
            workflow_inputs=dict(workflow_inputs),
            workflow_input_schema=_workflow_input_schema(raw),
            workflow_budget=dict(workflow_budget),
            workflow_outputs={},
            workflow_output_ui={},
        )
    workflow_outputs = dag.get("outputs") if isinstance(dag, dict) and isinstance(dag.get("outputs"), dict) else {}
    dag_ui = dag.get("ui") if isinstance(dag, dict) and isinstance(dag.get("ui"), dict) else {}
    workflow_output_ui = (
        dag_ui.get("workflow_output")
        if isinstance(dag_ui.get("workflow_output"), dict)
        else {}
    )
    return build_dag_report(
        dag,
        component_configs=components,
        workflow_inputs=dict(workflow_inputs),
        workflow_input_schema=_workflow_input_schema(raw),
        workflow_budget=dict(workflow_budget),
        workflow_outputs=dict(workflow_outputs),
        workflow_output_ui=dict(workflow_output_ui),
    )


def _workflow_component_config(
    raw: dict[str, Any],
    components: dict[str, Any],
    workflow_qualname: str | None = None,
) -> dict[str, Any]:
    workflow_path = str(raw.get("workflow") or "")
    keys = [
        key
        for key in (
            workflow_qualname,
            workflow_path,
            workflow_path.rpartition(".")[2],
        )
        if key
    ]
    for key in keys:
        cfg = components.get(key)
        if isinstance(cfg, dict):
            return cfg
    return {}


def _workflow_input_schema(raw: dict[str, Any]) -> dict[str, Any]:
    declared = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
    budget = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    referenced = _input_ref_ids(raw)
    names = {"problem", *declared, *referenced} - set(budget)
    workflow_path = raw.get("workflow")
    props: dict[str, Any] = {}
    if not isinstance(workflow_path, str) or "." not in workflow_path:
        return {name: _schema_for_default(declared.get(name)) for name in sorted(names)}
    try:
        module_name, _, class_name = workflow_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        schema = cls.Inputs.model_json_schema()  # type: ignore[attr-defined]
        raw_props = schema.get("properties", {})
        if isinstance(raw_props, dict):
            props = raw_props
    except Exception:
        props = {}
    return {
        name: props.get(name, _schema_for_default(declared.get(name)))
        for name in sorted(names)
    }


def _input_ref_ids(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(re.findall(r"\$input\.([A-Za-z_][A-Za-z0-9_-]*)", value))
    elif isinstance(value, list):
        for item in value:
            refs.update(_input_ref_ids(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(_input_ref_ids(item))
    return refs


def _schema_for_default(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        return {"type": "array"}
    if isinstance(value, dict):
        return {"type": "object"}
    return {"type": "string"}


def save_preset_yaml(name: str, raw_yaml: str, root: Path | None = None) -> Path:
    root = root or DEFAULT_PRESET_ROOT
    path = _preset_write_path(name, root)
    validation = validate_preset_yaml(raw_yaml)
    if not validation.get("ok"):
        raise PresetError("; ".join(validation.get("errors") or ["invalid DAG YAML"]))
    path.write_text(raw_yaml, encoding="utf-8")
    return path


def preset_file_version(name: str, root: Path | None = None) -> str:
    root = root or DEFAULT_PRESET_ROOT
    return _path_file_version(_preset_write_path(name, root))


def delete_preset(name: str, root: Path | None = None) -> None:
    root = root or DEFAULT_PRESET_ROOT
    path = _preset_write_path(name, root)
    if not path.exists():
        raise PresetError(f"agent not found: {name}")
    path.unlink()


def normalize_preset_name(value: str) -> str:
    name = str(value or "").strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = name.strip("_-")
    if not name:
        raise PresetError("agent file name cannot be empty")
    return name


def _preset_write_path(name: str, root: Path) -> Path:
    normalized = normalize_preset_name(name)
    if normalized != name:
        raise PresetError(f"invalid agent file name: use {normalized!r}")
    path = (root / f"{normalized}.yaml").resolve()
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as e:
        raise PresetError(f"refusing to write outside preset root: {path}") from e
    return path


def mutate_preset_yaml(raw_yaml: str, operation: dict[str, Any]) -> dict[str, Any]:
    """Apply one visual-editor operation to a preset YAML document.

    The editor intentionally has no separate graph JSON persistence. It sends
    small operations here; this function mutates the YAML mapping directly and
    returns the new YAML plus a fresh typed DAG report.
    """
    try:
        raw = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as e:
        return _mutation_error(f"YAML parse error: {e}", raw_yaml=raw_yaml)
    if not isinstance(raw, dict):
        return _mutation_error("top-level YAML must be a mapping", raw_yaml=raw_yaml)
    if not isinstance(operation, dict):
        return _mutation_error("operation must be a mapping", raw_yaml=raw_yaml)

    try:
        op = str(operation.get("op") or "")
        if op == "update_agent_label":
            _op_update_agent_label(raw, operation)
        elif op == "move_node":
            _op_move_node(raw, operation)
        elif op == "move_nodes":
            _op_move_nodes(raw, operation)
        elif op == "move_repeat_body_node":
            _op_move_repeat_body_node(raw, operation)
        elif op == "move_repeat_virtual_node":
            _op_move_repeat_virtual_node(raw, operation)
        elif op == "move_repeat_zone_visuals":
            _op_move_repeat_zone_visuals(raw, operation)
        elif op == "move_node_into_repeat_zone":
            _op_move_node_into_repeat_zone(raw, operation)
        elif op == "move_node_path_into_repeat_zone":
            _op_move_node_path_into_repeat_zone(raw, operation)
        elif op == "move_node_range_into_repeat_zone":
            _op_move_node_path_into_repeat_zone(raw, operation)
        elif op == "create_repeat_zone_from_path":
            _op_create_repeat_zone_from_path(raw, operation)
        elif op == "add_node":
            _op_add_node(raw, operation)
        elif op == "copy_node":
            _op_copy_node(raw, operation)
        elif op == "copy_nodes":
            _op_copy_nodes(raw, operation)
        elif op == "update_node":
            _op_update_node(raw, operation)
        elif op == "update_component":
            _op_update_component(raw, operation)
        elif op == "update_node_inputs":
            _op_update_node_inputs(raw, operation)
        elif op == "update_node_outputs":
            _op_update_node_outputs(raw, operation)
        elif op == "update_node_condition":
            _op_update_node_condition(raw, operation)
        elif op == "update_node_when":
            _op_update_node_when(raw, operation)
        elif op == "update_node_default_outputs":
            _op_update_node_default_outputs(raw, operation)
        elif op == "update_loop_node":
            _op_update_loop_node(raw, operation)
        elif op == "update_workflow_ref":
            _op_update_workflow_ref(raw, operation)
        elif op == "tie_component":
            _op_tie_component(raw, operation)
        elif op == "untie_component":
            _op_untie_component(raw, operation)
        elif op == "update_workflow_inputs":
            _op_update_workflow_inputs(raw, operation)
        elif op == "update_workflow_budget":
            _op_update_workflow_budget(raw, operation)
        elif op == "add_workflow_output":
            _op_add_workflow_output(raw, operation)
        elif op == "update_workflow_output":
            _op_update_workflow_output(raw, operation)
        elif op == "delete_workflow_output":
            _op_delete_workflow_output(raw, operation)
        elif op == "update_workflow_output_node":
            _op_update_workflow_output_node(raw, operation)
        elif op == "move_workflow_output_node":
            _op_move_workflow_output_node(raw, operation)
        elif op == "connect_edge":
            _op_connect_edge(raw, operation)
        elif op == "reconnect_edge":
            _op_reconnect_edge(raw, operation)
        elif op == "disconnect_edge":
            _op_disconnect_edge(raw, operation)
        elif op == "delete_node":
            _op_delete_node(raw, operation)
        else:
            raise PresetError(f"unknown editor operation: {op!r}")
    except Exception as e:
        return _mutation_error(str(e), raw_yaml=raw_yaml)

    new_yaml = _dump_preset_yaml(raw)
    report = validate_preset_yaml(new_yaml)
    return {
        "ok": bool(report.get("ok")),
        "raw_yaml": new_yaml,
        "report": report,
        "errors": report.get("errors") or [],
    }


def _mutation_error(message: str, *, raw_yaml: str) -> dict[str, Any]:
    return {
        "ok": False,
        "raw_yaml": raw_yaml,
        "report": validate_preset_yaml(raw_yaml),
        "errors": [message],
    }


def _editable_dag(raw: dict[str, Any]) -> dict[str, Any]:
    dag = raw.setdefault("dag", {})
    if not isinstance(dag, dict):
        raise PresetError("top-level dag must be a mapping")
    nodes = dag.setdefault("nodes", [])
    if not isinstance(nodes, list):
        raise PresetError("dag.nodes must be a list")
    dag.setdefault("outputs", {})
    if not isinstance(dag["outputs"], dict):
        raise PresetError("dag.outputs must be a mapping")
    return dag


def _components(raw: dict[str, Any]) -> dict[str, Any]:
    components = raw.setdefault("components", {})
    if not isinstance(components, dict):
        raise PresetError("components must be a mapping")
    return components


def _node_by_id(raw: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in _editable_dag(raw)["nodes"]:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    raise PresetError(f"node not found: {node_id}")


def _node_by_editor_id(raw: dict[str, Any], node_id: str) -> dict[str, Any]:
    if REPEAT_BODY_MARKER in node_id:
        loop_id, body_node_id = node_id.split(REPEAT_BODY_MARKER, 1)
        loop = _node_by_id(raw, loop_id)
        return _repeat_body_node_by_id(loop, body_node_id)
    return _node_by_id(raw, node_id)


def _editor_node_ref(raw: dict[str, Any], node_id: str) -> EditorNodeRef:
    text = str(node_id or "").strip()
    if _is_workflow_output_target(text):
        return EditorNodeRef("workflow_output", WORKFLOW_OUTPUT_NODE_ID)
    if text.endswith(REPEAT_INPUT_SUFFIX):
        loop_id = _clean_id(text[: -len(REPEAT_INPUT_SUFFIX)])
        loop = _node_by_id(raw, loop_id)
        if not _is_repeat_node(loop):
            raise PresetError(f"{loop_id}: target is not a repeat zone")
        return EditorNodeRef("repeat_input", loop_id, loop, loop_id=loop_id, loop=loop)
    if text.endswith(REPEAT_OUTPUT_SUFFIX):
        loop_id = _clean_id(text[: -len(REPEAT_OUTPUT_SUFFIX)])
        loop = _node_by_id(raw, loop_id)
        if not _is_repeat_node(loop):
            raise PresetError(f"{loop_id}: target is not a repeat zone")
        return EditorNodeRef("repeat_output", loop_id, loop, loop_id=loop_id, loop=loop)
    if REPEAT_BODY_MARKER in text:
        loop_id_raw, body_node_id_raw = text.split(REPEAT_BODY_MARKER, 1)
        loop_id = _clean_id(loop_id_raw)
        body_node_id = _clean_id(body_node_id_raw)
        loop = _node_by_id(raw, loop_id)
        if not _is_repeat_node(loop):
            raise PresetError(f"{loop_id}: target is not a repeat zone")
        body_node = _repeat_body_node_by_id(loop, body_node_id)
        return EditorNodeRef(
            "repeat_body",
            body_node_id,
            body_node,
            loop_id=loop_id,
            loop=loop,
            body_node_id=body_node_id,
        )
    clean_id = _clean_id(text)
    return EditorNodeRef("top", clean_id, _node_by_id(raw, clean_id))


def _op_update_agent_label(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    label = str(operation.get("label") or "").strip()
    if not label:
        raise PresetError("agent name cannot be empty")
    export = raw.setdefault("export", {})
    if not isinstance(export, dict):
        export = {}
        raw["export"] = export
    export["label"] = label


def _op_move_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_id(raw, str(operation.get("node_id") or ""))
    ui = _node_ui(node)
    ui["x"] = int(float(operation.get("x", ui.get("x", 0))))
    ui["y"] = int(float(operation.get("y", ui.get("y", 0))))


def _op_move_nodes(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    moves = operation.get("nodes")
    if not isinstance(moves, list) or not moves:
        raise PresetError("nodes must be a non-empty list")
    for move in moves:
        if not isinstance(move, dict):
            raise PresetError("each node move must be a mapping")
        _op_move_node(raw, move)


def _op_move_repeat_body_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    loop = _node_by_id(raw, _clean_id(str(operation.get("loop_id") or "")))
    body_node = _repeat_body_node_by_id(loop, _clean_id(str(operation.get("body_node_id") or "")))
    ui = _node_ui(body_node)
    ui["x"] = int(float(operation.get("x", ui.get("x", 0))))
    ui["y"] = int(float(operation.get("y", ui.get("y", 0))))


def _op_move_repeat_virtual_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    loop = _node_by_id(raw, _clean_id(str(operation.get("loop_id") or "")))
    if not _is_repeat_node(loop):
        raise PresetError(f"{loop.get('id')}: target is not a repeat zone")
    visual_type = str(operation.get("visual_type") or "")
    if visual_type not in {"repeat_input", "repeat_output"}:
        raise PresetError("visual_type must be repeat_input or repeat_output")
    _set_repeat_virtual_position(loop, visual_type, operation)


def _op_move_repeat_zone_visuals(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    loop = _node_by_id(raw, _clean_id(str(operation.get("loop_id") or "")))
    if not _is_repeat_node(loop):
        raise PresetError(f"{loop.get('id')}: target is not a repeat zone")
    for visual_type in ("repeat_input", "repeat_output"):
        move = operation.get(visual_type)
        if isinstance(move, dict):
            _set_repeat_virtual_position(loop, visual_type, move)
    body_moves = operation.get("body_nodes")
    if body_moves is not None and not isinstance(body_moves, list):
        raise PresetError("body_nodes must be a list")
    for move in body_moves or []:
        if not isinstance(move, dict):
            raise PresetError("each body node move must be a mapping")
        body_node = _repeat_body_node_by_id(loop, _clean_id(str(move.get("body_node_id") or "")))
        ui = _node_ui(body_node)
        ui["x"] = int(float(move.get("x", ui.get("x", 0))))
        ui["y"] = int(float(move.get("y", ui.get("y", 0))))


def _set_repeat_virtual_position(loop: dict[str, Any], visual_type: str, move: dict[str, Any]) -> None:
    ui = _node_ui(loop)
    visual_ui = ui.setdefault(visual_type, {})
    if not isinstance(visual_ui, dict):
        visual_ui = {}
        ui[visual_type] = visual_ui
    visual_ui["x"] = int(float(move.get("x", visual_ui.get("x", 0))))
    visual_ui["y"] = int(float(move.get("y", visual_ui.get("y", 0))))


def _op_move_node_into_repeat_zone(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node_id = _clean_id(str(operation.get("node_id") or ""))
    loop_id = _clean_id(str(operation.get("loop_id") or ""))
    if not node_id or not loop_id:
        raise PresetError("node_id and loop_id are required")
    if node_id == loop_id:
        raise PresetError("cannot move a repeat zone into itself")
    dag = _editable_dag(raw)
    loop = _node_by_id(raw, loop_id)
    if not _is_repeat_node(loop):
        raise PresetError(f"{loop_id}: target is not a repeat zone")
    source_index = next(
        (
            index
            for index, node in enumerate(dag["nodes"])
            if isinstance(node, dict) and node.get("id") == node_id
        ),
        -1,
    )
    if source_index < 0:
        raise PresetError(f"node not found: {node_id}")
    source = dag["nodes"][source_index]
    if not isinstance(source, dict):
        raise PresetError(f"node not found: {node_id}")
    if _is_repeat_node(source):
        raise PresetError("moving nested repeat zones into a repeat zone is not supported yet")
    body = loop.setdefault("body", {})
    if not isinstance(body, dict):
        body = {}
        loop["body"] = body
    body_nodes = body.setdefault("nodes", [])
    if not isinstance(body_nodes, list):
        raise PresetError(f"{loop_id}: repeat zone body.nodes must be a list")
    moved = _prepare_node_for_repeat_body(raw, source, body_nodes)
    del dag["nodes"][source_index]
    body_nodes.append(moved)
    _preserve_repeat_state_refs(loop, [moved])
    _replace_top_level_needs(dag, {node_id}, loop_id)
    moved_pairs = {node_id: str(moved["id"])}
    _sync_repeat_state_updates_for_body_node(raw, loop, moved)
    _set_repeat_outputs_from_end_node(raw, loop, node_id, moved_pairs)
    _rewrite_top_level_refs_to_repeat_outputs(
        dag,
        loop_id,
        moved_pairs,
        public_old_ids={node_id},
    )
    _sync_loop_body_needs(loop)
    _sync_needs_from_refs(raw, loop)


def _op_move_node_range_into_repeat_zone(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    _op_move_node_path_into_repeat_zone(raw, operation)


def _op_move_node_path_into_repeat_zone(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    loop_id = _clean_id(str(operation.get("loop_id") or ""))
    start_id = _clean_id(str(operation.get("start_node_id") or ""))
    end_id = _clean_id(str(operation.get("end_node_id") or ""))
    if not loop_id or not start_id or not end_id:
        raise PresetError("loop_id, start_node_id, and end_node_id are required")
    dag = _editable_dag(raw)
    loop = _node_by_id(raw, loop_id)
    if not _is_repeat_node(loop):
        raise PresetError(f"{loop_id}: target is not a repeat zone")
    selected_ids = _node_ids_on_paths(dag, start_id, end_id)
    selected = _selected_top_level_nodes_for_repeat(dag, selected_ids, exclude={loop_id})
    if not selected:
        raise PresetError("no movable top-level nodes found on graph paths between the selected start and end nodes")
    body = loop.setdefault("body", {})
    if not isinstance(body, dict):
        body = {}
        loop["body"] = body
    body_nodes = body.setdefault("nodes", [])
    if not isinstance(body_nodes, list):
        raise PresetError(f"{loop_id}: repeat zone body.nodes must be a list")
    moved_pairs = _move_selected_nodes_into_repeat_body(raw, dag, loop, selected, body_nodes)
    _set_repeat_outputs_from_end_node(raw, loop, end_id, moved_pairs)
    _rewrite_top_level_refs_to_repeat_outputs(
        dag,
        loop_id,
        moved_pairs,
        public_old_ids={end_id},
    )
    _sync_loop_body_needs(loop)
    _sync_needs_from_refs(raw, loop)


def _op_create_repeat_zone_from_path(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    dag = _editable_dag(raw)
    start_id = _clean_id(str(operation.get("start_node_id") or ""))
    end_id = _clean_id(str(operation.get("end_node_id") or ""))
    if not start_id or not end_id:
        raise PresetError("start_node_id and end_node_id are required")
    selected_ids = _node_ids_on_paths(dag, start_id, end_id)
    selected = _selected_top_level_nodes_for_repeat(dag, selected_ids)
    if not selected:
        raise PresetError("no movable top-level nodes found on graph paths between the selected start and end nodes")
    loop_id = _unique_id(
        _clean_id(str(operation.get("node_id") or f"repeat_{start_id}_to_{end_id}")),
        dag["nodes"],
    )
    min_x = min(_node_x(node) for node in selected)
    min_y = min(_node_y(node) for node in selected)
    loop = {
        "id": loop_id,
        "kind": "repeat",
        "max_iterations": 3,
        "condition": {"python": "iteration < max_iterations"},
        "initial_state": {},
        "body": {"nodes": [], "state_updates": {}},
        "outputs": {},
        "ui": {
            "x": int(min_x) - 120,
            "y": int(min_y) - 80,
            "label": str(operation.get("label") or "Repeat"),
        },
    }
    selected_index = min(
        index
        for index, node in enumerate(dag["nodes"])
        if isinstance(node, dict) and str(node.get("id")) in {str(item.get("id")) for item in selected}
    )
    body_nodes = loop["body"]["nodes"]
    selected_set = {str(node.get("id")) for node in selected}
    loop["needs"] = sorted(_external_dependencies_for_nodes(dag, selected_set))
    moved_pairs = _move_selected_nodes_into_repeat_body(raw, dag, loop, selected, body_nodes)
    dag["nodes"].insert(selected_index, loop)
    _set_repeat_outputs_from_end_node(raw, loop, end_id, moved_pairs)
    _rewrite_top_level_refs_to_repeat_outputs(
        dag,
        loop_id,
        moved_pairs,
        public_old_ids={end_id},
    )
    _sync_loop_body_needs(loop)
    _sync_needs_from_refs(raw, loop)


def _op_add_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    dag = _editable_dag(raw)
    nodes = dag["nodes"]
    template = str(operation.get("template") or "prompt_agent")
    if template in REMOVED_NODE_TEMPLATES:
        raise PresetError(
            f"The {template!r} add-node template has been removed. "
            "Use Parallel Solve / Verify / Improve or a Basic I/O Agent instead."
        )
    default_base_id = operation.get("preset") if template == "workflow_ref" else ""
    if template == "python_agent":
        default_base_id = str(operation.get("node_id") or _agent_base_id(str(operation.get("agent") or "")))
    base_id = _clean_id(str(operation.get("node_id") or default_base_id or _template_base_id(template)))
    node_id = _unique_id(base_id, nodes)
    ui = {
        "x": int(float(operation.get("x", 80))),
        "y": int(float(operation.get("y", 80))),
        "label": str(operation.get("label") or _template_label(template, node_id)),
    }

    if template == "latex":
        component = _ensure_latex_cli_component(raw, node_id)
        node = {
            "id": node_id,
            "kind": "agent",
            "agent": CONFIGURABLE_CLI_AGENT,
            "name": component,
            "inputs": {"tex_body": ""},
            "ui": ui,
        }
    elif template == "if_else":
        node = {
            "id": node_id,
            "kind": "if_else",
            "condition": {"python": "False"},
            "inputs": {},
            "then": {"True": True},
            "else": {"False": True},
            "ui": ui,
        }
    elif template == "budget_fallback":
        component = _ensure_prompt_component(raw, node_id, "budget_fallback")
        node = {
            "id": node_id,
            "kind": "agent",
            "run_on": "budget_exhausted",
            "agent": CONFIGURABLE_PROMPT_AGENT,
            "name": component,
            "inputs": {
                "problem": "$input.problem",
                "best_tex": "$best_tex",
                "available_outputs": "$node",
                "budget_error": "$run.budget_error.message",
            },
            "best_tex": "$output.solution",
            "ui": ui,
        }
    elif template == "repeat":
        component = _ensure_prompt_component(raw, f"{node_id}_step", "loop_step")
        node = {
            "id": node_id,
            "kind": "repeat",
            "max_iterations": 3,
            "condition": {"python": "iteration < max_iterations"},
            "initial_state": {"solution": ""},
            "body": {
                "nodes": [
                    {
                        "id": "loop_step",
                        "kind": "agent",
                        "agent": CONFIGURABLE_PROMPT_AGENT,
                        "name": component,
                        "inputs": {
                            "problem": "$input.problem",
                            "current_solution": "$state.solution",
                        },
                    }
                ],
                "state_updates": {
                    "solution": {"coalesce": ["$node.loop_step.solution", "$node.loop_step.output", "$state.solution"]}
                },
            },
            "outputs": {
                "solution": "$state.solution",
                "iterations": "$loop.iterations",
                "history": "$history",
            },
            "ui": ui,
        }
    elif template == "workflow_ref":
        node = {
            "id": node_id,
            "kind": "workflow_ref",
            "preset": str(operation.get("preset") or "verify_improve"),
            "params": {},
            "inputs": {},
            "ui": ui,
        }
    elif template == "cli_agent":
        component = _ensure_cli_component(raw, node_id)
        node = {
            "id": node_id,
            "kind": "agent",
            "agent": CONFIGURABLE_CLI_AGENT,
            "name": component,
            "inputs": {},
            "ui": ui,
        }
    elif template == "python_agent":
        agent_path = str(operation.get("agent") or "")
        if not agent_path:
            raise PresetError("python_agent nodes require an agent path")
        component = _ensure_agent_component(raw, node_id, agent_path)
        node = {
            "id": node_id,
            "kind": "agent",
            "agent": agent_path,
            "name": component,
            "inputs": _default_python_agent_inputs(agent_path),
            "ui": ui,
        }
        if _python_agent_has_output(agent_path, "solution"):
            node["best_tex"] = "$output.solution"
    else:
        component = _ensure_prompt_component(raw, node_id, template)
        node = {
            "id": node_id,
            "kind": "agent",
            "agent": CONFIGURABLE_PROMPT_AGENT,
            "name": component,
            "inputs": _default_prompt_inputs(template),
            "ui": ui,
        }
    nodes.append(node)


def _op_copy_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    dag = _editable_dag(raw)
    source = _node_by_id(raw, str(operation.get("node_id") or ""))
    node = copy.deepcopy(source)
    old_id = str(node.get("id") or "node")
    node["id"] = _unique_id(_clean_id(str(operation.get("new_node_id") or f"{old_id}_copy")), dag["nodes"])
    node.pop("needs", None)
    ui = _node_ui(node)
    ui["label"] = str(ui.get("label") or old_id)
    ui["x"] = int(float(operation.get("x", int(float(ui.get("x", 80))) + 48)))
    ui["y"] = int(float(operation.get("y", int(float(ui.get("y", 80))) + 48)))
    ui.pop("managed_needs", None)
    dag["nodes"].append(node)
    _sync_needs_from_refs(raw, node)


def _op_copy_nodes(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    dag = _editable_dag(raw)
    requested = operation.get("nodes")
    if not isinstance(requested, list) or not requested:
        raise PresetError("nodes must be a non-empty list")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in requested:
        if not isinstance(entry, dict):
            raise PresetError("each copied node must be a mapping")
        node_id = _clean_id(str(entry.get("node_id") or ""))
        if node_id in seen:
            continue
        source = _node_by_id(raw, node_id)
        entries.append({"source": source, "operation": entry})
        seen.add(node_id)
    if not entries:
        raise PresetError("nodes must include at least one existing node")

    occupied = [*dag["nodes"]]
    id_map: dict[str, str] = {}
    for entry in entries:
        source = entry["source"]
        old_id = str(source.get("id") or "node")
        requested_id = str(entry["operation"].get("new_node_id") or f"{old_id}_copy")
        new_id = _unique_id(_clean_id(requested_id), occupied)
        id_map[old_id] = new_id
        occupied.append({"id": new_id})

    copied_nodes: list[dict[str, Any]] = []
    for entry in entries:
        source = entry["source"]
        spec = entry["operation"]
        old_id = str(source.get("id") or "node")
        node = copy.deepcopy(source)
        node["id"] = id_map[old_id]
        _rewrite_copied_node_refs(node, id_map)

        needs = node.get("needs")
        if isinstance(needs, list):
            node["needs"] = [
                id_map.get(str(dep), str(dep))
                for dep in needs
                if isinstance(dep, str)
            ]

        ui = _node_ui(node)
        ui["label"] = str(ui.get("label") or old_id)
        ui["x"] = int(float(spec.get("x", int(float(ui.get("x", 80))) + 48)))
        ui["y"] = int(float(spec.get("y", int(float(ui.get("y", 80))) + 48)))
        ui.pop("managed_needs", None)
        copied_nodes.append(node)

    dag["nodes"].extend(copied_nodes)
    for node in copied_nodes:
        if _is_repeat_node(node):
            _sync_loop_body_needs(node)
        _sync_needs_from_refs(raw, node)


def _rewrite_copied_node_refs(node: dict[str, Any], id_map: dict[str, str]) -> None:
    if _is_repeat_node(node):
        for key in ("inputs", "initial_state", "when", "default", "best_tex", "outputs"):
            if key in node:
                node[key] = _rewrite_copied_refs(node[key], id_map)
        body = node.get("body")
        if isinstance(body, dict):
            for key in ("nodes", "state_updates"):
                if key in body:
                    body[key] = _rewrite_copied_refs(body[key], id_map, rewrite_node_refs=False)
        return
    for key, value in list(node.items()):
        if key in {"id", "ui"}:
            continue
        node[key] = _rewrite_copied_refs(value, id_map)


def _rewrite_copied_refs(value: Any, id_map: dict[str, str], *, rewrite_node_refs: bool = True) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            prefix = match.group(1)
            node_id = match.group(2)
            if node_id not in id_map:
                return match.group(0)
            if prefix == "node" and not rewrite_node_refs:
                return match.group(0)
            return f"${prefix}.{id_map[node_id]}"

        return re.sub(r"\$(node|parent\.node)\.([A-Za-z_][A-Za-z0-9_-]*)", repl, value)
    if isinstance(value, list):
        return [_rewrite_copied_refs(item, id_map, rewrite_node_refs=rewrite_node_refs) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_copied_refs(item, id_map, rewrite_node_refs=rewrite_node_refs)
            for key, item in value.items()
        }
    return value


def _repeat_body_node_by_id(loop: dict[str, Any], body_node_id: str) -> dict[str, Any]:
    body = loop.get("body")
    nodes = body.get("nodes") if isinstance(body, dict) else None
    if not isinstance(nodes, list):
        raise PresetError(f"{loop.get('id')}: repeat zone has no body nodes")
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == body_node_id:
            return node
    raise PresetError(f"repeat body node not found: {body_node_id}")


def _move_selected_nodes_into_repeat_body(
    raw: dict[str, Any],
    dag: dict[str, Any],
    loop: dict[str, Any],
    selected: list[dict[str, Any]],
    body_nodes: list[Any],
) -> dict[str, str]:
    selected_ids = {str(node.get("id")) for node in selected}
    id_map: dict[str, str] = {}
    occupied = [*body_nodes]
    for source in selected:
        old_id = str(source.get("id") or "node")
        new_id = _unique_id(_clean_id(old_id), occupied)
        id_map[old_id] = new_id
        occupied.append({"id": new_id})
    moved_by_old_id: dict[str, dict[str, Any]] = {}
    for source in selected:
        old_id = str(source.get("id"))
        moved = _prepare_node_for_repeat_body(raw, source, body_nodes, id_map=id_map, selected_ids=selected_ids)
        body_nodes.append(moved)
        moved_by_old_id[old_id] = moved
    dag["nodes"] = [
        node
        for node in dag["nodes"]
        if not (isinstance(node, dict) and str(node.get("id")) in selected_ids)
    ]
    _preserve_repeat_state_refs(loop, moved_by_old_id.values())
    _replace_top_level_needs(dag, selected_ids, str(loop["id"]))
    return {old_id: moved["id"] for old_id, moved in moved_by_old_id.items()}


def _prepare_node_for_repeat_body(
    raw: dict[str, Any],
    source: dict[str, Any],
    body_nodes: list[Any],
    *,
    id_map: dict[str, str] | None = None,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    original_id = str(source.get("id") or "node")
    moved = copy.deepcopy(source)
    moved_id = id_map.get(original_id) if id_map else _unique_id(_clean_id(original_id), body_nodes)
    moved_id = moved_id or _unique_id(_clean_id(original_id), body_nodes)
    moved["id"] = moved_id
    moved = _rewrite_repeat_body_external_refs(moved, id_map or {original_id: moved_id}, selected_ids or {original_id})
    moved["inputs"] = _repeat_body_inputs(raw, moved)
    moved.pop("run_on", None)
    moved.pop("best_tex", None)
    moved.pop("retries", None)
    if isinstance(source.get("needs"), list):
        selected = selected_ids or set()
        mapping = id_map or {}
        moved["needs"] = [
            mapping.get(dep, dep)
            for dep in source["needs"]
            if isinstance(dep, str) and dep in selected
        ]
        if not moved["needs"]:
            moved.pop("needs", None)
    else:
        moved.pop("needs", None)
    ui = moved.get("ui")
    if isinstance(ui, dict):
        ui.pop("managed_needs", None)
    return moved


def _replace_top_level_needs(dag: dict[str, Any], removed_ids: set[str], loop_id: str) -> None:
    for node in dag["nodes"]:
        if not isinstance(node, dict):
            continue
        if isinstance(node.get("needs"), list):
            needs: list[str] = []
            saw_removed = False
            for dep in node["needs"]:
                if dep in removed_ids:
                    saw_removed = True
                    continue
                if dep not in needs:
                    needs.append(dep)
            if saw_removed and loop_id not in needs and node.get("id") != loop_id:
                needs.append(loop_id)
            node["needs"] = needs
        ui = node.get("ui")
        if isinstance(ui, dict):
            managed = ui.get("managed_needs")
            if isinstance(managed, list):
                ui["managed_needs"] = [dep for dep in managed if dep not in removed_ids]


def _rewrite_repeat_body_external_refs(value: Any, id_map: dict[str, str], selected_ids: set[str]) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            node_id = match.group(1)
            if node_id in selected_ids:
                return f"$node.{id_map.get(node_id, node_id)}"
            return f"$parent.node.{node_id}"

        return re.sub(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)", repl, value)
    if isinstance(value, list):
        return [_rewrite_repeat_body_external_refs(item, id_map, selected_ids) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_repeat_body_external_refs(item, id_map, selected_ids) for key, item in value.items()}
    return value


def _rewrite_top_level_refs_to_repeat_outputs(
    dag: dict[str, Any],
    loop_id: str,
    moved_pairs: dict[str, str],
    *,
    public_old_ids: set[str] | None = None,
) -> None:
    public_old_ids = public_old_ids or set()
    for node in dag.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("id") == loop_id:
            continue
        _rewrite_refs_in_top_level_node(node, loop_id, moved_pairs, public_old_ids)
    if isinstance(dag.get("outputs"), dict):
        dag["outputs"] = _rewrite_moved_node_refs(dag["outputs"], loop_id, moved_pairs, public_old_ids)


def _rewrite_refs_in_top_level_node(
    node: dict[str, Any],
    loop_id: str,
    moved_pairs: dict[str, str],
    public_old_ids: set[str],
) -> None:
    if _is_repeat_node(node):
        for key in ("inputs", "initial_state", "when", "default", "best_tex", "outputs"):
            if key in node:
                node[key] = _rewrite_moved_node_refs(node[key], loop_id, moved_pairs, public_old_ids)
        return
    for key, value in list(node.items()):
        if key == "ui":
            continue
        node[key] = _rewrite_moved_node_refs(value, loop_id, moved_pairs, public_old_ids)


def _rewrite_moved_node_refs(
    value: Any,
    loop_id: str,
    moved_pairs: dict[str, str],
    public_old_ids: set[str],
) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            old_id = match.group(1)
            field = match.group(2)
            if old_id not in moved_pairs:
                return match.group(0)
            if old_id in public_old_ids:
                if not field:
                    return f"$node.{loop_id}"
                return f"$node.{loop_id}.{field}"
            if not field:
                return f"$node.{loop_id}.state.{moved_pairs[old_id]}"
            return f"$node.{loop_id}.state.{moved_pairs[old_id]}_{field}"

        return re.sub(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)(?:\.([A-Za-z_][A-Za-z0-9_-]*))?", repl, value)
    if isinstance(value, list):
        return [_rewrite_moved_node_refs(item, loop_id, moved_pairs, public_old_ids) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_moved_node_refs(item, loop_id, moved_pairs, public_old_ids)
            for key, item in value.items()
        }
    return value


def _sync_repeat_state_updates_for_body_node(
    raw: dict[str, Any],
    loop: dict[str, Any],
    body_node: dict[str, Any],
    *,
    public_aliases: bool = False,
) -> None:
    body = loop.setdefault("body", {})
    if not isinstance(body, dict):
        body = {}
        loop["body"] = body
    updates = body.setdefault("state_updates", {})
    if not isinstance(updates, dict):
        updates = {}
        body["state_updates"] = updates
    if not public_aliases:
        return
    body_node_id = str(body_node["id"])
    for field in _body_output_fields(raw, body_node):
        if field in REPEAT_RUNTIME_FIELDS:
            continue
        updates[field] = f"$node.{body_node_id}.{field}"


def _preserve_repeat_state_refs(loop: dict[str, Any], body_nodes: Iterable[dict[str, Any]]) -> None:
    fields: set[str] = set()
    for node in body_nodes:
        fields.update(_state_ref_fields(node))
    if not fields:
        return
    initial_state = loop.setdefault("initial_state", {})
    if not isinstance(initial_state, dict):
        initial_state = {}
        loop["initial_state"] = initial_state
    for field in sorted(fields):
        initial_state.setdefault(field, f"$state.{field}")


def _state_ref_fields(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, str):
        fields.update(
            field
            for field in re.findall(r"\$state\.([A-Za-z_][A-Za-z0-9_]*)", value)
            if field not in REPEAT_RUNTIME_FIELDS
        )
    elif isinstance(value, list):
        for item in value:
            fields.update(_state_ref_fields(item))
    elif isinstance(value, dict):
        for item in value.values():
            fields.update(_state_ref_fields(item))
    return fields


def _set_repeat_outputs_from_end_node(
    raw: dict[str, Any],
    loop: dict[str, Any],
    end_id: str,
    moved_pairs: dict[str, str],
) -> None:
    end_id = str(end_id)
    if end_id not in moved_pairs:
        raise PresetError(f"repeat zone end node was not captured: {end_id}")
    end_body_id = moved_pairs[end_id]
    end_node = _repeat_body_node_by_id(loop, end_body_id)
    fields = [field for field in _body_output_fields(raw, end_node) if field not in REPEAT_RUNTIME_FIELDS]
    _sync_repeat_state_updates_for_body_node(raw, loop, end_node, public_aliases=True)
    loop["outputs"] = {field: f"$state.{field}" for field in fields}


def _node_x(node: dict[str, Any]) -> float:
    ui = node.get("ui") if isinstance(node.get("ui"), dict) else {}
    try:
        return float(ui.get("x", 0))
    except (TypeError, ValueError):
        return 0.0


def _node_y(node: dict[str, Any]) -> float:
    ui = node.get("ui") if isinstance(node.get("ui"), dict) else {}
    try:
        return float(ui.get("y", 0))
    except (TypeError, ValueError):
        return 0.0


def _node_ids_on_paths(dag: dict[str, Any], start_id: str, end_id: str) -> set[str]:
    ids = {
        str(node.get("id"))
        for node in dag.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    if start_id not in ids:
        raise PresetError(f"start node not found: {start_id}")
    if end_id not in ids:
        raise PresetError(f"end node not found: {end_id}")
    if start_id == end_id:
        return {start_id}
    adjacency = _top_level_adjacency(dag)
    reverse: dict[str, set[str]] = {node_id: set() for node_id in ids}
    for source, targets in adjacency.items():
        for target in targets:
            reverse.setdefault(target, set()).add(source)
    forward = _reachable(adjacency, start_id)
    if end_id not in forward:
        raise PresetError(f"no directed graph path found from {start_id} to {end_id}")
    reverse_reachable = _reachable(reverse, end_id)
    return forward & reverse_reachable


def _top_level_adjacency(dag: dict[str, Any]) -> dict[str, set[str]]:
    nodes = [node for node in dag.get("nodes", []) if isinstance(node, dict)]
    ids = {str(node.get("id")) for node in nodes if node.get("id") is not None}
    adjacency = {node_id: set() for node_id in ids}
    for node in nodes:
        target = str(node.get("id") or "")
        deps = set(str(dep) for dep in node.get("needs", []) if isinstance(dep, str))
        deps.update(_dependency_ref_ids(node))
        for dep in deps:
            if dep in ids and dep != target:
                adjacency.setdefault(dep, set()).add(target)
    return adjacency


def _reachable(adjacency: dict[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - seen))
    return seen


def _selected_top_level_nodes_for_repeat(
    dag: dict[str, Any],
    selected_ids: set[str],
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = exclude or set()
    selected: list[dict[str, Any]] = []
    for node in dag.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        if node_id not in selected_ids or node_id in exclude:
            continue
        if _is_repeat_node(node):
            raise PresetError("capturing nested repeat zones is not supported yet")
        selected.append(node)
    return selected


def _external_dependencies_for_nodes(dag: dict[str, Any], selected_ids: set[str]) -> set[str]:
    ids = {
        str(node.get("id"))
        for node in dag.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    deps: set[str] = set()
    for node in dag.get("nodes", []):
        if not isinstance(node, dict) or str(node.get("id")) not in selected_ids:
            continue
        node_deps = set(str(dep) for dep in node.get("needs", []) if isinstance(dep, str))
        node_deps.update(_dependency_ref_ids(node))
        deps.update(dep for dep in node_deps if dep in ids and dep not in selected_ids)
    return deps


def _repeat_body_inputs(raw: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    existing = node.get("inputs")
    inputs = dict(existing) if isinstance(existing, dict) else {}
    for field in _prompt_input_fields_for_node(raw, node):
        inputs.setdefault(field, _default_repeat_source(field))
    return inputs


def _prompt_input_fields_for_node(raw: dict[str, Any], node: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    name = str(node.get("name") or "")
    if name:
        cfg = _components(raw).get(name, {})
        input_schema = cfg.get("input_schema") if isinstance(cfg, dict) else {}
        if isinstance(input_schema, dict):
            fields.update(str(key) for key in input_schema)
        for key in ("system_prompt", "user_prompt", "prompt"):
            raw = cfg.get(key) if isinstance(cfg, dict) else None
            if isinstance(raw, str):
                fields.update(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)", raw))
    return fields


def _default_repeat_source(field: str) -> str:
    if field in {"problem", "problem_id"}:
        return f"$input.{field}"
    if field in {"solution", "previous_solution", "current_solution", "tex_body"}:
        return "$state.solution"
    if field == "findings":
        return "$state.findings"
    if field == "bug_report":
        return "$state.findings"
    if field == "iteration":
        return "$loop.iterations"
    return f"$state.{field}"


def _outer_node_refs_to_parent(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$node\.", "$parent.node.", value)
    if isinstance(value, list):
        return [_outer_node_refs_to_parent(item) for item in value]
    if isinstance(value, dict):
        return {key: _outer_node_refs_to_parent(item) for key, item in value.items()}
    return value


def _body_output_fields(raw: dict[str, Any], node: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    kind = str(node.get("kind", "agent"))
    if kind == "map_chain":
        fields.extend(["items", "drafts", "finals"])
        collect = node.get("collect")
        if isinstance(collect, dict):
            fields.extend(str(field) for field in collect)
        return list(dict.fromkeys(field for field in fields if field))
    if kind == "if_else":
        return _if_else_public_output_fields(node)
    if kind == "join_or_agent":
        return [str(node.get("output_field") or "solution")]
    if kind == "workflow_ref":
        fields.extend(_workflow_ref_output_fields(node))
    name = str(node.get("name") or "")
    cfg = _components(raw).get(name, {})
    output = cfg.get("output") if isinstance(cfg, dict) else {}
    if isinstance(output, dict):
        fields.extend(str(field) for field in (output.get("xml_lists") or {}))
        fields.extend(str(field) for field in (output.get("xml_tags") or []))
        if output.get("json_field"):
            fields.append(str(output["json_field"]))
        fields.extend(str(field) for field in (output.get("json_tags") or {}))
        if output.get("default_field"):
            fields.append(str(output["default_field"]))
    output_schema = cfg.get("output_schema") if isinstance(cfg, dict) else {}
    if isinstance(output_schema, dict):
        fields.extend(str(field) for field in output_schema)
    if not fields:
        fields.extend(["solution", "output"])
    return list(dict.fromkeys(field for field in fields if field))


def _workflow_ref_output_fields(node: dict[str, Any]) -> list[str]:
    configured = node.get("outputs")
    if isinstance(configured, dict):
        return [str(field) for field in configured]
    preset_name = str(node.get("preset") or "").strip()
    if not preset_name:
        return []
    try:
        preset = load_preset(preset_name)
    except Exception:
        return []
    workflow_cfg = preset.component_configs.get(preset.workflow_cls.__name__, {})
    dag = workflow_cfg.get("dag") if isinstance(workflow_cfg, dict) else None
    if not isinstance(dag, dict):
        return []
    outputs = dag.get("outputs")
    if not isinstance(outputs, dict):
        return []
    return [str(field) for field in outputs]


def _workflow_input_fields(raw: dict[str, Any]) -> list[str]:
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        return []
    budget = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    return [str(field) for field in inputs if field not in budget]


def _workflow_output_fields(raw: dict[str, Any]) -> list[str]:
    dag = raw.get("dag")
    outputs = dag.get("outputs") if isinstance(dag, dict) else None
    if not isinstance(outputs, dict):
        return []
    return [str(field) for field in outputs]


def _op_tie_component(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_editor_id(raw, str(operation.get("node_id") or ""))
    target_name = str(operation.get("target_name") or "").strip()
    if not target_name:
        raise PresetError("target prompt is required")
    components = _components(raw)
    target_cfg = components.get(target_name)
    if not isinstance(target_cfg, dict):
        raise PresetError(f"prompt config not found: {target_name}")
    old_name = str(node.get("name") or "")
    node["name"] = target_name
    if old_name and old_name != target_name:
        _remove_component_if_unused(raw, old_name)


def _op_untie_component(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node_id = str(operation.get("node_id") or "")
    node = _node_by_editor_id(raw, node_id)
    old_name = str(node.get("name") or "").strip()
    if not old_name:
        raise PresetError("node has no prompt config to untie")
    components = _components(raw)
    old_cfg = components.get(old_name)
    if not isinstance(old_cfg, dict):
        raise PresetError(f"prompt config not found: {old_name}")
    base = f"cfg_{_clean_id(node_id)}"
    new_name = _unique_component_name(components, base)
    components[new_name] = copy.deepcopy(old_cfg)
    node["name"] = new_name
    _remove_component_if_unused(raw, old_name)


def _op_update_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    target_ref = _editor_node_ref(raw, str(operation.get("node_id") or ""))
    node = target_ref.node
    if node is None:
        raise PresetError("node not found")
    fields = operation.get("fields") or {}
    if not isinstance(fields, dict):
        raise PresetError("update_node.fields must be a mapping")

    old_id = str(node["id"])
    new_id = fields.get("id")
    if isinstance(new_id, str) and new_id and new_id != old_id:
        if target_ref.kind != "top":
            raise PresetError("renaming repeat body node ids is not supported")
        clean = _clean_id(new_id)
        if clean != old_id and any(
            isinstance(n, dict) and n.get("id") == clean
            for n in _editable_dag(raw)["nodes"]
        ):
            raise PresetError(f"node id already exists: {clean}")
        node["id"] = clean
        _rename_node_refs(raw, old_id, clean)

    ui = _node_ui(node)
    if "label" in fields:
        label = str(fields.get("label") or node.get("id"))
        ui["label"] = label
    if "subtitle" in fields:
        subtitle = str(fields.get("subtitle") or "").strip()
        if subtitle:
            ui["subtitle"] = subtitle
        else:
            ui.pop("subtitle", None)
    if "collapsed" in fields:
        if bool(fields.get("collapsed")):
            ui["collapsed"] = True
        else:
            ui.pop("collapsed", None)

    for key in ("kind", "agent", "name", "source", "output_field", "foreach"):
        if key in fields:
            value = fields[key]
            if value in ("", None):
                node.pop(key, None)
            else:
                node[key] = value
    if "max_parallel" in fields:
        node["max_parallel"] = int(float(fields["max_parallel"] or 1))
    if "needs" in fields:
        node["needs"] = _parse_csv_list(fields["needs"])


def _op_update_component(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    name = str(operation.get("name") or "")
    if not name:
        raise PresetError("component name is required")
    components = _components(raw)
    cfg = components.setdefault(name, {})
    if not isinstance(cfg, dict):
        raise PresetError(f"component {name!r} must be a mapping")
    fields = operation.get("fields") or {}
    if not isinstance(fields, dict):
        raise PresetError("update_component.fields must be a mapping")

    handled_fields: set[str] = set()

    if "__rename_output_refs" in fields:
        _rename_component_output_refs(raw, name, fields["__rename_output_refs"])
        handled_fields.add("__rename_output_refs")

    for key in ("model", "system_prompt", "user_prompt", "prompt"):
        if key in fields:
            cfg[key] = str(fields[key] or "")
            handled_fields.add(key)
    if "model_reasoning_effort" in fields:
        effort = str(fields["model_reasoning_effort"] or "").strip()
        if effort:
            cfg["model_reasoning_effort"] = effort
        else:
            cfg.pop("model_reasoning_effort", None)
        handled_fields.add("model_reasoning_effort")

    if "input_schema" in fields:
        cfg["input_schema"] = _parse_schema_fields(str(fields["input_schema"] or ""))
        handled_fields.add("input_schema")
    if "tools" in fields:
        tools = _parse_tools_config(fields.get("tools"))
        if tools:
            cfg["tools"] = tools
        else:
            cfg.pop("tools", None)
        handled_fields.add("tools")
    if "tool_refs" in fields:
        refs = _parse_tool_refs(fields.get("tool_refs"))
        if refs:
            cfg["tool_refs"] = refs
        else:
            cfg.pop("tool_refs", None)
        cfg.pop("tools", None)
        handled_fields.add("tool_refs")
    if "max_tool_calls" in fields:
        raw_max = str(fields.get("max_tool_calls") or "").strip()
        if raw_max:
            cfg["max_tool_calls"] = int(float(raw_max))
        else:
            cfg.pop("max_tool_calls", None)
        handled_fields.add("max_tool_calls")

    edits_output = any(key in fields for key in ("default_field", "xml_tags", "xml_list_field", "xml_list_tag"))
    output = cfg.get("output")
    if edits_output or isinstance(output, dict):
        if not isinstance(output, dict):
            output = {}
            cfg["output"] = output
        if "default_field" in fields:
            default_field = str(fields["default_field"] or "").strip()
            if default_field:
                output["default_field"] = default_field
            else:
                output.pop("default_field", None)
            handled_fields.add("default_field")
        if "xml_tags" in fields:
            tags = _parse_csv_list(fields["xml_tags"])
            if tags:
                output["xml_tags"] = tags
            else:
                output.pop("xml_tags", None)
            handled_fields.add("xml_tags")
        if "xml_list_field" in fields or "xml_list_tag" in fields:
            field = str(fields.get("xml_list_field") or "")
            tag = str(fields.get("xml_list_tag") or "")
            if field and tag:
                output["xml_lists"] = {field: tag}
            else:
                output.pop("xml_lists", None)
            handled_fields.update({"xml_list_field", "xml_list_tag"})
        _ensure_multi_output_prompt_instruction(cfg, output)

    for key, value in fields.items():
        if key in handled_fields or str(key).startswith("__"):
            continue
        cfg[str(key)] = value


def _op_update_node_inputs(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_editor_id(raw, str(operation.get("node_id") or ""))
    old_refs = _node_ref_ids(node)
    _apply_wiring_entries(node, operation.get("inputs"), allow_special=True)
    _sync_needs_from_refs(raw, node)
    _prune_unreferenced_needs(node, old_refs)


def _op_update_node_outputs(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_editor_id(raw, str(operation.get("node_id") or ""))
    kind = str(node.get("kind", "agent"))
    if kind == "join_or_agent":
        old_field = str(node.get("output_field") or "solution")
        output_field = _clean_id(str(operation.get("output_field") or "solution"))
        if not output_field:
            raise PresetError("output field is required")
        node["output_field"] = output_field
        if node.get("best_tex") in (None, "", f"$output.{old_field}"):
            node["best_tex"] = f"$output.{output_field}"
        return
    if kind == "if_else":
        if "then_field" in operation or "else_field" in operation:
            _update_if_else_branch_outputs(node, operation)
        else:
            _update_if_else_output_schema(node, operation.get("output_schema"))
        return
    raise PresetError(f"{node.get('id')}: outputs are defined by the component or Python agent")


def _update_if_else_output_schema(node: dict[str, Any], entries: Any) -> None:
    allowed = set(_if_else_public_output_fields(node))
    schema: dict[str, Any] = {}
    if isinstance(entries, dict):
        iterable = entries.items()
    elif isinstance(entries, list):
        iterable = (
            (entry.get("field"), entry)
            for entry in entries
            if isinstance(entry, dict)
        )
    else:
        iterable = ()
    existing = node.get("output_schema") if isinstance(node.get("output_schema"), dict) else {}
    for raw_field, raw_entry in iterable:
        field = str(raw_field or "").strip()
        if field not in allowed:
            continue
        old = existing.get(field) if isinstance(existing.get(field), dict) else {}
        if isinstance(raw_entry, dict):
            description = str(raw_entry.get("description") or "").strip()
            type_name = str(raw_entry.get("type") or old.get("type") or "").strip()
        else:
            description = str(raw_entry or "").strip()
            type_name = str(old.get("type") or "").strip()
        item: dict[str, Any] = {}
        if type_name:
            item["type"] = type_name
        if description:
            item["description"] = description
        if item:
            schema[field] = item
    if schema:
        node["output_schema"] = schema
    else:
        node.pop("output_schema", None)


def _update_if_else_branch_outputs(node: dict[str, Any], operation: dict[str, Any]) -> None:
    _set_if_else_branch_output(
        node,
        "then",
        str(operation.get("then_field") or ""),
        fallback_field="True",
    )
    _set_if_else_branch_output(
        node,
        "else",
        str(operation.get("else_field") or ""),
        fallback_field="False",
    )
    allowed = set(_if_else_public_output_fields(node))
    schema = node.get("output_schema")
    if isinstance(schema, dict):
        node["output_schema"] = {
            field: value
            for field, value in schema.items()
            if field in allowed
        }
        if not node["output_schema"]:
            node.pop("output_schema", None)


def _set_if_else_branch_output(
    node: dict[str, Any],
    branch: str,
    raw_field: str,
    *,
    fallback_field: str,
) -> None:
    field = _clean_id(raw_field) or fallback_field
    node[branch] = {field: True}


def _if_else_public_output_fields(node: dict[str, Any]) -> list[str]:
    fields: list[str] = ["condition"]
    outputs = node.get("outputs")
    if isinstance(outputs, dict):
        fields.extend(str(key) for key in outputs)
    for branch_key in ("then", "else"):
        branch_outputs = node.get(branch_key)
        if isinstance(branch_outputs, dict):
            fields.extend(str(key) for key in branch_outputs)
    public = sorted(dict.fromkeys(field for field in fields if field))
    return public if len(public) > 1 else ["False", "True", "condition"]


def _op_update_node_condition(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    target_ref = _editor_node_ref(raw, str(operation.get("node_id") or ""))
    node = target_ref.node
    if node is None:
        raise PresetError("node not found")
    mode = str(operation.get("mode") or "python")
    body = str(operation.get("body") or "").strip()
    old_refs = _node_ref_ids(node)
    if not body:
        if str(node.get("kind")) == "if_else":
            node["condition"] = {"python": "False"}
        else:
            node.pop("condition", None)
    elif mode in {"yaml", "json"}:
        parsed = yaml.safe_load(body)
        if not isinstance(parsed, (dict, bool, str)):
            raise PresetError("condition YAML must be a mapping, boolean, or string")
        node["condition"] = parsed
    elif mode == "python_code":
        node["condition"] = {"python_code": body}
    elif mode == "equals":
        node["condition"] = {"ref": body, "equals": _parse_editor_value(operation.get("compare_value"))}
    elif mode == "not_equals":
        node["condition"] = {"ref": body, "not_equals": _parse_editor_value(operation.get("compare_value"))}
    elif mode == "ref":
        node["condition"] = _condition_with_length_limits({"ref": body}, operation)
    else:
        node["condition"] = {"python": body}
    if target_ref.kind == "repeat_body":
        if target_ref.loop is None:
            raise PresetError("repeat body target is missing its loop")
        _sync_loop_body_needs(target_ref.loop)
    else:
        _sync_needs_from_refs(raw, node)
        _prune_unreferenced_needs(node, old_refs)


def _op_update_node_when(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_id(raw, str(operation.get("node_id") or ""))
    mode = str(operation.get("mode") or "python")
    body = str(operation.get("body") or "").strip()
    old_refs = _node_ref_ids(node)
    old_when = node.get("when")
    old_inputs = old_when.get("inputs") if isinstance(old_when, dict) and isinstance(old_when.get("inputs"), dict) else None
    if not body:
        node.pop("when", None)
    elif mode in {"yaml", "json"}:
        parsed = yaml.safe_load(body)
        if not isinstance(parsed, (dict, bool, str)):
            raise PresetError("run condition YAML must be a mapping, boolean, or string")
        node["when"] = parsed
    elif mode == "python_code":
        node["when"] = {"python_code": body}
    elif mode == "equals":
        node["when"] = {"ref": body, "equals": _parse_editor_value(operation.get("compare_value"))}
    elif mode == "not_equals":
        node["when"] = {"ref": body, "not_equals": _parse_editor_value(operation.get("compare_value"))}
    elif mode == "ref":
        node["when"] = _condition_with_length_limits({"ref": body}, operation)
    else:
        node["when"] = {"python": body}
    if old_inputs and isinstance(node.get("when"), dict) and mode not in {"yaml", "json"}:
        node["when"]["inputs"] = old_inputs
    _sync_needs_from_refs(raw, node)
    _prune_unreferenced_needs(node, old_refs)


def _condition_with_length_limits(condition: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
    for key, label in (("min_len", "Minimum items"), ("max_len", "Maximum items")):
        value = operation.get(key)
        raw = "" if value is None else str(value).strip()
        if not raw:
            continue
        parsed = _parse_editor_value(raw)
        if not isinstance(parsed, (int, float)):
            raise PresetError(f"{label} must be a number")
        condition[key] = max(0, int(parsed))
    return condition


def _op_update_node_default_outputs(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_id(raw, str(operation.get("node_id") or ""))
    old_refs = _node_ref_ids(node)
    mapping = _entries_to_mapping(operation.get("outputs"))
    if mapping:
        node["default"] = mapping
    else:
        node.pop("default", None)
    _sync_needs_from_refs(raw, node)
    _prune_unreferenced_needs(node, old_refs)


def _op_update_loop_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_id(raw, str(operation.get("node_id") or ""))
    if not _is_repeat_node(node):
        raise PresetError(f"{node.get('id')}: not a repeat node")
    if "max_iterations" in operation:
        parsed_max = _parse_editor_value(operation.get("max_iterations"))
        if isinstance(parsed_max, (int, float)):
            node["max_iterations"] = max(1, int(parsed_max))
        else:
            node["max_iterations"] = parsed_max or 1
    if "condition_python" in operation:
        body = str(operation.get("condition_python") or "").strip()
        old_condition = node.get("condition")
        next_condition = {"python": body or "iteration < max_iterations"}
        if isinstance(old_condition, dict) and isinstance(old_condition.get("inputs"), dict):
            next_condition["inputs"] = dict(old_condition["inputs"])
        node["condition"] = next_condition
    if "initial_state" in operation:
        node["initial_state"] = _entries_to_mapping(operation.get("initial_state"))
    body = node.setdefault("body", {})
    if not isinstance(body, dict):
        body = {}
        node["body"] = body
    if "state_updates" in operation:
        body["state_updates"] = _entries_to_mapping(operation.get("state_updates"))
    if "outputs" in operation:
        node["outputs"] = _entries_to_mapping(operation.get("outputs"))


def _op_update_workflow_ref(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node = _node_by_editor_id(raw, str(operation.get("node_id") or ""))
    if str(node.get("kind")) != "workflow_ref":
        raise PresetError(f"{node.get('id')}: not a subworkflow node")
    old_refs = _dependency_ref_ids(node)
    if "preset" in operation:
        preset = str(operation.get("preset") or "").strip()
        if not preset:
            raise PresetError("subworkflow preset is required")
        node["preset"] = preset
    if "params" in operation:
        node["params"] = _entries_to_mapping(operation.get("params"))
    if "inputs" in operation:
        node["inputs"] = _entries_to_mapping(operation.get("inputs"))
    if "component_overrides" in operation:
        overrides = _parse_editor_mapping(str(operation.get("component_overrides") or ""))
        if overrides:
            node["component_overrides"] = overrides
        else:
            node.pop("component_overrides", None)
    if "model_overrides" in operation:
        overrides = _parse_editor_mapping(str(operation.get("model_overrides") or ""))
        if overrides:
            node["model_overrides"] = overrides
        else:
            node.pop("model_overrides", None)
    _sync_needs_from_refs(raw, node)
    _prune_unreferenced_needs(node, old_refs)


def _op_update_workflow_inputs(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    budget = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    raw["inputs"] = {
        key: value
        for key, value in _entries_to_mapping(operation.get("inputs")).items()
        if key not in budget
    }


def _op_update_workflow_budget(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    budget = _entries_to_mapping(operation.get("budget"))
    if budget:
        raw["budget"] = budget
    else:
        raw.pop("budget", None)


def _workflow_outputs(raw: dict[str, Any]) -> dict[str, Any]:
    dag = _editable_dag(raw)
    outputs = dag.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        dag["outputs"] = outputs
    return outputs


def _clean_output_field(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"^[^A-Za-z_]+", "", cleaned)
    return cleaned.strip("_")


def _op_add_workflow_output(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    outputs = _workflow_outputs(raw)
    base = _clean_output_field(str(operation.get("field") or "")) or "solution"
    field = base
    idx = 2
    while field in outputs:
        field = f"{base}_{idx}"
        idx += 1
    outputs[field] = _parse_editor_value(operation.get("value", ""))


def _op_update_workflow_output(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    outputs = _workflow_outputs(raw)
    old_field = _clean_output_field(str(operation.get("field") or ""))
    new_field = _clean_output_field(str(operation.get("new_field") or old_field))
    if not old_field:
        raise PresetError("workflow output field is required")
    if not new_field:
        raise PresetError("workflow output name is required")
    if old_field not in outputs:
        outputs[old_field] = ""
    if new_field != old_field and new_field in outputs:
        raise PresetError(f"workflow output already exists: {new_field}")
    value = _parse_editor_value(operation.get("value"))
    updated: dict[str, Any] = {}
    for field, current in outputs.items():
        if field == old_field:
            updated[new_field] = value
        else:
            updated[field] = current
    dag = _editable_dag(raw)
    dag["outputs"] = updated


def _op_delete_workflow_output(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    field = _clean_output_field(str(operation.get("field") or ""))
    if not field:
        raise PresetError("workflow output field is required")
    _workflow_outputs(raw).pop(field, None)


def _op_update_workflow_output_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    fields = operation.get("fields") or {}
    if not isinstance(fields, dict):
        raise PresetError("workflow output node fields must be a mapping")
    output_ui = _workflow_output_ui(raw)
    if "label" in fields:
        label = str(fields.get("label") or "").strip()
        if label:
            output_ui["label"] = label
        else:
            output_ui.pop("label", None)
    if "subtitle" in fields:
        subtitle = str(fields.get("subtitle") or "").strip()
        if subtitle:
            output_ui["subtitle"] = subtitle
        else:
            output_ui.pop("subtitle", None)
    if "collapsed" in fields:
        if bool(fields.get("collapsed")):
            output_ui["collapsed"] = True
        else:
            output_ui.pop("collapsed", None)


def _op_move_workflow_output_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    output_ui = _workflow_output_ui(raw)
    output_ui["x"] = int(float(operation.get("x", 0)))
    output_ui["y"] = int(float(operation.get("y", 0)))


def _apply_wiring_entries(
    spec: dict[str, Any],
    entries: Any,
    *,
    allow_special: bool,
) -> None:
    mapping = _entries_to_mapping(entries)
    special_fields = _special_fields_for_spec(spec) if allow_special else set()
    inputs: dict[str, Any] = {}
    for field, value in mapping.items():
        if field in special_fields:
            if value == "":
                spec.pop(field, None)
            else:
                spec[field] = value
        else:
            inputs[field] = value
    if inputs:
        spec["inputs"] = inputs
    else:
        spec.pop("inputs", None)


def _entries_to_mapping(entries: Any) -> dict[str, Any]:
    if isinstance(entries, dict):
        iterable = entries.items()
    elif isinstance(entries, list):
        iterable = (
            (entry.get("field"), entry.get("value"))
            for entry in entries
            if isinstance(entry, dict)
        )
    else:
        iterable = ()
    out: dict[str, Any] = {}
    for raw_field, raw_value in iterable:
        raw_name = str(raw_field or "").strip()
        if not raw_name:
            continue
        field = _clean_id(raw_name)
        if not field:
            continue
        out[field] = _parse_editor_value(raw_value)
    return out


def _parse_editor_value(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    if text.startswith("$"):
        return text
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _parse_editor_mapping(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise PresetError(f"invalid YAML/JSON mapping: {e}") from e
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise PresetError("expected a YAML/JSON mapping")
    return parsed


def _condition_from_editor(mode: str, body: str, *, default: Any = None) -> Any:
    if not body:
        return default if default is not None else None
    if mode in {"yaml", "json"}:
        parsed = yaml.safe_load(body)
        if not isinstance(parsed, (dict, bool, str)):
            raise PresetError("condition YAML must be a mapping, boolean, or string")
        return parsed
    if mode == "python_code":
        return {"python_code": body}
    if mode == "equals":
        return {"ref": body, "equals": ""}
    if mode == "not_equals":
        return {"ref": body, "not_equals": ""}
    if mode == "ref":
        return {"ref": body}
    return {"python": body}


def _sync_needs_from_refs(raw: dict[str, Any], node: dict[str, Any]) -> None:
    node_id = str(node.get("id") or "")
    known = {
        str(item.get("id"))
        for item in _editable_dag(raw)["nodes"]
        if isinstance(item, dict) and item.get("id") is not None
    }
    refs = {
        ref
        for ref in _dependency_ref_ids(node)
        if ref in known and ref != node_id
    }
    needs = node.get("needs", [])
    if not isinstance(needs, list):
        needs = []
        node["needs"] = needs
    managed = _managed_needs(node)
    for dep in sorted(managed - refs):
        if dep in needs:
            needs.remove(dep)
    managed.intersection_update(refs)
    for ref in sorted(refs):
        if ref not in needs:
            needs.append(ref)
        managed.add(ref)
    if needs:
        node["needs"] = needs
    else:
        node.pop("needs", None)
    _set_managed_needs(node, managed)


def _sync_loop_body_needs(node: dict[str, Any]) -> None:
    body = node.get("body") if isinstance(node.get("body"), dict) else {}
    steps = body.get("nodes") if isinstance(body, dict) else None
    if not isinstance(steps, list):
        return
    known = {
        str(step.get("id"))
        for step in steps
        if isinstance(step, dict) and step.get("id") is not None
    }
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        if not step_id:
            continue
        refs = {
            ref
            for ref in _node_ref_ids(step)
            if ref in known and ref != step_id
        }
        existing = step.get("needs")
        needs = [
            str(dep)
            for dep in existing
            if str(dep) in known and str(dep) != step_id
        ] if isinstance(existing, list) else []
        for ref in sorted(refs):
            if ref not in needs:
                needs.append(ref)
        if needs:
            step["needs"] = needs
        else:
            step.pop("needs", None)


def _prune_unreferenced_needs(node: dict[str, Any], candidates: set[str]) -> None:
    if not candidates:
        return
    current_refs = _dependency_ref_ids(node)
    needs = node.get("needs")
    if not isinstance(needs, list):
        return
    managed = _managed_needs(node)
    for ref in sorted(candidates):
        if ref in current_refs:
            continue
        while ref in needs:
            needs.remove(ref)
        managed.discard(ref)
    if needs:
        node["needs"] = needs
    else:
        node.pop("needs", None)
    _set_managed_needs(node, managed)


def _managed_needs(node: dict[str, Any]) -> set[str]:
    ui = node.get("ui")
    if not isinstance(ui, dict):
        return set()
    raw = ui.get("managed_needs")
    if not isinstance(raw, list):
        return set()
    return {str(dep) for dep in raw if isinstance(dep, str)}


def _set_managed_needs(node: dict[str, Any], managed: set[str]) -> None:
    if managed:
        ui = _node_ui(node)
        ui["managed_needs"] = sorted(managed)
    else:
        ui = node.get("ui")
        if isinstance(ui, dict):
            ui.pop("managed_needs", None)


def _special_fields_for_spec(spec: dict[str, Any]) -> set[str]:
    kind = str(spec.get("kind", "agent"))
    if kind == "map_chain":
        return {"foreach"}
    if kind == "join_or_agent":
        return {"source"}
    return set()


def _fallback_output_field(target_field: str) -> str | None:
    if target_field == "__fallback":
        return ""
    if target_field.startswith("__fallback."):
        return _clean_output_field(target_field.split(".", 1)[1])
    return None


def _set_condition_ref(node: dict[str, Any], expr: Any) -> None:
    current = node.get("when")
    if isinstance(current, dict):
        updated = dict(current)
        updated["ref"] = expr
        node["when"] = updated
    else:
        node["when"] = {"ref": expr}


def _set_branch_condition_ref(node: dict[str, Any], source_field: str, expr: Any) -> None:
    field = _clean_output_field(source_field) or "condition"
    node["when"] = {
        "inputs": {field: expr},
        "python": f"inputs.get({json.dumps(field)})",
    }


def _is_if_branch_output(node: dict[str, Any] | None, field: str) -> bool:
    if not isinstance(node, dict) or str(node.get("kind")) != "if_else":
        return False
    then_fields = {str(key) for key in (node.get("then") or {}) if key}
    else_fields = {str(key) for key in (node.get("else") or {}) if key}
    common_fields = {str(key) for key in (node.get("outputs") or {}) if key}
    return bool(field) and field not in common_fields and (field in then_fields or field in else_fields)


def _set_fallback_ref(node: dict[str, Any], target_field: str, expr: Any) -> None:
    fallback_field = _fallback_output_field(target_field)
    if fallback_field is None:
        return
    current = node.get("default")
    if not fallback_field and isinstance(current, dict) and len(current) == 1:
        fallback_field = next(iter(current))
    if not fallback_field:
        fallback_field = "solution"
    default = current if isinstance(current, dict) else {}
    default = dict(default)
    default[fallback_field] = expr
    node["default"] = default


def _repeat_state_updates(loop: dict[str, Any]) -> dict[str, Any]:
    body = loop.setdefault("body", {})
    if not isinstance(body, dict):
        body = {}
        loop["body"] = body
    updates = body.setdefault("state_updates", {})
    if not isinstance(updates, dict):
        raise PresetError(f"{loop.get('id')}: repeat body.state_updates must be a mapping")
    return updates


def _node_ref_ids(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(match.group(1) for match in re.finditer(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)", value))
    elif isinstance(value, list):
        for item in value:
            refs.update(_node_ref_ids(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(_node_ref_ids(item))
    return refs


def _dependency_ref_ids(node: dict[str, Any]) -> set[str]:
    if not _is_repeat_node(node):
        return _node_ref_ids(node)
    refs: set[str] = set()
    for key in ("inputs", "initial_state", "when", "default", "best_tex"):
        if key in node:
            refs.update(_node_ref_ids(node[key]))
    return refs


def _op_connect_edge(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    source_raw = str(operation.get("source_node") or "").strip()
    target_raw = str(operation.get("target_node") or "").strip()
    source_field = str(operation.get("source_field") or "").strip()
    target_field = str(operation.get("target_field") or "").strip()
    transform = str(operation.get("transform") or "direct")
    if not source_raw or not target_raw or not source_field or not target_field:
        raise PresetError("source node/field and target node/field are required")
    if source_raw == target_raw:
        raise PresetError("self edges are not supported in the visual editor")
    source_ref = _editor_node_ref(raw, source_raw)
    target_ref = _editor_node_ref(raw, target_raw)
    if target_ref.kind == "workflow_output":
        output_field = _clean_output_field(target_field)
        if not output_field:
            raise PresetError("workflow output field is required")
        expr = _edge_expression(
            source_ref.node_id,
            source_field,
            transform,
            operation,
            source_ref=source_ref,
            target_scope="top",
        )
        _add_workflow_output_source(_workflow_outputs(raw), output_field, expr)
        return

    target_scope = "repeat_body" if target_ref.kind in {"repeat_body", "repeat_output"} else "top"
    expr = _edge_expression(
        source_ref.node_id,
        source_field,
        transform,
        operation,
        source_ref=source_ref,
        target_scope=target_scope,
    )
    if target_ref.kind == "repeat_input":
        loop = target_ref.loop
        if loop is None:
            raise PresetError("repeat input target is missing its loop")
        initial_state = loop.setdefault("initial_state", {})
        if not isinstance(initial_state, dict):
            raise PresetError(f"{target_ref.loop_id}: initial_state must be a mapping")
        initial_state[target_field] = expr
        _sync_needs_from_refs(raw, loop)
        return
    if target_ref.kind == "repeat_output":
        loop = target_ref.loop
        if loop is None:
            raise PresetError("repeat output target is missing its loop")
        _repeat_state_updates(loop)[target_field] = expr
        _sync_loop_body_needs(loop)
        return

    node = target_ref.node
    if node is None:
        raise PresetError("target node not found")
    if target_field == "__condition":
        if _is_if_branch_output(source_ref.node, source_field):
            _set_branch_condition_ref(node, source_field, expr)
        else:
            _set_condition_ref(node, expr)
    elif _fallback_output_field(target_field) is not None:
        _set_fallback_ref(node, target_field, expr)
    elif target_field in _special_fields_for_spec(node):
        node[target_field] = expr
    elif target_ref.kind == "top" and _is_repeat_node(node):
        initial_state = node.setdefault("initial_state", {})
        if not isinstance(initial_state, dict):
            raise PresetError(f"{target_ref.node_id}: initial_state must be a mapping")
        initial_state[target_field] = expr
    else:
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            raise PresetError(f"{target_ref.node_id}: inputs must be a mapping")
        if _is_named_multi_source_input(target_field):
            _add_multi_input_source(inputs, target_field, source_ref, source_field, expr)
        else:
            _add_coalesced_input_source(inputs, target_field, expr)
    if target_ref.kind == "repeat_body":
        if target_ref.loop is None:
            raise PresetError("repeat body target is missing its loop")
        _sync_loop_body_needs(target_ref.loop)
    else:
        _sync_needs_from_refs(raw, node)


def _op_disconnect_edge(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    target_raw = str(operation.get("target_node") or "").strip()
    target_field = str(operation.get("target_field") or "").strip()
    if not target_raw or not target_field:
        raise PresetError("target node and field are required")
    target_ref = _editor_node_ref(raw, target_raw)
    if target_ref.kind == "workflow_output":
        output_field = _clean_output_field(target_field)
        if not output_field:
            raise PresetError("workflow output field is required")
        source_raw = str(operation.get("source_node") or "").strip()
        source_field = str(operation.get("source_field") or "").strip()
        if source_raw and source_field:
            source_ref = _editor_node_ref(raw, source_raw)
            expr = _edge_expression(
                source_ref.node_id,
                source_field,
                str(operation.get("transform") or "direct"),
                operation,
                source_ref=source_ref,
                target_scope="top",
            )
            _remove_workflow_output_source(_workflow_outputs(raw), output_field, expr)
        else:
            _workflow_outputs(raw)[output_field] = ""
        return
    if target_ref.kind == "repeat_input":
        loop = target_ref.loop
        if loop is None:
            raise PresetError("repeat input target is missing its loop")
        old_value = None
        initial_state = loop.get("initial_state")
        if isinstance(initial_state, dict):
            old_value = initial_state.pop(target_field, None)
            if not initial_state:
                loop.pop("initial_state", None)
        _sync_needs_from_refs(raw, loop)
        _prune_unreferenced_needs(loop, _node_ref_ids(old_value))
        return
    if target_ref.kind == "repeat_output":
        loop = target_ref.loop
        if loop is None:
            raise PresetError("repeat output target is missing its loop")
        old_value = _repeat_state_updates(loop).pop(target_field, None)
        return
    node = target_ref.node
    if node is None:
        raise PresetError("target node not found")
    old_value = None
    fallback_field = _fallback_output_field(target_field)
    if target_field == "__condition":
        old_value = node.pop("when", None)
    elif fallback_field is not None:
        default = node.get("default")
        if fallback_field and isinstance(default, dict):
            old_value = default.pop(fallback_field, None)
            if not default:
                node.pop("default", None)
        else:
            old_value = node.pop("default", None)
    elif target_field in _special_fields_for_spec(node):
        old_value = node.pop(target_field, None)
    elif _is_repeat_node(node) and isinstance(node.get("initial_state"), dict):
        old_value = node["initial_state"].pop(target_field, None)
        if not node["initial_state"]:
            node.pop("initial_state", None)
    else:
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            if _is_named_multi_source_input(target_field):
                source_raw = str(operation.get("source_node") or "").strip()
                source_field = str(operation.get("source_field") or "").strip()
                if source_raw and source_field:
                    source_ref = _editor_node_ref(raw, source_raw)
                    expr = _edge_expression(
                        source_ref.node_id,
                        source_field,
                        str(operation.get("transform") or "direct"),
                        operation,
                        source_ref=source_ref,
                        target_scope="repeat_body" if target_ref.kind == "repeat_body" else "top",
                    )
                    old_value = _remove_multi_input_source(inputs, target_field, expr)
                else:
                    old_value = inputs.get(target_field)
                    inputs[target_field] = {}
            elif target_field in inputs:
                source_raw = str(operation.get("source_node") or "").strip()
                source_field = str(operation.get("source_field") or "").strip()
                if source_raw and source_field:
                    source_ref = _editor_node_ref(raw, source_raw)
                    expr = _edge_expression(
                        source_ref.node_id,
                        source_field,
                        str(operation.get("transform") or "direct"),
                        operation,
                        source_ref=source_ref,
                        target_scope="repeat_body" if target_ref.kind == "repeat_body" else "top",
                    )
                    old_value = _remove_coalesced_input_source(inputs, target_field, expr)
                else:
                    old_value = inputs.get(target_field)
                    inputs[target_field] = f"$input.{target_field}"
    if target_ref.kind == "repeat_body":
        if target_ref.loop is None:
            raise PresetError("repeat body target is missing its loop")
        _sync_loop_body_needs(target_ref.loop)
    else:
        _sync_needs_from_refs(raw, node)
    _prune_unreferenced_needs(node, _node_ref_ids(old_value))


def _op_reconnect_edge(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    old_target_raw = str(operation.get("old_target_node") or "").strip()
    old_field = str(operation.get("old_target_field") or "").strip()
    if not old_target_raw or not old_field:
        raise PresetError("existing target node and field are required")
    _op_disconnect_edge(
        raw,
        {
            "source_node": operation.get("source_node"),
            "source_field": operation.get("source_field"),
            "target_node": old_target_raw,
            "target_field": old_field,
            "transform": operation.get("transform") or "direct",
        },
    )
    new_target_raw = str(operation.get("target_node") or "").strip()
    new_field = str(operation.get("target_field") or "").strip()
    if not new_target_raw and not new_field:
        return
    if not new_target_raw or not new_field:
        raise PresetError("new target node and field are required")
    _op_connect_edge(
        raw,
        {
            "source_node": operation.get("source_node"),
            "source_field": operation.get("source_field"),
            "target_node": new_target_raw,
            "target_field": new_field,
            "transform": operation.get("transform") or "direct",
        },
    )


def _is_workflow_output_target(target: str) -> bool:
    return target in {"outputs", WORKFLOW_OUTPUT_NODE_ID}


def _op_delete_node(raw: dict[str, Any], operation: dict[str, Any]) -> None:
    node_id = _clean_id(str(operation.get("node_id") or ""))
    dag = _editable_dag(raw)
    before = len(dag["nodes"])
    removed_names = [
        str(node.get("name"))
        for node in dag["nodes"]
        if isinstance(node, dict) and node.get("id") == node_id and node.get("name")
    ]
    dag["nodes"] = [
        node
        for node in dag["nodes"]
        if not (isinstance(node, dict) and node.get("id") == node_id)
    ]
    if len(dag["nodes"]) == before:
        raise PresetError(f"node not found: {node_id}")
    for node in dag["nodes"]:
        if isinstance(node, dict) and isinstance(node.get("needs"), list):
            node["needs"] = [dep for dep in node["needs"] if dep != node_id]
        if isinstance(node, dict) and isinstance(node.get("ui"), dict):
            managed = node["ui"].get("managed_needs")
            if isinstance(managed, list):
                node["ui"]["managed_needs"] = [dep for dep in managed if dep != node_id]
    mapping = dag.get("outputs")
    if isinstance(mapping, dict):
        dag["outputs"] = {
            field: _without_workflow_output_node(value, node_id)
            for field, value in mapping.items()
        }
    for name in removed_names:
        _remove_component_if_unused(raw, name)


def _add_workflow_output_source(outputs: dict[str, Any], field: str, expr: Any) -> None:
    current = outputs.get(field)
    if _empty_output_source(current):
        outputs[field] = expr
        return
    sources = _workflow_output_sources(current)
    if any(source == expr for source in sources):
        outputs[field] = _coalesce_output_sources(sources)
        return
    outputs[field] = _coalesce_output_sources([*sources, expr])


def _remove_workflow_output_source(outputs: dict[str, Any], field: str, expr: Any) -> None:
    current = outputs.get(field)
    if _empty_output_source(current):
        outputs[field] = ""
        return
    sources = [source for source in _workflow_output_sources(current) if source != expr]
    outputs[field] = _coalesce_output_sources(sources)


def _without_workflow_output_node(value: Any, node_id: str) -> Any:
    sources = [
        source
        for source in _workflow_output_sources(value)
        if node_id not in _node_ref_ids(source)
    ]
    return _coalesce_output_sources(sources)


def _workflow_output_sources(value: Any) -> list[Any]:
    if _empty_output_source(value):
        return []
    if isinstance(value, dict) and isinstance(value.get("coalesce"), list):
        return list(value["coalesce"])
    return [value]


def _coalesce_output_sources(sources: list[Any]) -> Any:
    sources = [source for source in sources if not _empty_output_source(source)]
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0]
    return {"coalesce": sources}


def _empty_output_source(value: Any) -> bool:
    return value is None or value == "" or value == []


def _is_named_multi_source_input(field: str) -> bool:
    return field == "available_outputs"


def _add_coalesced_input_source(inputs: dict[str, Any], field: str, expr: Any) -> None:
    current = inputs.get(field)
    if _empty_output_source(current) or current == f"$input.{field}":
        inputs[field] = expr
        return
    sources = _coalesced_input_sources(current)
    if expr not in sources:
        sources.append(expr)
    inputs[field] = _coalesce_input_sources(sources, fallback=f"$input.{field}")


def _remove_coalesced_input_source(inputs: dict[str, Any], field: str, expr: Any) -> Any:
    current = inputs.get(field)
    if isinstance(current, dict) and isinstance(current.get("coalesce"), list):
        sources = list(current["coalesce"])
        remaining = [source for source in sources if source != expr]
        removed = expr if len(remaining) != len(sources) else None
        inputs[field] = _coalesce_input_sources(remaining, fallback=f"$input.{field}")
        return removed
    if current == expr:
        inputs[field] = f"$input.{field}"
        return current
    return None


def _coalesced_input_sources(value: Any) -> list[Any]:
    if _empty_output_source(value):
        return []
    if isinstance(value, dict) and isinstance(value.get("coalesce"), list):
        return list(value["coalesce"])
    return [value]


def _coalesce_input_sources(sources: list[Any], *, fallback: Any) -> Any:
    sources = [source for source in sources if not _empty_output_source(source)]
    if not sources:
        return fallback
    if len(sources) == 1:
        return sources[0]
    return {"coalesce": sources}


def _add_multi_input_source(
    inputs: dict[str, Any],
    field: str,
    source_ref: EditorNodeRef,
    source_field: str,
    expr: Any,
) -> None:
    sources = _multi_input_sources(inputs.get(field), field)
    key = _unique_input_source_key(sources, _input_source_key(source_ref, source_field), expr)
    sources[key] = expr
    inputs[field] = sources


def _remove_multi_input_source(inputs: dict[str, Any], field: str, expr: Any) -> Any:
    sources = _multi_input_sources(inputs.get(field), field)
    removed: Any = None
    for key, value in list(sources.items()):
        if value == expr:
            removed = value
            del sources[key]
    inputs[field] = sources
    return removed


def _multi_input_sources(value: Any, field: str) -> dict[str, Any]:
    if _empty_output_source(value):
        return {}
    if isinstance(value, str) and value == f"$input.{field}":
        return {}
    if isinstance(value, dict) and not _is_expression_dict(value):
        return dict(value)
    if isinstance(value, dict) and isinstance(value.get("coalesce"), list):
        out: dict[str, Any] = {}
        for item in value["coalesce"]:
            key = _unique_input_source_key(out, _input_source_key_from_expr(item), item)
            out[key] = item
        return out
    key = _input_source_key_from_expr(value)
    return {key: value}


def _is_expression_dict(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("coalesce", "join", "len", "format", "bug_report_from_findings", "not", "bare_wrap"))


def _input_source_key(source_ref: EditorNodeRef, source_field: str) -> str:
    source = source_ref.body_node_id if source_ref.kind == "repeat_body" else source_ref.node_id
    return f"{source}.{source_field or 'output'}"


def _input_source_key_from_expr(value: Any) -> str:
    if isinstance(value, str):
        match = re.search(r"\$(?:parent\.)?node\.([A-Za-z_][A-Za-z0-9_-]*)(?:\.([A-Za-z_][A-Za-z0-9_.-]*))?", value)
        if match:
            return f"{match.group(1)}.{match.group(2) or 'output'}"
    return "value"


def _unique_input_source_key(sources: dict[str, Any], key: str, expr: Any) -> str:
    key = str(key or "value")
    if key not in sources or sources.get(key) == expr:
        return key
    idx = 2
    while f"{key}_{idx}" in sources and sources[f"{key}_{idx}"] != expr:
        idx += 1
    return f"{key}_{idx}"


def _edge_expression(
    source: str,
    source_field: str,
    transform: str,
    operation: dict[str, Any],
    *,
    source_ref: EditorNodeRef | None = None,
    target_scope: str = "top",
) -> Any:
    ref = _edge_source_ref(source, source_field, source_ref, target_scope)
    return _edge_expression_from_ref(ref, transform, operation)


def _edge_source_ref(
    source: str,
    source_field: str,
    source_ref: EditorNodeRef | None,
    target_scope: str,
) -> str:
    source_ref = source_ref or EditorNodeRef("top", source)
    if source_ref.kind == "top":
        if target_scope == "repeat_body":
            return f"$parent.node.{source_ref.node_id}.{source_field}"
        return f"$node.{source_ref.node_id}.{source_field}"
    if source_ref.kind == "repeat_body":
        if target_scope != "repeat_body":
            raise PresetError("repeat body outputs must first connect to the repeat output node")
        return f"$node.{source_ref.body_node_id}.{source_field}"
    if source_ref.kind == "repeat_input":
        if target_scope != "repeat_body":
            raise PresetError("repeat memory can only be wired inside its repeat zone")
        if source_field == "iteration":
            return "$iteration"
        return f"$state.{source_field}"
    if source_ref.kind == "repeat_output":
        if target_scope == "repeat_body":
            raise PresetError("repeat outputs cannot be wired back into the same repeat body")
        return f"$node.{source_ref.loop_id}.{source_field}"
    raise PresetError("workflow outputs cannot be used as edge sources")


def _edge_expression_from_ref(ref: str, transform: str, operation: dict[str, Any]) -> Any:
    if transform == "direct":
        return ref
    if transform == "join":
        return {"join": ref, "sep": str(operation.get("sep") or "\n\n")}
    if transform == "len":
        return {"len": ref}
    if transform == "bug_report_from_findings":
        return {"bug_report_from_findings": ref}
    if transform == "format":
        template = str(operation.get("template") or "{value}")
        return {"format": template, "fields": {"value": ref}}
    raise PresetError(f"unknown edge transform: {transform}")


def _ensure_prompt_component(raw: dict[str, Any], base_id: str, template: str) -> str:
    components = _components(raw)
    component_name = f"cfg_{_clean_id(base_id)}"
    if component_name not in components:
        components[component_name] = _prompt_component_template(template)
    return component_name


def _ensure_cli_component(raw: dict[str, Any], base_id: str) -> str:
    components = _components(raw)
    component_name = f"cfg_{_clean_id(base_id)}"
    if component_name not in components:
        components[component_name] = _cli_component_template()
    return component_name


def _ensure_latex_cli_component(raw: dict[str, Any], base_id: str) -> str:
    components = _components(raw)
    component_name = f"cfg_{_clean_id(base_id)}"
    if component_name not in components:
        components[component_name] = _latex_cli_component_template()
    return component_name


def _ensure_agent_component(raw: dict[str, Any], base_id: str, agent_path: str) -> str:
    components = _components(raw)
    component_name = f"cfg_{_clean_id(base_id)}"
    if component_name not in components:
        components[component_name] = _default_component_config_for_agent(agent_path)
    return component_name


def _default_component_config_for_agent(agent_path: str) -> dict[str, Any]:
    try:
        cls = _import_class(agent_path)
        raw = getattr(cls, "default_component_config", None)
        if callable(raw):
            cfg = raw()
            return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}
    return {}


def _default_python_agent_inputs(agent_path: str) -> dict[str, Any]:
    fields = _python_agent_input_fields(agent_path)
    return {
        field: f"$input.{field}"
        for field in ("problem", "problem_id")
        if field in fields
    }


def _python_agent_input_fields(agent_path: str) -> set[str]:
    try:
        cls = _import_class(agent_path)
        inputs = getattr(cls, "Inputs", None)
        model_fields = getattr(inputs, "model_fields", {}) or {}
        return {str(field) for field in model_fields}
    except Exception:
        return set()


def _python_agent_has_output(agent_path: str, field: str) -> bool:
    try:
        cls = _import_class(agent_path)
        outputs = getattr(cls, "Outputs", None)
        model_fields = getattr(outputs, "model_fields", {}) or {}
        return field in model_fields
    except Exception:
        return False


def _agent_base_id(agent_path: str) -> str:
    if not agent_path:
        return "python_agent"
    return _clean_id(agent_path.rsplit(".", 1)[-1])


def _import_class(path: str) -> type:
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise PresetError(f"invalid agent path: {path}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise PresetError(f"agent path is not a class: {path}")
    return obj


def _cli_component_template() -> dict[str, Any]:
    return {
        "cmd": [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
        ],
        "model": "gpt-5.4-mini",
        "model_reasoning_effort": "low",
        "codex_sandbox": "auto",
        "copy_codex_auth": True,
        "prompt": (
            "Complete this CLI task. Write any requested files in the workspace. "
            "When finished, run: finish '{\"status\":\"done\",\"summary\":\"completed\"}'"
        ),
        "input_schema": {},
        "sandbox": {"timeout_s": 900, "backend": "docker", "docker_no_new_privileges": False},
        "output_schema": {"workspace": "string", "status": "string", "summary": "string"},
        "done_outputs": {"status": "status", "summary": "summary"},
        "usage": {"type": "codex_jsonl", "model": "gpt-5.4-mini", "cost_config": "models/openai/gpt-54-mini"},
    }


def _latex_cli_component_template() -> dict[str, Any]:
    return {
        "cmd": [
            "sh",
            "-c",
            (
                "set +e\n"
                "rm -f main.aux main.bbl main.blg main.log main.pdf pages.txt compile.log\n"
                "pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > compile.log 2>&1\n"
                "if [ -f main.pdf ]; then\n"
                "  pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2; found=1} END {if (!found) print 0}' > pages.txt || printf '0\\n' > pages.txt\n"
                "  finish '{\"status\":\"done\",\"summary\":\"compiled\"}'\n"
                "else\n"
                "  printf '0\\n' > pages.txt\n"
                "  finish '{\"status\":\"error\",\"summary\":\"LaTeX compile failed; see compile.log\"}'\n"
                "fi\n"
            ),
        ],
        "sandbox": {"timeout_s": 120, "cpu_limit": 2, "memory_gb": 2, "docker_no_new_privileges": False},
        "input_schema": {"tex_body": "string"},
        "input_files": {
            "main.tex": {
                "template": (
                    "\\documentclass[11pt]{article}\n"
                    "\\usepackage[utf8]{inputenc}\n"
                    "\\usepackage[T1]{fontenc}\n"
                    "\\usepackage{amsmath,amssymb,amsthm}\n"
                    "\\usepackage{hyperref}\n"
                    "\\title{Solution}\n"
                    "\\date{}\n"
                    "\\begin{document}\n"
                    "\\maketitle\n\n"
                    "{tex_body}\n\n"
                    "\\end{document}\n"
                )
            }
        },
        "output_schema": {
            "tex": "string",
            "tex_path": "string",
            "pdf_path": "string",
            "compiled": "boolean",
            "pages": "integer",
            "notes": "string",
        },
        "output_files": {
            "tex": "main.tex",
            "tex_path": {"path": "main.tex", "type": "path"},
            "pdf_path": {"path": "main.pdf", "type": "path"},
            "compiled": {"path": "main.pdf", "type": "exists"},
            "pages": {"path": "pages.txt", "type": "int", "default": 0},
            "notes": {"path": "compile.log", "default": ""},
        },
    }


def _prompt_component_template(template: str) -> dict[str, Any]:
    if template == "prompt_agent":
        return {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "You are an expert mathematical assistant.",
            "user_prompt": "Complete this node's task.",
            "input_schema": {},
            "output": {"default_field": "output"},
        }
    if template == "ideator":
        return {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": (
                "You are an expert mathematician. Given a problem, propose distinct "
                "proof strategies. Emit each strategy in its own <approach> tag."
            ),
            "user_prompt": "Problem:\n\n{problem}\n\nSuggest exactly {n} distinct approaches.",
            "input_schema": {"problem": "string", "n": "integer"},
            "output": {"xml_lists": {"approaches": "approach"}, "default_field": "text"},
        }
    if template == "budget_fallback":
        return {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": (
                "You are the emergency fallback for a mathematical proof workflow. "
                "Use the partial work available and return the best complete proof "
                "you can. Emit only a LaTeX body inside <solution>...</solution>."
            ),
            "user_prompt": (
                "Problem:\n{problem}\n\n"
                "Best proof so far:\n{best_tex}\n\n"
                "Budget error:\n{budget_error}\n\n"
                "Available node outputs:\n{available_outputs}\n\n"
                "Synthesize one final proof."
            ),
            "input_schema": {
                "problem": "string",
                "best_tex": "string",
                "budget_error": "string",
                "available_outputs": "object",
            },
            "output": {"xml_tags": ["solution"], "default_field": "solution"},
        }
    if template == "loop_step":
        return {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": (
                "You are improving a mathematical proof inside a bounded loop. "
                "Return the updated proof body inside <solution>...</solution>."
            ),
            "user_prompt": (
                "Problem:\n{problem}\n\n"
                "Current proof:\n{current_solution}\n\n"
                "Improve or complete the proof."
            ),
            "input_schema": {
                "problem": "string",
                "current_solution": "string",
            },
            "output": {"xml_tags": ["solution"], "default_field": "solution"},
        }
    return {
        "model": "models/openai/gpt-54-mini",
        "system_prompt": "You are an expert mathematician.",
        "user_prompt": "Problem:\n\n{problem}",
        "input_schema": {"problem": "string"},
        "output": {"default_field": "text"},
    }


def _default_prompt_inputs(template: str) -> dict[str, Any]:
    if template == "cli_agent":
        return {}
    if template == "prompt_agent":
        return {}
    if template == "ideator":
        return {"problem": "$input.problem", "n": {"coalesce": ["$input.n_approaches", 2]}}
    return {"problem": "$input.problem"}


def _template_base_id(template: str) -> str:
    return {
        "prompt_agent": "prompt",
        "if_else": "if_else",
        "budget_fallback": "budget_fallback",
        "repeat": "repeat",
        "workflow_ref": "subworkflow",
        "cli_agent": "cli",
        "latex": "latex",
    }.get(template, template)


def _template_label(template: str, node_id: str) -> str:
    return {
        "prompt_agent": "Basic I/O Agent",
        "ideator": "Ideator",
        "latex": "LaTeX",
        "if_else": "If / Else",
        "budget_fallback": "Budget Fallback",
        "repeat": "Repeat",
        "workflow_ref": "Subagent Preset",
        "cli_agent": "CLI Agent",
    }.get(template, node_id)


def _clean_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"^[^A-Za-z_]+", "", cleaned)
    return cleaned or "node"


def _unique_id(base_id: str, nodes: list[Any]) -> str:
    existing = {str(node.get("id")) for node in nodes if isinstance(node, dict)}
    candidate = base_id
    counter = 2
    while candidate in existing:
        candidate = f"{base_id}_{counter}"
        counter += 1
    return candidate


def _unique_component_name(components: dict[str, Any], base: str) -> str:
    candidate = base
    counter = 2
    while candidate in components:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def _node_ui(node: dict[str, Any]) -> dict[str, Any]:
    ui = node.setdefault("ui", {})
    if not isinstance(ui, dict):
        ui = {}
        node["ui"] = ui
    return ui


def _workflow_output_ui(raw: dict[str, Any]) -> dict[str, Any]:
    dag = _editable_dag(raw)
    ui = dag.setdefault("ui", {})
    if not isinstance(ui, dict):
        ui = {}
        dag["ui"] = ui
    output_ui = ui.setdefault("workflow_output", {})
    if not isinstance(output_ui, dict):
        output_ui = {}
        ui["workflow_output"] = output_ui
    return output_ui


def _parse_csv_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_schema_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
        else:
            key, value = line, "string"
        fields[_clean_id(key)] = value.strip() or "string"
    return fields


def _parse_tools_config(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        return [dict(item) for item in parsed if isinstance(item, dict)]
    return []


def _parse_tool_refs(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else str(raw or "").split(",")
    refs = []
    seen = set()
    for value in values:
        ref = normalize_tool_name(str(value))
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def _ensure_multi_output_prompt_instruction(cfg: dict[str, Any], output: dict[str, Any]) -> None:
    tags = [str(tag).strip() for tag in (output.get("xml_tags") or []) if str(tag).strip()]
    if len(tags) <= 1:
        return
    prompt_text = f"{cfg.get('system_prompt') or ''}\n{cfg.get('user_prompt') or ''}"
    missing = [
        tag
        for tag in tags
        if f"<{tag}>" not in prompt_text or f"</{tag}>" not in prompt_text
    ]
    if not missing:
        return
    lines = [
        "",
        "",
        "Output each named result using exactly these tags:",
        *[f"<{tag}>...</{tag}>" for tag in missing],
    ]
    instruction = "\n".join(lines)
    cfg["user_prompt"] = f"{str(cfg.get('user_prompt') or '').rstrip()}{instruction}"


def _rename_node_refs(raw: Any, old_id: str, new_id: str) -> None:
    pattern = re.compile(rf"\$node\.{re.escape(old_id)}(?=\.|$)")

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(f"$node.{new_id}", value)
        if isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = replace(item)
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "id" and item == new_id:
                    continue
                value[key] = replace(item)
            return value
        return value

    replace(raw)
    for node in _editable_dag(raw)["nodes"]:
        if isinstance(node, dict) and isinstance(node.get("needs"), list):
            node["needs"] = [new_id if dep == old_id else dep for dep in node["needs"]]
        if isinstance(node, dict) and isinstance(node.get("ui"), dict):
            managed = node["ui"].get("managed_needs")
            if isinstance(managed, list):
                node["ui"]["managed_needs"] = [
                    new_id if dep == old_id else dep for dep in managed
                ]


def _rename_component_output_refs(raw: dict[str, Any], component_name: str, renames: Any) -> None:
    if not isinstance(renames, dict):
        return
    clean_renames = {
        _clean_output_field(str(old)): _clean_output_field(str(new))
        for old, new in renames.items()
    }
    clean_renames = {
        old: new
        for old, new in clean_renames.items()
        if old and new and old != new
    }
    if not clean_renames:
        return
    for node_id in _component_node_ids(raw, component_name):
        for old_field, new_field in clean_renames.items():
            _rename_node_output_ref(raw, node_id, old_field, new_field)


def _component_node_ids(raw: dict[str, Any], component_name: str) -> list[str]:
    node_ids: list[str] = []

    def visit(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "")
            if node_id and str(node.get("name") or "") == component_name:
                node_ids.append(node_id)
            body = node.get("body")
            if isinstance(body, dict):
                visit(body.get("nodes"))

    visit(_editable_dag(raw).get("nodes"))
    return node_ids


def _rename_node_output_ref(raw: Any, node_id: str, old_field: str, new_field: str) -> None:
    pattern = re.compile(
        rf"(\$(?:parent\.)?node\.{re.escape(node_id)}\.)"
        rf"{re.escape(old_field)}(?=\.|$|[^A-Za-z0-9_-])"
    )

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(rf"\g<1>{new_field}", value)
        if isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = replace(item)
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = replace(item)
            return value
        return value

    replace(raw)


def _remove_component_if_unused(raw: dict[str, Any], name: str) -> None:
    for node in _editable_dag(raw)["nodes"]:
        if _node_uses_component(node, name):
            return
    _components(raw).pop(name, None)


def _node_uses_component(node: Any, name: str) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("name") == name:
        return True
    for key in ("steps", "nodes"):
        children = node.get(key)
        if isinstance(children, list) and any(_node_uses_component(child, name) for child in children):
            return True
    body = node.get("body")
    if isinstance(body, dict):
        children = body.get("nodes")
        if isinstance(children, list) and any(_node_uses_component(child, name) for child in children):
            return True
    return False


def _dump_preset_yaml(raw: dict[str, Any]) -> str:
    return yaml.safe_dump(
        raw,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def _preset_to_info(path: Path) -> PresetInfo:
    raw_yaml = path.read_text(encoding="utf-8")
    file_version = _path_file_version(path)
    try:
        preset = load_preset(str(path))
    except PresetError as e:
        return PresetInfo(
            name=path.stem,
            label=_preset_display_label_from_yaml(raw_yaml, path.stem),
            workflow_qualname="(invalid)",
            description="",
            inputs={},
            model_overrides={},
            component_configs={},
            budget=None,
            raw_yaml=raw_yaml,
            file_version=file_version,
            error=str(e),
        )
    cls = preset.workflow_cls
    budget = None
    if preset.budget is not None:
        budget = {
            k: v
            for k, v in {
                "max_usd": preset.budget.max_usd,
                "max_tokens": preset.budget.max_tokens,
                "max_wallclock_s": preset.budget.max_wallclock_s,
                "max_tool_calls": preset.budget.max_tool_calls,
            }.items()
            if v is not None
        }
    return PresetInfo(
        name=preset.name,
        label=_preset_display_label(preset.raw, preset.name),
        workflow_qualname=cls.__qualname__,
        description=preset.description,
        inputs=dict(preset.inputs),
        model_overrides=dict(preset.model_overrides),
        component_configs=dict(preset.component_configs),
        budget=budget,
        raw_yaml=raw_yaml,
        file_version=file_version,
    )


def _preset_display_label_from_yaml(raw_yaml: str, fallback: str) -> str:
    try:
        raw = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        raw = {}
    return _preset_display_label(raw if isinstance(raw, dict) else {}, fallback)


def _preset_display_label(raw: dict[str, Any], fallback: str) -> str:
    export = raw.get("export")
    if isinstance(export, dict):
        label = str(export.get("label") or "").strip()
        if label:
            return label
    return fallback.replace("_", " ").title()


def _path_file_version(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


__all__ = [
    "AgentInfo",
    "CallDetail",
    "CallNode",
    "ExecutionGraph",
    "ExecutionGraphNode",
    "PresetInfo",
    "RenderedMessages",
    "RunEventTree",
    "RunInfo",
    "ToolDefinition",
    "create_tool_definition",
    "delete_preset",
    "discover_agent_palette_items",
    "discover_agents",
    "discover_exported_presets",
    "discover_model_options",
    "discover_presets",
    "discover_runs",
    "discover_tool_definitions",
    "find_agent",
    "find_preset",
    "find_run",
    "find_tool_definition",
    "load_call_detail",
    "load_event_tree",
    "load_monitor_summaries",
    "load_execution_graph",
    "mutate_preset_yaml",
    "normalize_preset_name",
    "normalize_tool_name",
    "presets_registry_version",
    "preset_file_version",
    "preset_dag_report",
    "render_recorded_messages",
    "safe_blob_path",
    "save_preset_yaml",
    "save_tool_definition",
    "tool_definition_to_dict",
    "validate_preset_yaml",
    "workflow_input_from_tree",
    "workflow_output_from_tree",
]
