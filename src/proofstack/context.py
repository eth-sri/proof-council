"""Per-run plumbing: workdir, events, budgets, artifacts, model factory."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING

from proofstack.budget import BudgetRegistry, BudgetSpec, BudgetTracker
from proofstack.events import EventEmitter, JSONLSink
from proofstack.state import ArtifactRegistry

if TYPE_CHECKING:
    from proofstack.agent import Agent

# A model spec is either a path-like config reference understood by
# mathagents.config_loader.load_solver_config (e.g.
# "models/openai/gpt-54") or an already-loaded config dict.
ModelSpec = str | Path | dict


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_api_client_factory(model_spec: ModelSpec):
    """Build a mathagents.APIClient from a config reference."""
    # Imported lazily to avoid pulling heavy provider SDKs at module load.
    from mathagents import APIClient, load_solver_config

    cfg = load_solver_config(model_spec)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
    return APIClient(**cfg)


class ResumeCache:
    """Hash-keyed JSON cache of prior agent Outputs.

    Always writes to the current run's cache directory. Reads also fall
    back to a previous run's directory when ``resume_from`` is set,
    enabling deterministic resume on rerun.
    """

    def __init__(self, write_dir: Path, read_dirs: list[Path] | None = None) -> None:
        self.write_dir = write_dir
        self.read_dirs = read_dirs or []
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path_in(self, root: Path, key: str) -> Path:
        return root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        for root in [self.write_dir, *self.read_dirs]:
            path = self._path_in(root, key)
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return None
        return None

    def put(self, key: str, value: Any) -> None:
        path = self._path_in(self.write_dir, key)
        path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")


@dataclass
class RunContext:
    """Per-run plumbing object passed implicitly to every sub-agent.

    Only ``run_id`` and ``root_workdir`` are required to construct one;
    use ``RunContext.create(...)`` for the standard wiring (events sink
    in workdir, budget registry with a "run" root, artifact registry).
    """

    run_id: str
    root_workdir: Path
    events: EventEmitter
    budgets: BudgetRegistry
    artifacts: ArtifactRegistry
    api_client_factory: Callable[[ModelSpec], Any]
    resume_cache: ResumeCache
    mode: Literal["benchmark", "research"] = "benchmark"
    resume_from: str | None = None
    model_overrides: dict[str, ModelSpec] = field(default_factory=dict)
    component_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    _agent_call_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None = None,
        root_workdir: Path | str | None = None,
        run_budget: BudgetSpec | None = None,
        api_client_factory: Callable[[ModelSpec], Any] | None = None,
        mode: Literal["benchmark", "research"] = "benchmark",
        resume_from: str | None = None,
        model_overrides: dict[str, ModelSpec] | None = None,
        component_configs: dict[str, dict[str, Any]] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        flat: bool = False,
    ) -> "RunContext":
        """Build a RunContext + workdir tree.

        ``flat=False`` (default): ``root_workdir`` is the *outputs root*
        (``outputs/`` by convention) and the actual run lives at
        ``<root_workdir>/<run_id>/``.

        ``flat=True``: ``root_workdir`` IS the run directory; nothing is
        nested under ``run_id``. Used by the First Proof entrypoint so
        ``/data/output/events.jsonl`` is the canonical event log.
        """
        run_id = run_id or default_run_id()
        if flat:
            if root_workdir is None:
                raise ValueError("flat=True requires an explicit root_workdir")
            run_dir = Path(root_workdir)
            outputs_root = run_dir.parent
        else:
            outputs_root = Path(root_workdir) if root_workdir else Path("outputs")
            run_dir = outputs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        sink = JSONLSink.at(run_dir)
        events = EventEmitter(sink=sink, run_id=run_id, agent=None)

        budgets = BudgetRegistry()
        budgets.register_root("run", run_budget)

        artifacts = ArtifactRegistry()

        resume_dirs: list[Path] = []
        if resume_from:
            prior = outputs_root / resume_from / "resume_cache"
            if prior.exists():
                resume_dirs.append(prior)
        resume_cache = ResumeCache(
            write_dir=run_dir / "resume_cache",
            read_dirs=resume_dirs,
        )

        return cls(
            run_id=run_id,
            root_workdir=run_dir,
            events=events,
            budgets=budgets,
            artifacts=artifacts,
            api_client_factory=api_client_factory or _default_api_client_factory,
            resume_cache=resume_cache,
            mode=mode,
            resume_from=resume_from,
            model_overrides=model_overrides or {},
            component_configs=component_configs or {},
            config_snapshot=config_snapshot or {},
        )

    # --- agent-facing helpers --------------------------------------------------

    def workdir_for(self, agent: "Agent", call_id: str | None = None) -> Path:
        """Allocate a per-invocation workdir for an agent.

        The directory is named ``<agent-name>-<seq>`` (always nested per
        call, never reused). ``call_id`` is appended when provided so
        debug traces can correlate a workdir to the matching event log
        entry. Created lazily.
        """
        seq = self._agent_call_counts.get(agent.name, 0)
        self._agent_call_counts[agent.name] = seq + 1
        suffix = f"-c{seq}" + (f"-{call_id}" if call_id else "")
        path = self.root_workdir / "agents" / f"{agent.name}{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_for(self, agent: "Agent", default: ModelSpec) -> ModelSpec:
        """Look up an override by instance/class key, then ``"*"``,
        then the agent's configured default."""
        for key in reversed(_agent_lookup_keys(agent)):
            if key in self.model_overrides:
                return self.model_overrides[key]
        if "*" in self.model_overrides:
            return self.model_overrides["*"]
        return default

    def component_config_for(self, agent: "Agent") -> dict[str, Any]:
        """Return merged per-agent config for ``agent``.

        Keys are checked from broad to narrow, so a workflow preset can set
        ``"*"``, then class-level defaults such as ``"Solver"``, then a
        specific instance name such as ``"branch_solver"``.
        """
        merged: dict[str, Any] = {}
        for key in ("*", *_agent_lookup_keys(agent)):
            cfg = self.component_configs.get(key)
            if isinstance(cfg, dict):
                merged = _merge_dicts(merged, cfg)
        return merged

    def write_metadata(self, extra: dict[str, Any] | None = None) -> Path:
        meta = {
            "run_id": self.run_id,
            "mode": self.mode,
            "resume_from": self.resume_from,
            "config_snapshot": self.config_snapshot,
            **(extra or {}),
        }
        path = self.root_workdir / "run-metadata.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path


__all__ = ["ModelSpec", "ResumeCache", "RunContext", "default_run_id"]


def _agent_lookup_keys(agent: "Agent") -> tuple[str, ...]:
    cls = type(agent)
    return (
        f"{cls.__module__}.{cls.__qualname__}",
        cls.__qualname__,
        cls.__name__,
        agent.name,
    )


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
