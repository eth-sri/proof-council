"""Workflow preset registry (SPEC §4.1).

A preset is a YAML file under ``configs/workflows/`` that binds a
top-level ``Agent`` class to construction parameters, model overrides,
and a budget. Presets are a developer-ergonomics layer, not a new
type in the Agent hierarchy.

Example::

    # configs/workflows/example_workflow.yaml
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      n_approaches: 2
    dag:
      nodes:
        - id: solver
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
    budget:
      max_usd: 5.0

Loaded via ``load_preset(name)`` and launched via
``scripts/run_workflow.py`` or (POST-MVP) the Flask dev UI.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from proofstack.agent import Agent
from proofstack.budget import BudgetSpec
from proofstack.context import ModelSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_ROOT_ENV = os.environ.get("MATHAGENTS_CONFIGS_ROOT")
CONFIGS_ROOT = Path(_CONFIGS_ROOT_ENV) if _CONFIGS_ROOT_ENV else REPO_ROOT / "configs"
DEFAULT_PRESET_ROOT = CONFIGS_ROOT / "workflows"


class PresetError(Exception):
    """Malformed preset file or missing target class."""


@dataclass
class WorkflowPreset:
    """One named top-level Agent recipe."""

    name: str
    source_path: Path
    workflow_cls: type[Agent]
    inputs: dict[str, Any] = field(default_factory=dict)
    model_overrides: dict[str, ModelSpec] = field(default_factory=dict)
    component_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget: BudgetSpec | None = None
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls, name_or_path: str, *, root: Path | None = None
    ) -> "WorkflowPreset":
        root = root or DEFAULT_PRESET_ROOT
        path = _resolve_preset_path(name_or_path, root)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise PresetError(f"{path}: top-level must be a mapping")

        workflow_path = raw.get("workflow")
        if not workflow_path or not isinstance(workflow_path, str):
            raise PresetError(f"{path}: missing or non-string 'workflow' key")
        workflow_cls = _import_class(workflow_path)
        if not issubclass(workflow_cls, Agent):
            raise PresetError(
                f"{path}: 'workflow' {workflow_path} is not an Agent subclass"
            )

        budget_block = raw.get("budget") or {}
        if not isinstance(budget_block, dict):
            raise PresetError(f"{path}: 'budget' must be a mapping")
        budget = BudgetSpec(**budget_block) if budget_block else None

        model_overrides = raw.get("model_overrides") or {}
        if not isinstance(model_overrides, dict):
            raise PresetError(f"{path}: 'model_overrides' must be a mapping")

        inputs = raw.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise PresetError(f"{path}: 'inputs' must be a mapping")

        component_configs = raw.get("components") or {}
        if not isinstance(component_configs, dict):
            raise PresetError(f"{path}: 'components' must be a mapping")
        for name, cfg in component_configs.items():
            if not isinstance(name, str) or not isinstance(cfg, dict):
                raise PresetError(
                    f"{path}: each component config must map a string name to a mapping"
                )
        if "dag" in raw:
            dag = raw["dag"]
            if not isinstance(dag, dict):
                raise PresetError(f"{path}: 'dag' must be a mapping")
            workflow_cfg = dict(component_configs.get(workflow_cls.__name__, {}))
            workflow_cfg.setdefault("dag", dag)
            component_configs = dict(component_configs)
            component_configs[workflow_cls.__name__] = workflow_cfg

        return cls(
            name=path.stem,
            source_path=path,
            workflow_cls=workflow_cls,
            inputs=inputs,
            model_overrides=model_overrides,
            component_configs=component_configs,
            budget=budget,
            description=str(raw.get("description", "") or ""),
            raw=raw,
        )

    def build_inputs(
        self,
        *,
        problem: str | None = None,
        problem_id: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge preset defaults + launched problem data + CLI overrides,
        filtered to the workflow's ``Inputs`` field set. ``problem`` and
        ``problem_id`` are only injected when the target workflow declares
        them; explicit CLI overrides still win.
        """
        fields = self.workflow_cls.Inputs.model_fields  # type: ignore[attr-defined]
        extra_allowed = getattr(self.workflow_cls.Inputs, "model_config", {}).get("extra") == "allow"  # type: ignore[attr-defined]
        merged: dict[str, Any] = dict(self.inputs)
        if ("problem" in fields or extra_allowed) and problem is not None:
            merged["problem"] = problem
        if ("problem_id" in fields or extra_allowed) and problem_id is not None:
            merged["problem_id"] = problem_id
        if cli_overrides:
            merged.update(cli_overrides)
        if extra_allowed:
            return merged
        return {k: v for k, v in merged.items() if k in fields}


def list_presets(root: Path | None = None) -> list[Path]:
    """Return preset YAML paths under ``root`` (recursively). Sorted by name.

    Searching subdirectories lets a deployment keep curated examples at the top
    level and park personal/experimental presets in an (optionally gitignored)
    subdir like ``local/`` without losing dashboard/CLI access to them.
    """
    root = root or DEFAULT_PRESET_ROOT
    if not root.exists():
        return []
    return sorted(root.rglob("*.yaml"), key=lambda p: p.stem)


def load_preset(name_or_path: str, *, root: Path | None = None) -> WorkflowPreset:
    return WorkflowPreset.load(name_or_path, root=root)


# --- helpers -----------------------------------------------------------------


def _resolve_preset_path(name_or_path: str, root: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix == ".yaml" and candidate.exists():
        return candidate.resolve()
    short = root / f"{name_or_path}.yaml"
    if short.exists():
        return short.resolve()
    # Fall back to a recursive search so presets parked in a subdir (e.g.
    # ``local/``) still resolve by their bare name.
    nested = sorted(root.rglob(f"{name_or_path}.yaml"), key=lambda p: len(p.parts))
    if nested:
        return nested[0].resolve()
    raise PresetError(
        f"preset not found: looked for {candidate!s} and {short!s}"
    )


def _import_class(dotted: str) -> type:
    if "." not in dotted:
        raise PresetError(f"workflow path must be dotted: {dotted!r}")
    module_path, _, cls_name = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise PresetError(f"cannot import {module_path}: {e}") from e
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise PresetError(f"{module_path} has no attribute {cls_name}")
    return cls


__all__ = [
    "DEFAULT_PRESET_ROOT",
    "PresetError",
    "WorkflowPreset",
    "list_presets",
    "load_preset",
]
