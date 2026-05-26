"""Reusable model tool definitions loaded from ``configs/tools``."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import yaml

from mathagents.config_loader import CONFIGS_ROOT

ToolPair = tuple[Callable[..., Any] | None, dict[str, Any]]

DEFAULT_TOOL_ROOT = CONFIGS_ROOT / "tools"


def load_tool_pair(name: str, root: Path | None = None) -> ToolPair:
    root = root or DEFAULT_TOOL_ROOT
    safe = _safe_name(name)
    yaml_path = root / f"{safe}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"tool definition not found: {safe}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{safe}: tool YAML must be a mapping")
    descriptor = raw.get("tool") or raw.get("descriptor") or {}
    if not isinstance(descriptor, dict):
        raise TypeError(f"{safe}: tool descriptor must be a mapping")
    if descriptor.get("type") != "function":
        return None, dict(descriptor)
    fn_name = str(raw.get("python_function") or "")
    function_desc = descriptor.get("function")
    if not fn_name and isinstance(function_desc, dict):
        fn_name = str(function_desc.get("name") or "")
    if not fn_name:
        raise TypeError(f"{safe}: function tools need python_function")
    return _load_function(root / f"{safe}.py", fn_name), dict(descriptor)


def resolve_tool_pairs(refs: list[str] | tuple[str, ...] | None) -> list[ToolPair]:
    pairs: list[ToolPair] = []
    for ref in refs or []:
        pairs.append(load_tool_pair(str(ref)))
    return pairs


def _load_function(path: Path, name: str) -> Callable[..., Any]:
    if not path.exists():
        raise FileNotFoundError(f"tool Python file not found: {path.name}")
    module_name = f"_proofstack_tool_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import tool Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, name)
    if not callable(fn):
        raise TypeError(f"{path.name}: {name} is not callable")
    return fn


def _safe_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    cleaned = re.sub(r"^[^A-Za-z_]+", "", cleaned)
    if not cleaned:
        raise ValueError("tool name cannot be empty")
    return cleaned


__all__ = ["DEFAULT_TOOL_ROOT", "ToolPair", "load_tool_pair", "resolve_tool_pairs"]
