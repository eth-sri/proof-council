from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# When the package is installed (e.g. inside the First Proof Docker image)
# the source-tree ``configs/`` directory is not next to the installed
# module. Honour ``MATHAGENTS_CONFIGS_ROOT`` so deployments can point at
# the actual configs location explicitly. Falls back to the dev-tree path.
_CONFIGS_ROOT_ENV = os.environ.get("MATHAGENTS_CONFIGS_ROOT")
CONFIGS_ROOT = Path(_CONFIGS_ROOT_ENV) if _CONFIGS_ROOT_ENV else PACKAGE_ROOT / "configs"


def _safe_deepcopy(value: Any) -> Any:
    if isinstance(value, dict):
        return {_safe_deepcopy(key): _safe_deepcopy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_deepcopy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_deepcopy(item) for item in value)
    if isinstance(value, set):
        return {_safe_deepcopy(item) for item in value}
    try:
        return deepcopy(value)
    except TypeError:
        return value


def _ensure_yaml_path(path_or_ref: str | Path) -> Path:
    path = Path(path_or_ref)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    if path.suffix:
        return (CONFIGS_ROOT / path).resolve()
    return (CONFIGS_ROOT / f"{path.as_posix()}.yaml").resolve()


def load_yaml_config(path_or_ref: str | Path) -> dict[str, Any]:
    path = _ensure_yaml_path(path_or_ref)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config in {path}, got {type(data).__name__}.")
    loaded = _safe_deepcopy(data)
    loaded["__config_path__"] = str(path)
    loaded["__config_dir__"] = str(path.parent)
    return loaded


def _copy_mapping(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config, got {type(config).__name__}.")
    return _safe_deepcopy(config)


def _merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = _safe_deepcopy(base)
    for key, value in overrides.items():
        if key.startswith("__"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = _safe_deepcopy(value)
    return merged


def load_solver_config(path_or_ref: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(path_or_ref, dict):
        config = _copy_mapping(path_or_ref)
    else:
        config = load_yaml_config(path_or_ref)
    if "base" in config:
        base_config = load_solver_config(config["base"])
        overrides = {k: v for k, v in config.items() if k != "base"}
        config = _merge_config(base_config, overrides)
    return config
