"""Load evaluator prompts and model settings without starting a run."""

import argparse
import math
from pathlib import Path

import yaml


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def load_prompts(path: Path):
    prompts = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("marking_scheme_prompt", "grading_prompt"):
        if not isinstance(prompts.get(key), str) or not prompts[key].strip():
            raise ValueError(f"Missing evaluator prompt: {key}")
    return prompts


def load_solver_timeout(repository_root: Path):
    path = repository_root / "configs/evaluators/benchmark.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = config.get("solver_timeout_seconds") if isinstance(config, dict) else None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: solver_timeout_seconds must be a finite positive number")
    return value


def resolve_model(reference):
    from mathagents import load_solver_config

    config = load_solver_config(reference)
    return {key: value for key, value in config.items() if not key.startswith("__")}
