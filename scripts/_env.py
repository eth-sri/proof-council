"""Bootstrap helper: load `.env` BEFORE importing project modules.

This module deliberately depends only on the standard library. Scripts
that read env vars at import time (e.g. ``mathagents.config_loader``
captures ``MATHAGENTS_CONFIGS_ROOT`` when imported) need the dotenv
file applied first; importing ``mathagents.utils.load_dotenv_file``
would itself trigger ``mathagents.config_loader`` and lock in the
pre-dotenv values.

Keep this module dependency-free.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: Path | str) -> None:
    """Read KEY=VALUE pairs from a .env file into os.environ.

    Existing env vars win — sourcing a real shell takes precedence over
    the file. Silently no-ops if the file is missing or unreadable.
    """
    path = Path(path)
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if not key or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


__all__ = ["load_dotenv_file"]
