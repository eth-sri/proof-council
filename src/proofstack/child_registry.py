"""Durable registry of CLI child process groups per run.

Sandbox subprocesses start their own POSIX session, so killing the
workflow worker's process group does NOT reach them: a dashboard Stop
could leave codex/claude children running (and spending) with no owner.
Every CLI spawn is recorded in ``<run>/run-children.json``; the
dashboard's Stop reads the file and hard-kills whatever the dead worker
left behind. Entries are removed again on normal child termination, so
the file is empty (absent) whenever nothing is at risk.

Docker-backend children are only partially covered: the registered pid
is the ``docker run`` client, whose death does not always stop the
container itself.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_FILENAME = "run-children.json"


def _path(run_dir: Path) -> Path:
    return Path(run_dir) / _FILENAME


def _load(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _store(path: Path, entries: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        if entries:
            tmp.write_text(json.dumps(entries), encoding="utf-8")
            os.replace(tmp, path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        tmp.unlink(missing_ok=True)


def register_child(run_dir: Path, *, pid: int, cmd0: str = "", label: str = "") -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = pid
    with _LOCK:
        path = _path(run_dir)
        entries = [e for e in _load(path) if e.get("pid") != pid]
        entries.append(
            {
                "pid": int(pid),
                "pgid": int(pgid),
                "cmd0": os.path.basename(str(cmd0)),
                "label": str(label),
            }
        )
        _store(path, entries)


def unregister_child(run_dir: Path, pid: int) -> None:
    with _LOCK:
        path = _path(run_dir)
        _store(path, [e for e in _load(path) if e.get("pid") != pid])


def _entry_still_matches(entry: dict) -> bool:
    """Guard against pid reuse: the pid must still exist, belong to the
    recorded process group, and (when /proc is available) run the recorded
    executable. Anything else means the child already exited."""
    pid = int(entry.get("pid") or 0)
    pgid = int(entry.get("pgid") or 0)
    if pid <= 1 or pgid <= 1:
        return False
    try:
        if os.getpgid(pid) != pgid:
            return False
    except OSError:
        return False
    cmd0 = str(entry.get("cmd0") or "")
    if cmd0:
        try:
            raw = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .decode("utf-8", "replace")
            )
        except OSError:
            return True  # no /proc: pgid match is the best we have
        if cmd0 not in raw:
            return False
    return True


def kill_registered_children(run_dir: Path, *, grace_s: float = 3.0) -> dict:
    """SIGTERM every registered child group, escalate to SIGKILL after
    ``grace_s``, and clear the registry. Returns what was signalled."""
    with _LOCK:
        entries = [e for e in _load(_path(run_dir)) if _entry_still_matches(e)]
    signalled: list[int] = []
    for entry in entries:
        try:
            os.killpg(int(entry["pgid"]), signal.SIGTERM)
            signalled.append(int(entry["pid"]))
        except OSError:
            continue
    if signalled:
        deadline = time.time() + max(0.0, grace_s)
        remaining = list(signalled)
        while remaining and time.time() < deadline:
            time.sleep(0.1)
            remaining = [
                pid
                for pid in remaining
                if _pid_alive(pid)
            ]
        for pid in remaining:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass
    with _LOCK:
        _store(_path(run_dir), [])
    return {"children_signalled": signalled}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = [
    "register_child",
    "unregister_child",
    "kill_registered_children",
]
