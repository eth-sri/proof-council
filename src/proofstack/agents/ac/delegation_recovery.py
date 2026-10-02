"""Local, bounded research recovery; never replay provider tool conversations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from proofstack.atomic import write_text_atomic
from .delegation_context import MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES


class CheckpointInputMismatch(ValueError):
    """A checkpoint belongs to a different Author invocation of this round."""


def problem_key(problem):
    return hashlib.sha256(problem.strip().encode()).hexdigest()


def input_key(inp):
    values = {k: getattr(inp, k) for k in (
        "problem", "round", "answer_tex", "research_notes_tex", "references_bib", "prev_critique",
        "prev_council", "prev_compute_response",
    )}
    values["problem"] = (inp.recovery_problem or inp.problem).strip()
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _local(root, relative):
    if not isinstance(relative, str) or not relative or len(relative) > 1024:
        raise ValueError("invalid recovery path")
    p = PurePosixPath(relative)
    if p.is_absolute() or str(p) != relative or ".." in p.parts or "\\" in relative:
        raise ValueError("unsafe recovery path")
    path = root
    for component in p.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("symlink in recovery path")
    return path


def _read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ValueError("recovery artifact is not a regular file")
        body = f.read(limit + 1)
    if len(body) > limit:
        raise ValueError("recovery file exceeds limit")
    return body


def read_checkpoint(root, path, problem, round=None, identity=None):
    path = _local(root, str(path.relative_to(root)))
    row = json.loads(_read(path, MAX_FILE_BYTES))
    if (not isinstance(row, dict) or row.get("version") != 1 or row.get("problem_hash") != problem_key(problem)
            or (round is not None and row.get("round") != round)):
        raise ValueError("recovery checkpoint identity mismatch")
    if not isinstance(row.get("input_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["input_hash"]):
        raise ValueError("invalid recovery input hash")
    if identity is not None and row.get("input_hash") != identity:
        raise CheckpointInputMismatch("recovery checkpoint inputs changed")
    if any(type(row.get(k)) is not int or not 0 <= row[k] <= 10000 for k in ("round", "waves_done", "agent_counter")):
        raise ValueError("invalid recovery counters")
    seats, log = row.get("seats"), row.get("log")
    if not isinstance(seats, dict) or not isinstance(log, list) or len(log) > 10000:
        raise ValueError("invalid recovery task records")
    for name, seat in seats.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z]+[0-9]+", name)
                or not isinstance(seat, dict) or not isinstance(seat.get("role"), str)
                or not isinstance(seat.get("artifacts"), list)):
            raise ValueError("invalid recovered seat")
    for rec in log:
        if (not isinstance(rec, dict) or not isinstance(rec.get("agent_id"), str)
                or not re.fullmatch(r"[a-z]+[0-9]+", rec["agent_id"])
                or type(rec.get("wave")) is not int or not 1 <= rec["wave"] <= row["waves_done"]
                or not isinstance(rec.get("role"), str) or not isinstance(rec.get("task"), str)
                or not isinstance(rec.get("artifacts"), list)):
            raise ValueError("invalid recovered task")
    context = _local(root, row["context"])
    manifest = json.loads(_read(_local(context, "manifest.json"), MAX_FILE_BYTES))
    if (not isinstance(manifest, dict) or manifest.get("round") != row["round"]
            or not isinstance(manifest.get("files"), list) or len(manifest["files"]) > MAX_FILES):
        raise ValueError("invalid recovery context")
    files, total = [], 0
    names = set()
    for meta in manifest["files"]:
        if not isinstance(meta, dict) or not isinstance(meta.get("source"), str):
            raise ValueError("invalid recovered artifact metadata")
        name = meta["path"]
        artifact_path = _local(context, name)
        if name in names:
            raise ValueError("duplicate recovery artifact")
        names.add(name)
        body = _read(artifact_path, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        if len(body) != meta["bytes"] or hashlib.sha256(body).hexdigest() != meta["sha256"]:
            raise ValueError("recovery artifact digest mismatch")
        total += len(body)
        files.append((meta, body.decode("utf-8")))
    for entry in [*seats.values(), *log]:
        if any(not isinstance(p, str) or p not in names for p in entry["artifacts"]):
            raise ValueError("missing recovered artifact reference")
    return row, files


def has_research_checkpoint(root, problem, round=0):
    directory = root / "helper-recovery" / problem_key(problem)
    if directory.is_symlink():
        return False
    for path in sorted(directory.glob("round-*.json"), reverse=True):
        try:
            row, files = read_checkpoint(root, path, problem, round=round)
            if row["waves_done"] and any(meta["path"].startswith("helpers/") for meta, _ in files):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def save_checkpoint(lead):
    context = lead._context
    with context._lock:
        row = {
            "version": 1, "problem_hash": problem_key(lead._current_inp.recovery_problem or lead._current_inp.problem),
            "input_hash": input_key(lead._current_inp), "round": lead._current_inp.round,
            "context": str(context.root.relative_to(lead.ctx.root_workdir)),
            "waves_done": lead._waves_done, "agent_counter": lead._agent_counter,
            "seats": {name: {k: v for k, v in seat.items() if k != "messages_after"}
                      for name, seat in lead._seats.items()},
            "log": [{k: v for k, v in rec.items() if k != "report"} for rec in lead._delegation_log],
        }
        path = lead.ctx.root_workdir / "helper-recovery" / row["problem_hash"] / f"round-{row['round']}.json"
        write_text_atomic(path, json.dumps(row, indent=2))


def restore_checkpoint(lead):
    inp = lead._current_inp
    root = lead.ctx.root_workdir
    problem = inp.recovery_problem or inp.problem
    path = _local(root, f"helper-recovery/{problem_key(problem)}/round-{inp.round}.json")
    if not path.exists():
        return False
    try:
        row, files = read_checkpoint(root, path, problem, inp.round, input_key(inp))
    except CheckpointInputMismatch:
        # The workspace may have advanced before resume-state was committed.
        # Never reuse research or consumed allowances from different inputs.
        return False
    for meta, body in files:
        if meta["path"].startswith("round/"):
            continue  # The current prompt includes any new continuation instructions.
        lead._context.put(meta["path"], body, source=meta["source"], provenance=meta.get("provenance"))
    lead._waves_done = row["waves_done"]
    lead._agent_counter = row["agent_counter"]
    lead._seats = row["seats"]
    for seat in lead._seats.values():
        seat["messages_after"] = []  # Old remote attachments and sandboxes may no longer exist.
    lead._delegation_log = row["log"]
    for rec in lead._delegation_log:
        report = f"helpers/wave{rec['wave']}-{rec['agent_id']}/final-response.md"
        rec["report"] = lead._context.files.get(report, "")
        prefix = f"helpers/wave{rec['wave']}-{rec['agent_id']}/"
        rec["artifacts"] = sorted(set(rec.get("artifacts", [])) | {p for p in lead._context.files if p.startswith(prefix)})
        seat = lead._seats.setdefault(rec["agent_id"], {
            "role": rec["role"], "messages_after": [], "artifacts": [], "report_path": None,
            "shared_context": bool(rec.get("include_workspace")),
        })
        seat["artifacts"] = sorted(set(seat["artifacts"]) | set(rec["artifacts"]))
        if not rec.get("finished"):
            rec["error"] = "interrupted before helper completion; only published artifacts recovered"
    return True
