"""Read-only, bounded launch snapshots of the lead's saved canonical files."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import time

from .delegation_context import MAX_FILE_BYTES


CANONICAL_INPUTS = {"answer.tex": "answer_tex", "research_notes.tex": "research_notes_tex",
                    "references.bib": "references_bib"}
SNAPSHOT_TIMEOUT_S = 60.0
MAX_SCAN_ENTRIES = 256


def _capture(lead, container, deadline):
    files, warnings = {}, []

    def client():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("launch snapshot deadline")
        return lead._openai(timeout=min(10.0, remaining))

    try:
        candidates, duplicates = {}, set()
        for index, cf in enumerate(client().containers.files.list(container, limit=100)):
            if index >= MAX_SCAN_ENTRIES or time.monotonic() >= deadline:
                raise TimeoutError("launch snapshot scan limit")
            path = str(getattr(cf, "path", ""))
            if not path.startswith("/mnt/data/"):
                continue
            name = path.removeprefix("/mnt/data/")
            if name not in CANONICAL_INPUTS:
                continue
            if name in candidates:
                duplicates.add(name)
            candidates[name] = cf
        for name in CANONICAL_INPUTS:
            try:
                if name not in candidates or name in duplicates:
                    raise ValueError("missing or ambiguous canonical path")
                cf = candidates[name]
                size = getattr(cf, "bytes", None)
                if size is not None and (type(size) is not int or not 0 <= size <= MAX_FILE_BYTES):
                    raise ValueError("snapshot file size")
                chunks, count = [], 0
                with client().containers.files.content.with_streaming_response.retrieve(
                    container_id=container, file_id=cf.id,
                ) as stream:
                    for chunk in stream.iter_bytes(chunk_size=65536):
                        count += len(chunk)
                        if count > MAX_FILE_BYTES or time.monotonic() >= deadline:
                            raise ValueError("snapshot transfer limit")
                        chunks.append(chunk)
                if size is not None and size != count:
                    raise ValueError("incomplete snapshot")
                files[name] = b"".join(chunks).decode("utf-8", errors="strict")
            except Exception as exc:
                warnings.append(f"{name}: current saved file unavailable ({type(exc).__name__}); using turn-start input.")
    except Exception as exc:
        warnings.append(f"Current sandbox listing unavailable ({type(exc).__name__}); using turn-start inputs for unread files.")
    return files, warnings


def describe_snapshot(inp, captured=None, *, warnings=(), container=None):
    captured = captured or {}
    files = {name: captured.get(name, getattr(inp, field)) for name, field in CANONICAL_INPUTS.items()}
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "container_id": container,
        "files": {name: {"source": "launch_sandbox" if name in captured else "turn_start",
                         "sha256": hashlib.sha256(body.encode()).hexdigest()}
                  for name, body in files.items()},
        "warnings": list(warnings),
    }
    metadata["sha256"] = hashlib.sha256(json.dumps(metadata["files"], sort_keys=True).encode()).hexdigest()
    return {"files": files, "metadata": metadata}


async def launch_snapshot(lead, container, call_deadline=None):
    inp = lead._current_inp
    if not container:
        return describe_snapshot(inp, warnings=["No current hosted sandbox was identified; files are turn-start inputs."])
    deadline = min(time.monotonic() + SNAPSHOT_TIMEOUT_S,
                   call_deadline if call_deadline is not None else float("inf"))
    if deadline <= time.monotonic():
        return describe_snapshot(inp, warnings=["Snapshot deadline reached; files are turn-start inputs."], container=container)
    worker = asyncio.create_task(asyncio.to_thread(_capture, lead, container, deadline))
    worker.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        files, warnings = await asyncio.wait_for(asyncio.shield(worker), max(0.0, deadline - time.monotonic()))
    except TimeoutError:
        files, warnings = {}, ["Snapshot transfer timed out; files are turn-start inputs."]
    return describe_snapshot(inp, files, warnings=warnings, container=container)
