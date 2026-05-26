"""Event log: append-only JSONL with size-thresholded $ref payloads.

Every agent invocation, model call, tool call, etc. is recorded as a
single JSON line. Strings or JSON blobs in `payload` larger than
``MAX_INLINE_BYTES`` are spilled to a sibling file and replaced with
``{"$ref": "<relative/path>"}`` so the JSONL stays streamable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Strings/blobs above this size are spilled to a $ref file.
MAX_INLINE_BYTES = 2_048

# Context var holds the parent call_id while inside an agent's __call__,
# so child events automatically pick up parent_call_id without each
# emitter having to thread it explicitly.
_PARENT_CALL_ID: ContextVar[str | None] = ContextVar(
    "_proofstack_parent_call_id", default=None
)
_AGENT_PATH: ContextVar[tuple[str, ...]] = ContextVar(
    "_proofstack_agent_path", default=()
)
_CURRENT_WORKDIR: ContextVar["object | None"] = ContextVar(
    "_proofstack_current_workdir", default=None
)


def new_call_id() -> str:
    """Short opaque id, sortable enough for debugging."""
    return secrets.token_hex(3)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


def _safe_filename(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in part)[:120]


@dataclass
class JSONLSink:
    """Append-only JSONL writer with $ref spill-out for large payloads.

    The sink is responsible for both ``events.jsonl`` and the per-event
    spill directory (``events_blobs/``). A single asyncio lock guards
    the file handle; events from concurrent agents are serialized.
    """

    events_path: Path
    blobs_dir: Path
    max_inline_bytes: int = MAX_INLINE_BYTES
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def at(cls, root: Path) -> "JSONLSink":
        events_path = root / "events.jsonl"
        blobs_dir = root / "events_blobs"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        blobs_dir.mkdir(parents=True, exist_ok=True)
        return cls(events_path=events_path, blobs_dir=blobs_dir)

    async def write(self, record: dict[str, Any]) -> None:
        record = self._spill_payload(record)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _spill_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return record
        record = {**record, "payload": self._spill_value(payload, prefix=record)}
        return record

    def _spill_value(self, value: Any, *, prefix: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {k: self._spill_value(v, prefix=prefix) for k, v in value.items()}
        if isinstance(value, list):
            return [self._spill_value(v, prefix=prefix) for v in value]
        if isinstance(value, str) and len(value.encode("utf-8")) > self.max_inline_bytes:
            ref = self._spill_string(value, prefix=prefix)
            return {"$ref": ref}
        return value

    def _spill_string(self, value: str, *, prefix: dict[str, Any]) -> str:
        agent = _safe_filename(str(prefix.get("agent", "anon")))
        kind = _safe_filename(str(prefix.get("kind", "evt")))
        call_id = _safe_filename(str(prefix.get("call_id", "")))
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        name = f"{agent}-{kind}-{call_id}-{digest}.txt"
        path = self.blobs_dir / name
        path.write_text(value, encoding="utf-8")
        return str(path.relative_to(self.events_path.parent))


@dataclass
class EventEmitter:
    """Emits structured events to a sink, with automatic agent_path / call_id wiring."""

    sink: JSONLSink
    run_id: str
    agent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def scoped(self, *, agent: str, **extra: Any) -> "EventEmitter":
        merged = {**self.extra, **extra}
        return EventEmitter(sink=self.sink, run_id=self.run_id, agent=agent, extra=merged)

    _UNSET = object()

    async def emit(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
        execution_mode: str | None = None,
        parent_call_id: "str | None | object" = _UNSET,
    ) -> None:
        """Emit a structured event.

        ``parent_call_id`` defaults to whatever the ambient
        ``_PARENT_CALL_ID`` contextvar holds. Pass it explicitly when
        emitting a boundary event from inside an agent — for
        ``agent.start``/``agent.end``/``agent.error`` the parent must be
        the *caller*'s call id, not the agent's own.
        """
        agent_path = list(_AGENT_PATH.get())
        if self.agent and (not agent_path or agent_path[-1] != self.agent):
            agent_path = agent_path + [self.agent]
        parent = (
            parent_call_id
            if parent_call_id is not EventEmitter._UNSET
            else _PARENT_CALL_ID.get()
        )
        record: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "run_id": self.run_id,
            "agent": self.agent,
            "agent_path": ".".join(agent_path) if agent_path else None,
            "call_id": call_id,
            "parent_call_id": parent,
            "execution_mode": execution_mode,
            "kind": kind,
            "payload": payload or {},
        }
        if self.extra:
            record["extra"] = dict(self.extra)
        await self.sink.write(record)


__all__ = [
    "EventEmitter",
    "JSONLSink",
    "MAX_INLINE_BYTES",
    "new_call_id",
    "_PARENT_CALL_ID",
    "_AGENT_PATH",
    "_CURRENT_WORKDIR",
]
