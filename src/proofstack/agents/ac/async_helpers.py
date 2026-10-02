"""Research-scoped helper jobs with immutable, recoverable handoffs."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import math
import time
from uuid import uuid4

from proofstack.atomic import write_text_atomic
from proofstack.budget import BudgetExhausted
from .delegation_context import DelegationContext, MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES
from .delegation_recovery import _local, _read, input_key, problem_key
from .delegation_snapshot import CANONICAL_INPUTS, describe_snapshot


_SESSIONS: ContextVar[dict | None] = ContextVar("research_helper_sessions", default=None)
_TERMINAL = {"completed", "incomplete", "failed", "cancelled", "interrupted"}
_LOG = logging.getLogger(__name__)
INPUT_MAX_BYTES = MAX_TOTAL_BYTES * 3 // 4
INPUT_MAX_FILES = MAX_FILES - 64
SHUTDOWN_GRACE_S = 60.0
MAX_OMISSION_DETAILS = 128
MAX_OMISSION_CHARS = 1024


def _omission_details(messages, count):
    details = [m if len(m) <= MAX_OMISSION_CHARS else m[:MAX_OMISSION_CHARS] + " [truncated]"
               for m in messages[:MAX_OMISSION_DETAILS]]
    if count > len(details):
        details.append(f"{count - len(details)} further omissions not listed; see context_omission_count for the total.")
    return details


async def _join_tasks(tasks, *, deadline=None):
    """Drain accounting without forwarding repeated cancellation into children."""
    joined = asyncio.gather(*tasks, return_exceptions=True)
    cancelled = None
    while not joined.done():
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if remaining == 0:
            return None, cancelled
        try:
            await asyncio.wait([joined], timeout=remaining)
        except asyncio.CancelledError as exc:
            cancelled = exc
    return joined.result(), cancelled


@asynccontextmanager
async def helper_scope():
    """Join helpers before research accounting/final cleanup, including on error."""
    sessions = {}
    token = _SESSIONS.set(sessions)
    try:
        yield
    finally:
        try:
            results, cancelled = await _join_tasks([s.close() for s in sessions.values()])
            for result in results:
                if isinstance(result, Exception):
                    _LOG.warning("Auxiliary helper cleanup failed: %s", result)
                elif isinstance(result, BaseException):
                    raise result
            if cancelled is not None:
                raise cancelled
        finally:
            _SESSIONS.reset(token)


def session_for(lead):
    problem = lead._current_inp.recovery_problem or lead._current_inp.problem
    key = (str(lead.ctx.root_workdir), problem_key(problem))
    sessions = _SESSIONS.get()
    if sessions is not None:
        if key not in sessions:
            sessions[key] = HelperSession(lead.ctx, key[1])
        return sessions[key]
    previous = getattr(lead, "_async_session", None)
    if previous is not None and previous.key == key[1] and not previous.closed:
        return previous
    return HelperSession(lead.ctx, key[1])


def workflow_owns_helpers():
    return _SESSIONS.get() is not None


class HelperSession:
    def __init__(self, ctx, key):
        self.ctx = ctx
        self.key = key
        self.root = ctx.root_workdir / "async-helpers" / key
        self.records = {}
        self.stores = {}
        self.tasks = {}
        self._jobs = {}
        self._detached = set()
        self._close_task = None
        self._close_deadline = None
        self._shutdown_sealed = False
        self.closed = False
        self.failure = None
        self.warnings = []
        self._warnings_emitted = 0
        self._persistence_disabled = False
        self.retired_launches = {}
        self.next_id = 1
        self.recovery = None
        self._trusted_launches = None
        try:
            self._restore()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._quarantine(exc)

    def _warn(self, kind, message):
        warning = {"kind": kind, "message": message}
        if warning not in self.warnings:
            self.warnings.append(warning)
            _LOG.warning("%s: %s", kind, message)

    async def _emit_warnings(self):
        while self._warnings_emitted < len(self.warnings):
            warning = self.warnings[self._warnings_emitted]
            self._warnings_emitted += 1
            try:
                await self.ctx.events.emit("ac.author.helper_warning", {"problem_hash": self.key, **warning})
            except Exception as exc:
                _LOG.warning("Could not persist helper warning: %s", exc)

    def _save(self):
        body = json.dumps({
            "version": 1, "problem_hash": self.key, "jobs": list(self.records.values()),
            "retired_launches": self.retired_launches, "next_id": self.next_id, "recovery": self.recovery,
        }, indent=2)
        if len(body.encode()) > MAX_FILE_BYTES:
            raise ValueError("async helper ledger capacity reached")
        write_text_atomic(self.root / "state.json", body)

    def _restore(self):
        path = _local(self.ctx.root_workdir, str((self.root / "state.json").relative_to(self.ctx.root_workdir)))
        if not path.exists():
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("helper directory has artifacts but no launch ledger")
            if any(self.root.parent.glob(f"{self.key}.quarantine-*")):
                raise ValueError("quarantined helper history exists but its replacement ledger is missing")
            return
        data = json.loads(_read(path, MAX_FILE_BYTES))
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("problem_hash") != self.key:
            raise ValueError("async helper checkpoint identity mismatch")
        rows = data.get("jobs")
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError("invalid async helper checkpoint")
        retired = data.get("retired_launches", {})
        def valid_hash(value):
            return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        if (not isinstance(retired, dict) or len(retired) > 1000
                or any(not valid_hash(k) or type(v) is not int or not 0 <= v <= 1000 for k, v in retired.items())
                or sum(retired.values()) + len(rows) > 1000):
            raise ValueError("invalid retired helper launch counts")
        ids = set()
        for rec in rows:
            if not isinstance(rec, dict):
                raise ValueError("invalid async helper record")
            ident = rec.get("agent_id")
            if (not isinstance(ident, str) or not ident.startswith("helper")
                    or not ident[6:].isascii() or not ident[6:].isdigit() or ident in ids
                    or not 1 <= int(ident[6:]) <= 1_000_000_000 or ident != f"helper{int(ident[6:])}"
                    or type(rec.get("round")) is not int or not valid_hash(rec.get("input_hash"))
                    or not isinstance(rec.get("status"), str) or rec["status"] not in _TERMINAL | {"running"}
                    or any(not isinstance(rec.get(k), str) for k in ("role", "task"))
                    or type(rec.get("include_workspace")) is not bool
                    or "report_path" not in rec
                    or (rec.get("report_path") is not None and not isinstance(rec["report_path"], str))):
                raise ValueError("invalid async helper record")
            ids.add(ident)
        next_id = data.get("next_id", max((int(i[6:]) for i in ids), default=0) + 1)
        if type(next_id) is not int or not 1 <= next_id <= 1_000_000_001 or any(int(i[6:]) >= next_id for i in ids):
            raise ValueError("invalid next helper ID")
        recovery = data.get("recovery")
        if recovery is not None and (not isinstance(recovery, dict)
                or type(recovery.get("disabled")) is not bool or not isinstance(recovery.get("message"), str)
                or not isinstance(recovery.get("kind", "checkpoint_quarantined"), str)):
            raise ValueError("invalid helper recovery warning")
        self.retired_launches, self.next_id, self.recovery = dict(retired), next_id, recovery
        # Launch counts are independent of artifact validity. Quarantine must
        # not reset a known allowance or reuse an earlier helper's identity.
        self._trusted_launches = dict(retired)
        for rec in rows:
            identity = rec["input_hash"]
            self._trusted_launches[identity] = self._trusted_launches.get(identity, 0) + 1
        if recovery is not None:
            self._warn(recovery.get("kind", "checkpoint_quarantined"), recovery["message"])
            if recovery["disabled"]:
                self.failure = recovery["message"]
        migrated = set()
        for rec in rows:
            ident = rec["agent_id"]
            store = DelegationContext(self.root / ident / "context", round=rec["round"], persist_omissions=True)
            manifest_path = _local(self.root, f"{ident}/context/manifest.json")
            manifest = json.loads(_read(manifest_path, MAX_FILE_BYTES))
            files = manifest.get("files") if isinstance(manifest, dict) else None
            if not isinstance(files, list) or len(files) > MAX_FILES:
                raise ValueError("invalid async helper artifact manifest")
            total = 0
            for meta in files:
                if not isinstance(meta, dict) or not isinstance(meta.get("source"), str):
                    raise ValueError("invalid async helper artifact metadata")
                name = meta["path"]
                artifact = _local(store.root, name)
                if name in store.files:
                    raise ValueError("duplicate async helper artifact")
                body = _read(artifact, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
                total += len(body)
                if len(body) != meta["bytes"] or hashlib.sha256(body).hexdigest() != meta["sha256"]:
                    raise ValueError("async helper artifact digest mismatch")
                store.files[name] = body.decode("utf-8")
                store.metadata[name] = meta
            store._bytes = total
            self.records[ident] = rec
            self.stores[ident] = store
            # Older ledgers embedded every omission. Migrate only after the
            # whole recovered tree has passed validation, preserving allowances.
            omissions = rec.get("context_omissions", manifest.get("omissions", []))
            count = rec.get("context_omission_count", len(omissions) if isinstance(omissions, list) else 0)
            if (not isinstance(omissions, list) or any(not isinstance(m, str) for m in omissions)
                    or type(count) is not int or count < len(omissions)):
                raise ValueError("invalid async helper context omissions")
            store.omissions = _omission_details(omissions, count)
            if "context_omissions" in rec or manifest.get("omissions") != store.omissions:
                migrated.add(ident)
            rec["context_omission_count"] = count
            rec.pop("context_omissions", None)
            report_path = rec.get("report_path")
            if report_path is not None and (report_path not in store.metadata or not report_path.startswith(f"helpers/{ident}/")):
                raise ValueError("invalid async helper report reference")
            store.files.clear()
        try:
            # Do not modify a recovered tree until every referenced artifact passed.
            for ident, rec in self.records.items():
                if ident in migrated:
                    self.stores[ident].save_manifest()
                if rec["status"] == "running":
                    store = self.stores[ident]
                    store.files.update({p: self.artifact_text(store, p) for p in store.metadata})
                    rec.update(status="interrupted", error="Process interrupted; only published work recovered.")
                    self._handoff(rec, "", completed=False)
                self.stores[ident].files.clear()
            self._save()
        except OSError as exc:
            self._persistence_disabled = True
            self.failure = f"Helper recovery could not persist state: {type(exc).__name__}"
            self._warn("recovery_write_failed", self.failure)

    def _quarantine(self, exc):
        self.records.clear()
        self.stores.clear()
        self.retired_launches = self._trusted_launches or {}
        disabled = self._trusted_launches is None or bool(self.recovery and self.recovery["disabled"])
        message = f"Prior helper checkpoint rejected ({type(exc).__name__}); its artifacts will not be used."
        try:
            parent = _local(self.ctx.root_workdir, "async-helpers")
            destination = parent / f"{self.key}.quarantine-{uuid4().hex}"
            self.root.rename(destination)
            message += f" Quarantined at {destination.relative_to(self.ctx.root_workdir)}."
            if disabled:
                message += " Launch history is untrusted; delegation is disabled, but the Author can continue."
            self.recovery = {"message": message, "disabled": disabled}
            self._save()
        except (OSError, ValueError) as quarantine_error:
            # In particular, never follow a symlinked parent or overwrite an
            # unmovable checkpoint just to make delegation available again.
            self._persistence_disabled = True
            disabled = True
            message += f" Recovery persistence failed ({type(quarantine_error).__name__}); delegation is disabled."
        if disabled:
            self.failure = message
        self._warn("checkpoint_quarantined", message)

    def artifacts(self, ident):
        store = self.stores[ident]
        with store._lock:
            return [p for p in store.metadata if p.startswith(f"helpers/{ident}/")]

    @staticmethod
    def artifact_text(store, path):
        with store._lock:
            if path in store.files:
                return store.files[path]
            body = _read(_local(store.root, path), MAX_FILE_BYTES)
            meta = store.metadata[path]
            if len(body) != meta["bytes"] or hashlib.sha256(body).hexdigest() != meta["sha256"]:
                raise ValueError("async helper artifact digest mismatch")
            return body.decode("utf-8")

    def read(self, path, offset=0, max_chars=12000):
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or max_chars < 1:
            return json.dumps({"error": "invalid character offset or limit"})
        parts = path.split("/") if isinstance(path, str) else []
        if len(parts) < 3 or parts[0] not in {"helpers", "helper-inputs"} or parts[1] not in self.stores:
            return json.dumps({"error": "unknown helper artifact"})
        store = self.stores[parts[1]]
        if parts[0] == "helper-inputs":
            if parts[2:] == ["omissions.json"]:
                body = json.dumps({"context_omission_count": self.records[parts[1]].get("context_omission_count", 0),
                                   "details": store.omissions})
                end = min(len(body), offset + min(max_chars, 24000))
                return json.dumps({"path": path, "content": body[offset:end], "total_chars": len(body),
                                   "next_offset": end if end < len(body) else None})
            if parts[2:] != ["task.txt"]:
                return json.dumps({"error": "unknown helper input"})
            local_path = "task.txt"
        else:
            local_path = path
        if local_path != "task.txt" and path not in self.artifacts(parts[1]):
            return json.dumps({"error": "unknown helper artifact"})
        with store._lock:
            cached = local_path in store.files
            if not cached:
                store.files[local_path] = self.artifact_text(store, local_path)
            try:
                result = json.loads(store.view([local_path]).read(local_path, offset, max_chars))
                result["path"] = path
                return json.dumps(result)
            finally:
                if not cached:
                    store.files.pop(local_path, None)

    def manifest_entries(self):
        entries = []
        for ident, store in self.stores.items():
            with store._lock:
                entries.extend(dict(store.metadata[p]) for p in self.artifacts(ident))
        return entries

    def _selected(self, ids):
        if ids is None:
            return list(self.records)
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in self.records for i in ids):
            raise ValueError("agent_ids must name existing helpers")
        return list(dict.fromkeys(ids))

    def finished(self, ident):
        task = self.tasks.get(ident)
        return self.records[ident]["status"] in _TERMINAL and (task is None or task.done())

    def status(self, ids=None):
        rows = []
        for ident in self._selected(None if ids == [] else ids):
            rec = self.records[ident]
            # Full task text is already durable; do not replay it on every poll.
            rows.append({**{k: v for k, v in rec.items() if k != "task"},
                         "task_summary": rec["task"][:300], "task_path": f"helper-inputs/{ident}/task.txt",
                         "context_omissions_path": f"helper-inputs/{ident}/omissions.json",
                         "artifacts": self.artifacts(ident)})
        return {"helpers": rows, "closed": self.closed, "error": self.failure, "warnings": list(self.warnings),
                "note": "Results are unreviewed, versioned snapshots. Read artifacts with read_context; only the Author edits the manuscript."}

    async def wait(self, ids, timeout_s):
        selected = self._selected(ids)
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s < 0 or timeout_s > 600):
            raise ValueError("timeout_s must be between 0 and 600; waiting does not cancel helpers")
        pending = [self.tasks[i] for i in selected if i in self.tasks and not self.tasks[i].done()]
        if pending and not any(self.finished(i) for i in selected):
            await asyncio.wait(pending, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        return self.status(selected or None)

    async def cancel(self, ids=None, reason="Cancelled by the Author"):
        selected = self._selected(ids)
        pending = []
        for ident in selected:
            task = self.tasks.get(ident)
            if task is not None and not task.done():
                # A second cancellation can interrupt the first one's accounting.
                if not task.cancelling():
                    self.records[ident]["error"] = reason
                    task.cancel()
                pending.append(task)
        if pending:
            results, cancelled = await _join_tasks(pending, deadline=self._close_deadline or time.monotonic() + SHUTDOWN_GRACE_S)
            if results is None:
                self._seal_unfinished(selected)
            if cancelled is not None:
                raise cancelled
        return self.status(selected or None)

    async def close(self):
        self.closed = True
        if self._close_task is None:
            self._close_deadline = time.monotonic() + SHUTDOWN_GRACE_S
            self._close_task = asyncio.create_task(self._close())
        results, cancelled = await _join_tasks([self._close_task], deadline=self._close_deadline)
        if results is None:
            self._shutdown_sealed = True
            self._warn("cleanup_timeout", "Helper shutdown grace expired; retained work is sealed before research continues.")
            self._seal_unfinished(list(self.records))
            self._close_task.cancel()
            self._close_task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            results = []
        for result in results:
            if isinstance(result, Exception):
                self._warn("cleanup_failed", f"{type(result).__name__}: {result}")
            elif isinstance(result, BaseException):
                raise result
        if cancelled is not None:
            raise cancelled

    async def _close(self):
        await self.cancel(reason="Research phase ended; saved work remains available")
        if self._shutdown_sealed:
            return
        if not self._persistence_disabled:
            try:
                self._save()
            except Exception as exc:
                self._warn("ledger_write_failed", f"{type(exc).__name__}: {exc}")
        if self.failure:
            self._warn("bookkeeping_failed", self.failure)
        await self._emit_warnings()

    def _seal_unfinished(self, ids):
        """Fence late publications; never pretend a timed-out join reconciled usage."""
        for ident in ids:
            task = self.tasks.get(ident)
            if task is None or task.done() or ident in self._detached:
                continue
            self._detached.add(ident)
            self.failure = "A helper's shutdown is incomplete; further delegation is disabled until provider reconciliation."
            self.recovery = {"message": self.failure, "disabled": True, "kind": "shutdown_incomplete"}
            rec, job = self.records[ident], self._jobs[ident]
            seat, view = job["seat"], job["view"]
            view.close()
            client = getattr(seat, "_client", None)
            if client is not None:
                try:
                    client.terminate()
                except Exception:
                    pass
            rec.update(status="interrupted", shutdown_incomplete=True, usage_unresolved=True,
                       error="Helper shutdown grace expired. Cancellation requested; final provider usage is unresolved.",
                       cost_usd=float(seat.tracker.counters.usd), duration_s=time.monotonic() - job["started"])
            try:
                self._handoff(rec, getattr(seat, "interrupted_report", None) or "", completed=False)
            except Exception as exc:
                self._warn("handoff_failed", f"{ident}: {type(exc).__name__}")
            finally:
                self.stores[ident].seal()
            work = job.get("work")
            if work is not None:
                work.add_done_callback(lambda done, i=ident, s=seat: self._late_usage(i, s, done))
                # Only after salvaging and fencing may we interrupt a stuck
                # cancellation handler. Its missing receipt stays unresolved.
                if not work.done():
                    work.cancel()
            self._warn("shutdown_incomplete", f"{ident}: {rec['error']}")
        if not self._persistence_disabled:
            try:
                self._save()
            except Exception as exc:
                self._warn("ledger_write_failed", f"{type(exc).__name__}: {exc}")

    def _late_usage(self, ident, seat, task):
        # A late receipt must not overwrite a resumed session's state or reports.
        error = None if task.cancelled() else task.exception()
        receipt = {"agent_id": ident, "input_hash": self.records[ident]["input_hash"],
                   "cost_usd": float(seat.tracker.counters.usd), "tokens": seat.tracker.counters.tokens,
                   "usage_unresolved": True, "error_type": type(error).__name__ if error else None,
                   "note": "Cumulative recorded helper usage, already propagated to shared counters; not an additional charge or invoice reconciliation."}
        try:
            write_text_atomic(self.root / "late-usage" / f"{ident}-{uuid4().hex}.json", json.dumps(receipt))
        except Exception as exc:
            _LOG.warning("Could not save late helper usage (%s): %s", ident, exc)

    async def launch(self, lead, tasks, briefing, call_deadline=None, snapshot=None):
        from .multi_author import ROLE_DESCRIPTIONS, SubAuthorSeat
        from mathagents import APIClient, load_solver_config

        if self.closed or self.failure:
            raise ValueError(self.failure or "helper session is closed")
        cfg = lead._delegation_cfg()
        lead.tracker.check()
        if not isinstance(tasks, list) or not tasks or len(tasks) > cfg["max_threads"]:
            raise ValueError(f"provide 1 to {cfg['max_threads']} tasks")
        active = sum(not t.done() for t in self.tasks.values())
        if active + len(tasks) > cfg["max_threads"]:
            raise ValueError("Helper slots are occupied; wait for or cancel selected helpers before launching more.")
        identity = input_key(lead._current_inp)
        used = self.retired_launches.get(identity, 0) + sum(r["input_hash"] == identity for r in self.records.values())
        if cfg["max_tasks_per_turn"] is not None and used + len(tasks) > cfg["max_tasks_per_turn"]:
            raise ValueError("This Author turn's helper launch allowance is exhausted (including recovered jobs).")
        if sum(self.retired_launches.values()) + len(self.records) + len(tasks) > 1000:
            raise ValueError("Helper ledger capacity reached")
        remaining = lead.tracker.remaining_wallclock_s()
        if remaining is None and call_deadline is not None:
            remaining = call_deadline - time.monotonic()
        if remaining is None:
            raise ValueError("Asynchronous helpers require a finite workflow or call deadline.")
        remaining -= cfg["synthesis_reserve_s"]
        cap = cfg["helper_timeout_s"]
        if cap is not None:
            remaining = min(remaining, cap)
        if remaining <= 0:
            raise ValueError("Not enough research time remains outside the synthesis reserve.")
        normalized = []
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("task"), str) or not task["task"].strip():
                raise ValueError("Each helper needs a nonempty task")
            role = task.get("role")
            previous = task.get("agent_id")
            if previous == "":
                previous = None
            if previous is not None:
                if not isinstance(previous, str) or previous not in self.records or not self.finished(previous):
                    raise ValueError("Continue only a finished helper; wait or cancel it first")
                role = self.records[previous]["role"]
            if role not in cfg["roles"] or role not in ROLE_DESCRIPTIONS:
                raise ValueError("Unknown helper role")
            shared = task.get("include_workspace", True)
            if type(shared) is not bool:
                raise ValueError("include_workspace must be boolean")
            if previous and not shared and self.records[previous]["include_workspace"]:
                raise ValueError("A continued helper cannot forget shared context; start a fresh blind helper")
            dependencies = task.get("depends_on", [])
            if (not isinstance(dependencies, list) or any(not isinstance(i, str) or i not in self.records
                    or not self.finished(i) for i in dependencies)):
                raise ValueError("depends_on must name finished helpers")
            normalized.append((task, role, previous, shared, dependencies))

        scheduled = []
        snapshot = snapshot or describe_snapshot(lead._current_inp, warnings=["Files are turn-start inputs; no launch snapshot was supplied."])
        research_deadline = time.monotonic() + remaining
        for task, role, previous, shared, dependencies in normalized:
            ident = f"helper{self.next_id + len(scheduled)}"
            inp = lead._current_inp
            model = cfg["role_models"].get(role) or cfg["subagent_model"] or lead.ctx.model_for(lead, lead.MODEL)
            model_cfg = load_solver_config(model)
            api_cap = model_cfg.get("max_wallclock_per_call_s", inspect.signature(APIClient).parameters["max_wallclock_per_call_s"].default)
            job_deadline = research_deadline
            if api_cap is not None:
                job_deadline = min(job_deadline, time.monotonic() + float(api_cap))
            job_seconds = max(0.0, job_deadline - time.monotonic())
            rec = {
                "agent_id": ident, "role": role, "model": str(model), "task": task["task"],
                "round": inp.round, "input_hash": identity, "include_workspace": shared,
                "continued_from": previous, "status": "running", "error": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "deadline_utc": datetime.fromtimestamp(time.time() + job_seconds, timezone.utc).isoformat(),
                "cost_usd": 0.0, "duration_s": 0.0, "report_path": None,
                "snapshot": snapshot["metadata"] if shared else {"kind": "blind"},
                "context_omission_count": 0,
            }
            store = DelegationContext(self.root / ident / "context", round=inp.round, persist_omissions=True)
            def omit(message):
                rec["context_omission_count"] += 1
                if len(store.omissions) < MAX_OMISSION_DETAILS:
                    store.omissions.append(message)

            def import_file(name, body, *, source, provenance=None):
                if name not in store.metadata and (len(store.metadata) >= INPUT_MAX_FILES
                        or store._bytes + len(body.encode()) > INPUT_MAX_BYTES):
                    raise ValueError("input capacity reached; output headroom is reserved")
                return store.put(name, body, source=source, provenance=provenance)

            import_file("task.txt", task["task"], source=f"Author input {identity}")
            if shared:
                import_file("launch_snapshot.json", json.dumps(snapshot["metadata"]), source="launch snapshot provenance")
                for name, body in snapshot["files"].items():
                    import_file("round/" + name, body, source=f"{snapshot['metadata']['files'][name]['source']}, snapshot {snapshot['metadata']['sha256']}")
                with lead._context._lock:
                    for name, body in lead._context.files.items():
                        if name.removeprefix("round/") in CANONICAL_INPUTS or name.startswith("compute/"):
                            continue
                        import_file(name, body, source=lead._context.metadata[name]["source"])
                    for message in lead._context.omissions:
                        omit(message)
            required = set(dependencies) | ({previous} if previous else set())
            sources = list(dict.fromkeys([*dependencies, *([previous] if previous else []),
                                         *(self.records if shared else [])]))
            for source in sources:
                other = self.stores[source]
                for name in self.artifacts(source):
                    try:
                        # Do not download/read entire omitted histories merely
                        # to discover that their bytes cannot fit the input budget.
                        if name not in store.metadata and (len(store.metadata) >= INPUT_MAX_FILES
                                or store._bytes + other.metadata[name]["bytes"] > INPUT_MAX_BYTES):
                            raise ValueError("input capacity reached; output headroom is reserved")
                        body = self.artifact_text(other, name)
                        import_file(name, body, source=other.metadata[name]["source"],
                                  provenance=other.metadata[name].get("provenance"))
                    except (ValueError, OSError) as exc:
                        if source in required:
                            raise ValueError(f"Required helper artifact {name} could not be included: {exc}") from exc
                        omit(f"{name}: {exc}")
            if shared:
                with lead._context._lock:
                    for name, body in lead._context.files.items():
                        if name.startswith("compute/"):
                            try:
                                import_file(name, body, source=lead._context.metadata[name]["source"])
                            except ValueError as exc:
                                omit(f"{name}: {exc}")
            store.omissions = _omission_details(store.omissions, rec["context_omission_count"])
            store.save_manifest()
            view = store.view(publisher=f"helpers/{ident}", reserve_report=True)
            seat = SubAuthorSeat(lead.ctx, model_ref=model, name=f"SubAuthor.{ident}",
                                 parent_budget_scope=lead.tracker.scope)
            seat.context_view = view
            seat.MAX_TOOL_CALLS = cfg["seat_max_tool_calls"]
            seat_input = dict(
                role=role, task=task["task"], problem=inp.problem, round=inp.round,
                briefing=str(briefing or ""), include_workspace=shared,
                answer_tex=snapshot["files"]["answer.tex"] if shared else "",
                research_notes_tex=snapshot["files"]["research_notes.tex"] if shared else "",
                references_bib=snapshot["files"]["references.bib"] if shared else "", remaining_seconds=job_seconds,
                started_at_utc=rec["started_at"], deadline_utc=rec["deadline_utc"],
                wrapup_seconds=cfg["wrapup_reserve_s"], asynchronous=True,
                context_notice=(f"Task {ident}; source Author round {inp.round}, input SHA256 {identity}. "
                                "The lead can work concurrently. Publish useful work as it becomes available; "
                                "files only in your sandbox and private reasoning are not shared. "
                                "Use read_context('manifest.json') for exact available snapshots and omissions. "
                                + ("Launch snapshot: " + json.dumps(snapshot["metadata"]) + " " if shared else "Blind assignment. ")
                                + (f"Continue {previous} from its published files, not a restored sandbox. " if previous else "")),
            )
            scheduled.append((rec, seat, seat_input, job_deadline, view))
        # No paid work can start before the launch ledger is durable.
        for rec, _, _, _, view in scheduled:
            self.records[rec["agent_id"]] = rec
            self.stores[rec["agent_id"]] = view.store
        self.next_id += len(scheduled)
        try:
            self._save()
        except Exception:
            self.next_id -= len(scheduled)
            for rec, _, _, _, view in scheduled:
                self.records.pop(rec["agent_id"])
                self.stores.pop(rec["agent_id"])
                view.close()
            raise
        for args in scheduled:
            rec = args[0]
            self._jobs[rec["agent_id"]] = {"seat": args[1], "view": args[-1], "started": time.monotonic(), "work": None}
            task = asyncio.create_task(self._run(lead, *args), name=f"AsyncHelper-{rec['agent_id']}")
            self.tasks[rec["agent_id"]] = task
            task.add_done_callback(lambda task, rec=rec: self._observe(rec, task))
        return self.status([r[0]["agent_id"] for r in scheduled])

    def _observe(self, rec, task):
        try:
            if rec["agent_id"] in self._detached:
                if not task.cancelled():
                    task.exception()
                return
            if task.cancelled() and rec["status"] == "running":
                rec.update(status="cancelled", error=rec["error"] or "Cancelled before helper started")
                self._handoff(rec, "", completed=False)
                self._save()
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()
        except Exception as exc:
            self.failure = f"Helper bookkeeping failed: {type(exc).__name__}: {exc}"
            for other in self.tasks.values():
                if not other.done() and not other.cancelling():
                    other.cancel()
        finally:
            # Finished input snapshots are disk-backed; otherwise a day's
            # repeated manuscript/context copies accumulate in process RAM.
            store = self.stores[rec["agent_id"]]
            with store._lock:
                store.files.clear()
            self._jobs.pop(rec["agent_id"], None)

    def _handoff(self, rec, report, *, completed):
        store = self.stores[rec["agent_id"]]
        artifacts = self.artifacts(rec["agent_id"])
        if not completed:
            report = (
                "# Incomplete helper handoff (unreviewed)\n\n"
                f"Source Author round: {rec['round']}; input SHA256: {rec['input_hash']}\n\n"
                f"Status: {rec['status']}. {rec.get('error') or ''}\n\n"
                + (report + "\n\n" if report else "No final model report was received.\n\n")
                + "Published artifacts (proofs, code and data are not independently verified):\n"
                + ("\n".join(f"- {p}" for p in artifacts) or "- None")
            )
        path = f"helpers/{rec['agent_id']}/final-response.md"
        if path in store.metadata:
            rec["report_path"] = path
            return
        try:
            rec["report_path"] = store.put(path, report, source=f"helper {rec['agent_id']}, input {rec['input_hash']}")
        except (ValueError, OSError) as exc:
            rec["artifact_error"] = f"Final report persistence failed: {type(exc).__name__}"
            if rec["status"] == "completed":
                rec["status"] = "incomplete"

    async def _run(self, lead, rec, seat, inp, deadline, view):
        started = time.monotonic()
        report = ""
        work = None
        try:
            await lead.events.emit("ac.author.helper_start", dict(rec))
            inp["remaining_seconds"] = max(0.0, deadline - time.monotonic())
            if inp["remaining_seconds"] <= 0:
                raise TimeoutError("Helper deadline reached before invocation")
            work = asyncio.create_task(seat(**inp))
            self._jobs[rec["agent_id"]]["work"] = work
            while not work.done():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Research deadline reached (synthesis reserve preserved)")
                lead.tracker.check()
                await asyncio.wait([work], timeout=min(1.0, left))
            out = work.result()
            report = out.report.strip()
            outcomes = getattr(seat, "provider_outcomes", [])
            last = outcomes[-1] if outcomes else {}
            if last.get("status") in {"failed", "cancelled", "incomplete"}:
                rec.update(status="failed", error=f"Provider ended with status {last['status']}; any returned text is partial.")
            elif report:
                rec["status"] = "completed"
            else:
                rec.update(status="failed", error="EmptyResponse: provider returned no final report")
        except asyncio.CancelledError:
            rec.update(status="cancelled", error=rec["error"] or "Helper cancelled")
            raise
        except BudgetExhausted as exc:
            completed = getattr(exc, "completed_output", None)
            report = getattr(completed, "report", "") or ""
            rec.update(status="incomplete", error=str(exc))
            for ident, task in self.tasks.items():
                if ident != rec["agent_id"] and not task.done() and not task.cancelling():
                    self.records[ident]["error"] = "Shared research budget exhausted"
                    task.cancel()
        except Exception as exc:
            rec.update(status="incomplete" if isinstance(exc, TimeoutError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        finally:
            cancelled = None
            if work is not None and not work.done():
                if not work.cancelling():
                    work.cancel()
                results, cancelled = await _join_tasks([work], deadline=self._close_deadline or time.monotonic() + SHUTDOWN_GRACE_S)
                if results is None:
                    self._seal_unfinished([rec["agent_id"]])
            if rec["agent_id"] in self._detached:
                if cancelled is not None:
                    raise cancelled
                return
            report = report or getattr(seat, "interrupted_report", None) or ""
            try:
                self._handoff(rec, report, completed=rec["status"] == "completed")
            finally:
                view.close()
            rec["duration_s"] = time.monotonic() - started
            rec["cost_usd"] = float(seat.tracker.counters.usd)
            rec["checkpoint_errors"] = list(seat._helper_sandbox.failures) if seat._helper_sandbox else []
            self._save()
            try:
                await lead.events.emit("ac.author.subagent_done", dict(rec))
            finally:
                if cancelled is not None:
                    raise cancelled
