"""Durable Codex review jobs owned by an editorial invocation, not an HTTP call."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid

from proofstack.atomic import write_text_atomic
from proofstack.agents.cleanup_tools import READ_CHUNK_CHARS, _inline_report


class CleanupReviews:
    def __init__(self, root, workspace, *, snapshot, execute, read_text, admit=None):
        self.directory = root / "review_jobs"
        if self.directory.is_symlink():
            raise ValueError("unsafe cleanup review directory")
        self.directory.mkdir(exist_ok=True)
        self.workspace = workspace
        self.snapshot = snapshot
        self.execute = execute
        self.read_text = read_text
        self.admit = admit
        self.records = {}
        self.tasks = {}
        self.closing = False
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(read_text(self.directory, path.name))
            if (not re.fullmatch(r"[0-9a-f]{32}", path.stem)
                    or record.get("review_id") != path.stem
                    or record.get("status") not in {"running", "completed", "failed", "cancelled", "interrupted"}):
                raise ValueError("invalid cleanup review record")
            if record["status"] == "running":
                record.update(status="interrupted", error="Previous supervisor stopped; reconcile before starting reviews.")
                self._save(record)
            self.records[path.stem] = record

    def _save(self, record):
        write_text_atomic(self.directory / f"{record['review_id']}.json", json.dumps(record, indent=2))

    @staticmethod
    def _summary(record):
        return {key: value for key, value in record.items() if key not in {"task", "fingerprint"}}

    async def start(self, task, purpose="general"):
        if self.closing:
            raise RuntimeError("cleanup invocation has ended")
        if not isinstance(task, str) or not task.strip() or len(task) > 20000:
            raise ValueError("review task must contain 1-20000 characters")
        if purpose not in {"attribution", "general"}:
            raise ValueError("review purpose must be attribution or general")
        snapshot = self.snapshot()
        fingerprint = hashlib.sha256(json.dumps(
            {"task": task.strip(), "purpose": purpose, "snapshot": snapshot}, sort_keys=True).encode()).hexdigest()
        for record in self.records.values():
            if record["fingerprint"] == fingerprint and record["status"] not in {"failed", "cancelled"}:
                return self._summary(record)
        for record in self.records.values():
            if record["status"] == "interrupted":
                raise RuntimeError("An interrupted review needs reconciliation; no new review was started.")
            if record["status"] == "running" or (record["status"] == "completed" and not record["retrieved"]):
                return {**self._summary(record), "note": "No new review started. Retrieve this existing review first."}
        if self.admit is not None:
            self.admit(purpose)
        review_id = uuid.uuid4().hex
        record = {"review_id": review_id, "status": "running", "purpose": purpose,
                  "task": task.strip(), "fingerprint": fingerprint, "retrieved": False, "read_until": 0,
                  "snapshot_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in snapshot.items()}}
        self._save(record)
        self.records[review_id] = record
        # There is no await between persistence and task creation. A dropped
        # response can be recovered by listing jobs without starting paid work.
        self.tasks[review_id] = asyncio.create_task(self._run(record, snapshot))
        return self._summary(record)

    async def _run(self, record, snapshot):
        try:
            result = await self.execute(record["task"], review_id=record["review_id"], snapshot=snapshot)
            report = self.read_text(self.workspace, result["path"])
            record.update(status="completed", path=result["path"], report_length=len(report),
                          report_sha256=hashlib.sha256(report.encode()).hexdigest(), cost_usd=result["cost_usd"])
        except asyncio.CancelledError:
            record.update(status="cancelled", error="Editorial invocation ended; worker shutdown and accounting were joined.")
            raise
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            self._save(record)

    async def status(self, review_id=None, wait_seconds=0):
        if self.closing:
            raise RuntimeError("cleanup invocation has ended")
        if not 0 <= wait_seconds <= 240:
            raise ValueError("wait_seconds must be between 0 and 240")
        if review_id is None:
            return {"reviews": [self._summary(r) for r in self.records.values()]}
        if review_id not in self.records:
            raise ValueError("unknown cleanup review ID")
        job = self.tasks.get(review_id)
        if job is not None and not job.done() and wait_seconds:
            # Cancelling the HTTP waiter must not cancel the paid worker.
            await asyncio.wait({job}, timeout=wait_seconds)
        if job is not None and job.done() and not job.cancelled():
            job.result()
        record = self.records[review_id]
        result = self._summary(record)
        if record["status"] == "completed":
            report = self.read_text(self.workspace, record["path"])
            if hashlib.sha256(report.encode()).hexdigest() != record["report_sha256"]:
                raise ValueError("saved review report changed")
            self.note_read(record["path"], 0, min(len(report), READ_CHUNK_CHARS))
            result = _inline_report({**self._summary(record), "report": report})
        return result

    def note_read(self, path, offset, end):
        for record in self.records.values():
            if record["status"] != "completed" or record.get("path") != path:
                continue
            if offset <= record["read_until"] < end:
                record["read_until"] = min(end, record["report_length"])
                record["retrieved"] = record["read_until"] == record["report_length"]
                self._save(record)

    def attribution_retrieved(self):
        return any(r["purpose"] == "attribution" and r["status"] == "completed" and r["retrieved"]
                   for r in self.records.values())

    async def __aenter__(self):
        return self

    def cancel(self):
        """Stop paid jobs before the transport waits for other helpers to drain."""
        if self.closing:
            return
        self.closing = True
        for task in self.tasks.values():
            if not task.done() and not task.cancelling():
                task.cancel()

    async def __aexit__(self, *exc):
        self.cancel()

        async def drain():
            results = await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise RuntimeError("cleanup review state could not be saved") from result
            for review_id, task in self.tasks.items():
                if task.cancelled() and self.records[review_id]["status"] == "running":
                    record = self.records[review_id]
                    record.update(status="cancelled", error="Invocation ended before the review job started.")
                    self._save(record)

        shutdown = asyncio.create_task(drain())
        interrupted = False
        while not shutdown.done():
            try:
                await asyncio.shield(shutdown)
            except asyncio.CancelledError:
                interrupted = True
        shutdown.result()
        if interrupted:
            raise asyncio.CancelledError
