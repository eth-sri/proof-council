"""Bounded, invocation-owned helper checkpoints across hosted tool boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import threading
import time

from proofstack.atomic import write_text_atomic
from proofstack.agents.ac.delegation_context import MAX_FILE_BYTES, TEXT_SUFFIXES, _tool


CHECKPOINT_ROOT = "/mnt/data/checkpoints"
MAX_CHECKPOINT_FILES = 16
MAX_CHECKPOINT_BYTES = 4_000_000
MAX_CHECKPOINT_VERSIONS = 64
MAX_SCAN_ENTRIES = 256
TRANSFER_TIMEOUT_S = 60.0


class CheckpointError(ValueError):
    """Only fixed, credential-free reason codes may reach tool output."""


class HelperSandbox:
    def __init__(self, view, workdir):
        self.view = view
        self.workdir = workdir
        self.container = None
        self.client_factory = None
        self._lock = threading.RLock()
        self._closed = False
        self._uploads = set()
        self._files = {}
        self._versions = {}
        self._attachments = {}
        self._last_boundary = None
        self._last_result = None
        self.failures = []

    def bind(self, container, client_factory):
        self.container = container
        self.client_factory = client_factory

    def _check(self, deadline):
        if self._closed or self.view.closed or time.monotonic() >= deadline:
            raise TimeoutError("helper checkpoint transfer closed or deadline reached")

    def _client(self, deadline):
        self._check(deadline)
        if self.client_factory is None or self.container is None:
            raise RuntimeError("helper checkpoint transport unavailable")
        return self.client_factory(timeout=min(10.0, max(0.001, deadline - time.monotonic())))

    @staticmethod
    def _name(path):
        if not isinstance(path, str) or len(path) > 512:
            raise ValueError("checkpoint path must be a short string")
        p = PurePosixPath(path)
        if (str(p) != path or str(p.parent) != CHECKPOINT_ROOT or "\\" in path
                or p.name.startswith(".") or p.suffix.lower() not in TEXT_SUFFIXES
                or p.name.lower() in {"auth.json", "credentials.json", "secrets.json"}):
            raise ValueError("only flat UTF-8 files in /mnt/data/checkpoints are transferable")
        return p.name

    def _download(self, cid, cf, deadline, remaining):
        size = getattr(cf, "bytes", None)
        limit = min(MAX_FILE_BYTES, remaining)
        if size is not None and (type(size) is not int or size < 0 or size > limit):
            raise CheckpointError("file_size_limit")
        client = self._client(deadline)
        chunks, count = [], 0
        # Stream with an actual byte ceiling; metadata alone cannot bound downloads.
        with client.containers.files.content.with_streaming_response.retrieve(
            container_id=cid, file_id=cf.id,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=65536):
                self._check(deadline)
                count += len(chunk)
                if count > limit:
                    raise CheckpointError("file_size_limit")
                chunks.append(chunk)
        self._check(deadline)
        if size is not None and count != size:
            raise CheckpointError("download_size_mismatch")
        return b"".join(chunks).decode("utf-8", errors="strict")

    def _delete(self, ids):
        deadline = time.monotonic() + 3.0
        for fid in ids:
            if time.monotonic() >= deadline:
                break
            try:
                self.client_factory(timeout=max(0.001, deadline - time.monotonic())).files.delete(fid)
            except Exception:
                pass  # Cleanup must not replace a completed helper reply.

    def close(self):
        with self._lock:
            self._closed = True
            ids = list(self._uploads)
            self._uploads.clear()
        return ids

    def _capture(self, cid, call_id, cf, content, deadline):
        name = self._name(cf.path)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        key = (name, digest)
        provenance = {"kind": "sandbox_file", "container_id": cid,
                      "file_id": cf.id, "call_id": call_id, "sandbox_path": cf.path}
        with self._lock:
            self._check(deadline)
            previous = self._versions.get(key)
            if previous is not None:
                # Keep immutable bytes once, but never reuse an old capture receipt.
                return {**previous, "provenance": provenance}
            if len(self._versions) >= MAX_CHECKPOINT_VERSIONS:
                raise CheckpointError("version_limit")
            artifact = f"checkpoints/{digest}/{name}"
            stored_path = f"{self.view.publisher}/{artifact}"
            if self.view.store.files.get(stored_path) == content:
                stored = {"status": "published", **self.view.store.metadata[stored_path]}
            else:
                stored = json.loads(self.view._publish(artifact, content, provenance=provenance))
            if stored.get("status") != "published":
                raise CheckpointError("artifact_rejected")
            uploaded_id = self._attachments.get((name, digest))
        created = uploaded_id is None
        if created:
            try:
                uploaded_id = self._client(deadline).files.create(
                    file=(name, content.encode("utf-8")), purpose="user_data",
                ).id
            except Exception as exc:
                raise CheckpointError("upload_error") from exc
        with self._lock:
            if self._closed or self.view.closed or time.monotonic() >= deadline:
                abandoned = True
            else:
                abandoned = False
                self._uploads.add(uploaded_id)
                self._attachments[name, digest] = uploaded_id
                record = {"sandbox_path": cf.path, "artifact_path": stored["path"],
                          "sha256": digest, "bytes": stored["bytes"], "provenance": provenance,
                          "attachment_path": f"/mnt/data/{uploaded_id}-{name}",
                          "attachment_id": uploaded_id}
                self._versions[key] = record
        if abandoned:
            if created:
                self._delete([uploaded_id])
            raise TimeoutError("checkpoint upload completed after transfer closed")
        return record

    def checkpoint(self, messages, call_deadline_monotonic_s=None):
        deadline = min(time.monotonic() + TRANSFER_TIMEOUT_S,
                       float("inf") if call_deadline_monotonic_s is None else call_deadline_monotonic_s)
        ci = next((m for m in reversed(messages or []) if m.get("type") == "code_interpreter_call"), None)
        if ci is None:
            return {"status": "not_started", "files": []}
        boundary = (ci.get("container_id"), ci.get("id"))
        with self._lock:
            self._check(deadline)
            if boundary == self._last_boundary and self._last_result["status"] == "passed":
                return self._last_result
        cid, call_id = boundary
        errors, omissions, captured = [], [], {}
        try:
            if not isinstance(cid, str) or not cid or not isinstance(call_id, str) or not call_id:
                raise ValueError("missing harness container provenance")
            candidates = {}
            duplicates = set()
            for index, cf in enumerate(self._client(deadline).containers.files.list(cid, limit=100)):
                self._check(deadline)
                if index >= MAX_SCAN_ENTRIES:
                    raise ValueError("checkpoint scan entry limit exceeded")
                path = str(getattr(cf, "path", ""))
                if not path.startswith(CHECKPOINT_ROOT + "/"):
                    continue
                try:
                    name = self._name(path)
                except ValueError:
                    reason = "nested_cache" if "__pycache__" in PurePosixPath(path).parts else "unsupported_path_or_format"
                    if reason not in omissions:
                        omissions.append(reason)
                    continue
                if name in candidates:
                    duplicates.add(name)
                candidates[name] = cf
            if duplicates:
                errors.append("duplicate_checkpoint_path")
            for name, cf in candidates.items():
                if name in duplicates:
                    continue
                try:
                    retained = {**self._files, **captured}
                    retained.pop(name, None)
                    if len(retained) >= MAX_CHECKPOINT_FILES:
                        raise CheckpointError("retained_file_limit")
                    remaining = MAX_CHECKPOINT_BYTES - sum(f["bytes"] for f in retained.values())
                    content = self._download(cid, cf, deadline, remaining)
                    captured[name] = self._capture(cid, call_id, cf, content, deadline)
                except Exception as exc:
                    # Never emit a provider message or an unvalidated path.
                    reason = str(exc) if isinstance(exc, CheckpointError) else f"checkpoint_file_failed:{type(exc).__name__}"
                    if reason not in errors:
                        errors.append(reason)
        except Exception as exc:
            # Provider messages and rejected paths may contain credentials.
            errors.append(f"checkpoint transfer incomplete ({type(exc).__name__})")
        with self._lock:
            self._check(deadline)
            merged = {**self._files, **captured}
            if len(merged) > MAX_CHECKPOINT_FILES or sum(f["bytes"] for f in merged.values()) > MAX_CHECKPOINT_BYTES:
                errors.append("checkpoint retained-file capacity exceeded")
            else:
                self._files = merged
                self.container["file_ids"] = [f["attachment_id"] for f in self._files.values()]
            result = {"status": "failed" if errors else "passed", "errors": errors,
                      "omissions": omissions, "captured": list(captured.values()),
                      "source_container_id": cid, "source_call_id": call_id,
                      "files": list(self._files.values()),
                      "note": "Only listed checkpoints are carried. After a sandbox change, copy the read-only attachment_path to sandbox_path before use. Missing files were not verified or regenerated."}
            # Persistence failure is surfaced, not silently treated as a successful boundary.
            write_text_atomic(self.workdir / "helper-checkpoints.json", json.dumps(result, indent=2))
            self._last_boundary, self._last_result = boundary, result
            return result

    def _boundary(self, messages, deadline):
        try:
            result = self.checkpoint(messages, deadline)
        except Exception as exc:
            result = {"status": "failed", "errors": [f"checkpoint unavailable ({type(exc).__name__})"], "files": []}
        if result["status"] == "failed":
            with self._lock:
                for error in result["errors"]:
                    if error not in self.failures:
                        self.failures.append(error)
        return result

    def read(self, path="manifest.json", offset=0, max_chars=12000, revision=None, *, messages=None, call_deadline_monotonic_s=None):
        checkpoint = self._boundary(messages, call_deadline_monotonic_s)
        result = json.loads(self.view.read(path, offset, max_chars, revision))
        return json.dumps({**result, "sandbox_checkpoint": checkpoint})

    def publish(self, name, content, *, messages=None, call_deadline_monotonic_s=None):
        checkpoint = self._boundary(messages, call_deadline_monotonic_s)
        result = json.loads(self.view.publish(name, content))
        return json.dumps({**result, "sandbox_checkpoint": checkpoint})

    def publish_file(self, path, *, messages=None, call_deadline_monotonic_s=None):
        checkpoint = self._boundary(messages, call_deadline_monotonic_s)
        try:
            name = self._name(path)
            record = next((r for r in checkpoint.get("captured", []) if r["sandbox_path"] == path), None)
            if (record is None
                    or record["provenance"]["container_id"] != checkpoint["source_container_id"]
                    or record["provenance"]["call_id"] != checkpoint["source_call_id"]):
                raise ValueError("file not captured from the current helper sandbox")
            content = self.view.store.files[record["artifact_path"]]
            result = json.loads(self.view._publish(name, content, provenance=record["provenance"]))
        except (ValueError, KeyError) as exc:
            result = {"error": str(exc) if isinstance(exc, ValueError) else "checkpoint unavailable"}
        if "error" in result:
            with self._lock:
                warning = "sandbox artifact publication failed: " + result["error"]
                if warning not in self.failures:
                    self.failures.append(warning)
        return json.dumps({**result, "sandbox_checkpoint": checkpoint})

    def tools(self):
        descriptions = [desc for _, desc in self.view.tools()]
        return [(self.read, descriptions[0]), (self.publish, descriptions[1]),
                (self.publish_file, _tool("publish_sandbox_artifact",
                    "Download and publish an actual UTF-8 file from this helper's /mnt/data/checkpoints directory. No transcription. Check status and sandbox_checkpoint errors; cite the returned artifact path.",
                    {"path": {"type": "string"}}, ["path"]))]
