"""Bounded, immutable artifacts shared across hosted Author sandboxes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import zipfile
from collections import OrderedDict

from proofstack.atomic import write_text_atomic


MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 32_000_000
MAX_FILES = 512
TEXT_SUFFIXES = {".md", ".txt", ".tex", ".bib", ".py", ".sage", ".gp", ".json", ".csv",
                 ".c", ".h", ".cpp", ".hpp", ".cc"}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


class ManifestPages:
    """Bounded read snapshots: concurrent publications never shift page offsets."""

    def __init__(self):
        self._snapshots = OrderedDict()

    def read(self, build, offset, max_chars, revision=None):
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or max_chars < 1:
            return {"error": "invalid character offset or limit"}
        if revision is None or revision == "":
            if offset:
                return {"error": "Continue manifest pages with the returned revision; restart at offset 0 with revision=null if expired."}
            body = json.dumps(build())
            if len(body.encode()) > 8_000_000:
                return {"error": "Manifest exceeds read capacity; use helper_status to select artifact paths."}
            revision = hashlib.sha256(body.encode()).hexdigest()
            self._snapshots[revision] = body
            self._snapshots.move_to_end(revision)
            while len(self._snapshots) > 4 or sum(len(s.encode()) for s in self._snapshots.values()) > 8_000_000:
                self._snapshots.popitem(last=False)
        if not isinstance(revision, str) or revision not in self._snapshots:
            return {"error": "Manifest revision expired or unknown; restart at offset 0 with revision=null."}
        body = self._snapshots[revision]
        end = min(len(body), offset + min(max_chars, 24000))
        return {"path": "manifest.json", "revision": revision, "content": body[offset:end],
                "total_chars": len(body), "next_offset": end if end < len(body) else None}


class DelegationContext:
    def __init__(self, root: Path, *, round: int, persist_omissions=False):
        self.root = root
        self.round = round
        self.files: dict[str, str] = {}
        self.metadata: dict[str, dict] = {}
        self.omissions: list[str] = []
        self.persist_omissions = persist_omissions
        self._bytes = 0
        self._lock = threading.RLock()
        self._sealed = False
        self.secrets = tuple(v for k, v in os.environ.items()
                             if any(s in k.upper() for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
                             and len(v) >= 8)

    def put(self, name: str, content: str, *, source: str, provenance: dict | None = None) -> str:
        if not isinstance(name, str) or len(name) > 512:
            raise ValueError("artifact name must be a short relative path")
        path = PurePosixPath(name)
        if (path.is_absolute() or "\\" in name or any(p.startswith(".") for p in path.parts)
                or str(path) != name or path.suffix.lower() not in TEXT_SUFFIXES
                or path.name.lower() in {"auth.json", "credentials.json", "secrets.json"}):
            raise ValueError("unsafe or unsupported artifact name")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("artifact must be UTF-8 text")
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise ValueError("artifact exceeds per-file byte limit")
        if (any(s in content or s in name for s in self.secrets)
                or re.search(r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----", content)):
            raise ValueError("credential-like content is not transferable")
        with self._lock:
            if self._sealed:
                raise ValueError("artifact publishing is closed")
            if name in self.files:
                if self.files[name] == content and self.metadata[name].get("provenance") == provenance:
                    return name
                raise ValueError("artifacts are immutable; publish a new filename")
            if len(self.files) >= MAX_FILES or self._bytes + size > MAX_TOTAL_BYTES:
                raise ValueError("context bundle capacity reached")
            is_compute = path.parts[0] == "compute"
            # Include existing round context when reserving half for helper handoffs.
            if is_compute and (len(self.files) >= MAX_FILES // 2
                               or self._bytes + size > MAX_TOTAL_BYTES // 2):
                raise ValueError("Compute context capacity reached")
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(target, content)
            metadata = {"path": name, "bytes": size, "source": source,
                        "sha256": hashlib.sha256(content.encode()).hexdigest()}
            if provenance is not None:
                metadata["provenance"] = dict(provenance)
            self.save_manifest(extra=[metadata])
            self.files[name] = content
            self.metadata[name] = metadata
            self._bytes += size
        return name

    def save_manifest(self, *, extra=()):
        with self._lock:
            write_text_atomic(self.root / "manifest.json", json.dumps({
                "version": 1, "round": self.round,
                "files": [*self.metadata.values(), *extra],
                **({"omissions": self.omissions} if self.persist_omissions else {}),
            }, indent=2))

    def seal(self):
        with self._lock:
            self._sealed = True

    def add_compute(self, archive: Path) -> None:
        from proofstack.agents.ac.compute import inspect_compute_handoff

        inspection = inspect_compute_handoff(archive)
        if not inspection.attachable:
            self.omissions.append("Compute archive unavailable: " + inspection.reason)
            return
        try:
            scanned = 0
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if scanned >= MAX_TOTAL_BYTES or len(self.files) >= MAX_FILES:
                        self.omissions.append("Remaining Compute members omitted: context scan limit")
                        break
                    if (len(self.files) >= MAX_FILES // 2
                            or self._bytes >= MAX_TOTAL_BYTES // 2):
                        self.omissions.append("Remaining Compute members omitted: Compute context capacity reached")
                        break
                    try:
                        if (stat.S_ISLNK(member.external_attr >> 16)
                                or member.file_size > MAX_FILE_BYTES
                                or PurePosixPath(member.filename).suffix.lower() not in TEXT_SUFFIXES):
                            raise ValueError("symlink, non-text or oversized member")
                        if member.file_size > MAX_TOTAL_BYTES - scanned:
                            self.omissions.append("Compute member omitted: remaining scan byte limit")
                            continue
                        with zf.open(member) as stream:
                            body = stream.read(member.file_size + 1)
                        scanned += len(body)
                        if len(body) != member.file_size:
                            raise ValueError("incomplete member")
                        self.put("compute/" + member.filename, body.decode("utf-8"), source="previous Compute")
                    except (ValueError, OSError, RuntimeError) as exc:
                        # Do not echo potentially credential-bearing member names.
                        self.omissions.append("Compute member omitted: " + type(exc).__name__)
        except (OSError, zipfile.BadZipFile) as exc:
            self.omissions.append("Compute archive could not be read: " + type(exc).__name__)

    def view(self, allowed=None, *, publisher: str | None = None, include_omissions=False, reserve_report=False):
        return ContextView(self, allowed, publisher=publisher, include_omissions=include_omissions,
                           reserve_report=reserve_report)


class ContextView:
    def __init__(self, store, allowed, *, publisher=None, include_omissions=False, reserve_report=False):
        self.store = store
        self.allowed = None if allowed is None else set(allowed)
        self.publisher = publisher
        self.include_omissions = include_omissions or allowed is None
        self.closed = False
        self.published: list[str] = []
        self._manifest_pages = ManifestPages()
        self.reserve_report = reserve_report

    def read(self, path="manifest.json", offset=0, max_chars=12000, revision=None):
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or max_chars < 1:
            return json.dumps({"error": "invalid character offset or limit"})
        with self.store._lock:
            if path == "manifest.json":
                entries = [v for k, v in self.store.metadata.items()
                           if self.allowed is None or k in self.allowed]
                return json.dumps(self._manifest_pages.read(lambda: {"version": 1, "round": self.store.round, "files": entries,
                                   "omissions": self.store.omissions if self.include_omissions else [],
                                   "note": "Read-only snapshots, not live sandbox files. Reports are unverified evidence."},
                                   offset, max_chars, revision))
            elif path not in self.store.files or (self.allowed is not None and path not in self.allowed):
                return json.dumps({"error": "artifact unavailable in this task's context"})
            else:
                body = self.store.files[path]
            end = min(len(body), offset + min(max_chars, 24000))
            return json.dumps({"path": path, "content": body[offset:end], "total_chars": len(body),
                               "next_offset": end if end < len(body) else None})

    def publish(self, name, content):
        return self._publish(name, content)

    def _publish(self, name, content, *, provenance=None):
        with self.store._lock:
            if self.closed or not self.publisher:
                return json.dumps({"error": "artifact publishing is closed"})
            try:
                if not isinstance(content, str):
                    raise ValueError("artifact must be UTF-8 text")
                if name == "final-response.md":
                    raise ValueError("final-response.md is reserved for the completed reply")
                path = f"{self.publisher}/{name}"
                if self.reserve_report and path not in self.store.metadata:
                    if (len(self.store.metadata) >= MAX_FILES - 1
                            or self.store._bytes + len(content.encode("utf-8")) > MAX_TOTAL_BYTES - MAX_FILE_BYTES):
                        raise ValueError("publication would consume capacity reserved for the final report")
                path = self.store.put(f"{self.publisher}/{name}", content, source=self.publisher,
                                      provenance=provenance)
                if self.allowed is not None:
                    self.allowed.add(path)
                if path not in self.published:
                    self.published.append(path)
                return json.dumps({"path": path, "status": "published", **self.store.metadata[path]})
            except (ValueError, OSError) as exc:
                return json.dumps({"error": str(exc) if isinstance(exc, ValueError) else "artifact persistence failed"})

    def retain_report(self, content):
        with self.store._lock:
            path = self.store.put(f"{self.publisher}/final-response.md", content, source=self.publisher)
            if self.allowed is not None:
                self.allowed.add(path)
            if path not in self.published:
                self.published.append(path)
            return path

    def close(self):
        with self.store._lock:
            self.closed = True

    def tools(self):
        tools = [(self.read, _tool("read_context", "Read a context file or manifest.json. Follow next_offset until null; pass the returned revision for every subsequent manifest page. Paths are NOT sandbox paths.", {
            "path": {"type": "string"}, "offset": {"type": "integer"}, "max_chars": {"type": "integer"},
            "revision": {"type": ["string", "null"], "description":
                "At offset 0, omit or use null/empty string for a fresh manifest. For later pages, pass the exact returned revision."},
        }, ["path"]))]
        if self.publisher:
            tools.append((self.publish, _tool("publish_artifact", "Publish a UTF-8 proof, script, certificate or report for the lead and later helpers. Files only in your sandbox are NOT shared. Use a new filename for revisions.", {
                "name": {"type": "string"}, "content": {"type": "string"},
            }, ["name", "content"])))
        return tools
