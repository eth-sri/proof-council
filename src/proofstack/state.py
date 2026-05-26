"""Typed wrappers for run state and file artifacts."""
from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """A typed reference to a file artifact written by some agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    path: Path
    role: str
    mime: str = "text/plain"
    sha256: str
    producer_agent: str
    producer_call_id: str
    parents: list[str] = Field(default_factory=list)


class ArtifactRegistry:
    """In-memory index of every ArtifactRef written during a run.

    Persistence is delegated to the artifact files themselves and the
    event log (``artifact.create`` events). This object is just a fast
    lookup that the future UI can walk without touching the filesystem.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ArtifactRef] = {}
        self._by_role: dict[str, list[ArtifactRef]] = {}
        self._lock = asyncio.Lock()

    async def register(self, ref: ArtifactRef) -> ArtifactRef:
        async with self._lock:
            self._by_id[ref.id] = ref
            self._by_role.setdefault(ref.role, []).append(ref)
        return ref

    def get(self, artifact_id: str) -> ArtifactRef | None:
        return self._by_id.get(artifact_id)

    def by_role(self, role: str) -> list[ArtifactRef]:
        return list(self._by_role.get(role, ()))

    def all(self) -> list[ArtifactRef]:
        return list(self._by_id.values())


def write_artifact(
    *,
    workdir: Path,
    name: str,
    content: str | bytes,
    role: str,
    producer_agent: str,
    producer_call_id: str,
    mime: str | None = None,
    parents: list[str] | None = None,
) -> ArtifactRef:
    """Write content to ``workdir/name`` and return an ArtifactRef.

    The framework hashes content for ``sha256`` and constructs a stable
    artifact id ``f"{producer_call_id}:{name}"``. Synchronous; agents
    typically wrap it in ``await asyncio.to_thread(...)`` for large blobs.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / name
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
        data = content.encode("utf-8")
        chosen_mime = mime or "text/plain"
    else:
        path.write_bytes(content)
        data = content
        chosen_mime = mime or "application/octet-stream"
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(
        id=f"{producer_call_id}:{name}",
        path=path,
        role=role,
        mime=chosen_mime,
        sha256=digest,
        producer_agent=producer_agent,
        producer_call_id=producer_call_id,
        parents=parents or [],
    )


def copy_artifact(*, ref: ArtifactRef, dest: Path) -> Path:
    """Copy an artifact's bytes to ``dest`` (used by the entrypoint to write
    final solutions to ``/data/output``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ref.path, dest)
    return dest


__all__ = [
    "ArtifactRef",
    "ArtifactRegistry",
    "copy_artifact",
    "write_artifact",
]
