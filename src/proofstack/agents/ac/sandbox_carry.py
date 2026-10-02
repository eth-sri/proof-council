"""Own one Author turn's temporary uploads, including late thread results."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxSnapshot:
    abandoned: bool = False
    files: list[tuple[str, str, str]] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)


class SandboxCarryState:
    def __init__(self) -> None:
        self.carried: dict[str, str] = {}
        self.file_ids_to_delete: list[str] = []
        self._container: dict[str, Any] | None = None
        self._closed = False
        self._lock = threading.Lock()

    def bind_container(self, container: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sandbox carry-over turn is closed")
            self._container = container

    def active(self, snapshot: SandboxSnapshot) -> bool:
        with self._lock:
            return not self._closed and not snapshot.abandoned

    def register_upload(self, snapshot: SandboxSnapshot, name: str, file_id: str, path: str) -> bool:
        with self._lock:
            if self._closed or snapshot.abandoned:
                return False
            self.file_ids_to_delete.append(file_id)
            snapshot.files.append((name, file_id, path))
            return True

    def commit(self, snapshot: SandboxSnapshot) -> bool:
        with self._lock:
            if self._closed or snapshot.abandoned:
                return False
            if (snapshot.files or snapshot.empty_files) and self._container is None:
                raise RuntimeError("No container descriptor for sandbox carry-over")
            for name in snapshot.empty_files:
                old = self.carried.pop(name, None)
                if old in self._container["file_ids"]:
                    self._container["file_ids"].remove(old)
            for name, file_id, _path in snapshot.files:
                file_ids = self._container["file_ids"]
                old = self.carried.get(name)
                if old in file_ids:
                    file_ids.remove(old)
                file_ids.append(file_id)
                self.carried[name] = file_id
            return True

    def abandon(self, snapshot: SandboxSnapshot) -> None:
        with self._lock:
            snapshot.abandoned = True

    def close(self) -> list[str]:
        with self._lock:
            self._closed = True
            ids = list(self.file_ids_to_delete)
            self.file_ids_to_delete.clear()
            return ids
