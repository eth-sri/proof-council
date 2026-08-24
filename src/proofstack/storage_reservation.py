"""Cooperative filesystem reservations for persistent CLI workers."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


class StorageReservationError(RuntimeError):
    """Raised when a requested filesystem reservation cannot be admitted."""


@dataclass(frozen=True)
class StorageReservationStatus:
    requested_bytes: int
    active_reserved_bytes: int
    filesystem_free_bytes: int
    minimum_free_bytes: int
    required_free_bytes: int
    filesystem_device: int


class StorageReservationLease:
    """A reservation represented by a locked lease file.

    The registry lock serializes admission. Each admitted process then keeps an
    exclusive lock on its own lease file; if it crashes, the kernel releases
    that lock and the next admission removes the stale file.
    """

    def __init__(
        self,
        *,
        registry_dir: Path,
        workspace_root: Path,
        requested_bytes: int,
        minimum_free_bytes: int,
        owner: str,
    ) -> None:
        self.registry_dir = Path(registry_dir).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.requested_bytes = max(0, int(requested_bytes))
        self.minimum_free_bytes = max(0, int(minimum_free_bytes))
        self.owner = str(owner)
        self.status: StorageReservationStatus | None = None
        self._lease_path: Path | None = None
        self._lease_file: IO[str] | None = None

    @property
    def active(self) -> bool:
        return self._lease_file is not None

    def acquire(self) -> StorageReservationStatus:
        if self.requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")
        if self._lease_file is not None:
            if self.status is None:
                raise RuntimeError("active reservation has no status")
            return self.status

        try:
            self.registry_dir.relative_to(self.workspace_root)
        except ValueError:
            pass
        else:
            raise StorageReservationError(
                "storage reservation registry must be outside the worker workspace"
            )

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        filesystem_device = self.workspace_root.stat().st_dev
        registry_lock_path = self.registry_dir / "registry.lock"

        with registry_lock_path.open("a+", encoding="utf-8") as registry_lock:
            fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
            active_reserved = self._active_reserved_bytes(filesystem_device)
            free_bytes = shutil.disk_usage(self.workspace_root).free
            required_free = (
                self.minimum_free_bytes + active_reserved + self.requested_bytes
            )
            if free_bytes < required_free:
                raise StorageReservationError(
                    "filesystem reservation denied: "
                    f"{free_bytes} bytes free, but {required_free} bytes are "
                    "required for the configured global floor, existing active "
                    "reservations, and this worker"
                )

            lease_path = self.registry_dir / f"lease-{uuid.uuid4().hex}.json"
            lease_file = lease_path.open("x+", encoding="utf-8")
            try:
                fcntl.flock(
                    lease_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                payload = {
                    "version": 1,
                    "owner": self.owner,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "workspace": str(self.workspace_root),
                    "filesystem_device": filesystem_device,
                    "reserved_bytes": self.requested_bytes,
                    "created_at": time.time(),
                }
                json.dump(payload, lease_file, ensure_ascii=False, indent=2)
                lease_file.write("\n")
                lease_file.flush()
                os.fsync(lease_file.fileno())
            except BaseException:
                lease_file.close()
                try:
                    lease_path.unlink()
                except OSError:
                    pass
                raise

            self._lease_path = lease_path
            self._lease_file = lease_file
            self.status = StorageReservationStatus(
                requested_bytes=self.requested_bytes,
                active_reserved_bytes=active_reserved,
                filesystem_free_bytes=free_bytes,
                minimum_free_bytes=self.minimum_free_bytes,
                required_free_bytes=required_free,
                filesystem_device=filesystem_device,
            )
            return self.status

    def release(self) -> None:
        lease_file = self._lease_file
        lease_path = self._lease_path
        if lease_file is None:
            return
        registry_lock_path = self.registry_dir / "registry.lock"
        try:
            with registry_lock_path.open("a+", encoding="utf-8") as registry_lock:
                fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
                self._lease_file = None
                self._lease_path = None
                try:
                    fcntl.flock(lease_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lease_file.close()
                if lease_path is not None:
                    try:
                        lease_path.unlink()
                    except FileNotFoundError:
                        pass
        finally:
            if not lease_file.closed:
                lease_file.close()
            self._lease_file = None
            self._lease_path = None

    def _active_reserved_bytes(self, filesystem_device: int) -> int:
        total = 0
        for lease_path in self.registry_dir.glob("lease-*.json"):
            try:
                lease_file = lease_path.open("r+", encoding="utf-8")
            except OSError:
                continue
            stale = False
            try:
                try:
                    fcntl.flock(
                        lease_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    payload = _read_lease_payload(lease_file)
                    if payload is None:
                        raise StorageReservationError(
                            f"active storage lease is unreadable: {lease_path}"
                        )
                    if int(payload.get("filesystem_device", -1)) == filesystem_device:
                        total += max(0, int(payload.get("reserved_bytes", 0) or 0))
                else:
                    stale = True
            finally:
                lease_file.close()
            if stale:
                try:
                    lease_path.unlink()
                except OSError:
                    pass
        return total

    def __enter__(self) -> "StorageReservationLease":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _read_lease_payload(handle: IO[str]) -> dict[str, Any] | None:
    try:
        handle.seek(0)
        value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


__all__ = [
    "StorageReservationError",
    "StorageReservationLease",
    "StorageReservationStatus",
]
