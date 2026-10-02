"""Cross-process Compute admission and conservative resident-memory accounting.

The official runner cannot delegate writable cgroups. This is a polling safety
guard, not a kernel memory quota: leave substantial headroom for allocation
bursts, orchestration, API responses, and processes outside Compute.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import psutil


GiB = 1024**3
REGISTRY_VERSION = 2
MAX_METADATA_BYTES = 4096
REGISTRY_PREFLIGHT_WAIT_S = 3.0
REGISTRY_PREFLIGHT_POLL_S = 0.05


class MemoryRegistryError(RuntimeError):
    """Registry state cannot be trusted for admission or memory accounting."""


def _read_metadata(fd: int, path: Path) -> dict | None:
    raw = os.pread(fd, MAX_METADATA_BYTES + 1, 0)
    if not raw:
        return None
    if len(raw) > MAX_METADATA_BYTES:
        raise MemoryRegistryError(f"memory registry metadata is too large: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MemoryRegistryError(f"invalid memory registry metadata: {path}") from exc
    if not isinstance(value, dict):
        raise MemoryRegistryError(f"memory registry metadata must be an object: {path}")
    return value


@dataclass(frozen=True)
class MemoryPolicy:
    registry: Path
    max_workers: int
    worker_bytes: int
    reserve_bytes: int
    poll_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.worker_bytes < 1 or self.reserve_bytes < 0:
            raise ValueError("invalid Compute memory policy")
        if not 0.05 <= self.poll_seconds <= 1:
            raise ValueError("memory polling interval must be between 0.05 and 1s")


def _check_control(policy: MemoryPolicy, control: int, *, initialize: bool = False) -> None:
    path = policy.registry / "control.json"
    expected = {
        "version": REGISTRY_VERSION,
        "max_workers": policy.max_workers,
        "worker_bytes": policy.worker_bytes,
        "reserve_bytes": policy.reserve_bytes,
    }
    saved = _read_metadata(control, path)
    if saved == expected:
        return
    if saved is None and initialize and not any(policy.registry.glob("slot-*")):
        os.pwrite(control, json.dumps(expected).encode(), 0)
        os.fsync(control)
        return
    raise MemoryRegistryError(
        "Compute workers sharing a registry must use the same memory policy "
        f"and registry version ({REGISTRY_VERSION}): {path}. "
        "Stop all workers and surviving descendants before using a fresh "
        "shared registry; do not migrate an active registry."
    )


def check_memory_registry(policy: MemoryPolicy) -> None:
    """Validate existing control metadata without creating or admitting workers."""
    path = policy.registry / "control.json"
    try:
        control = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if any(policy.registry.glob("slot-*")):
            raise MemoryRegistryError(f"memory registry has slots but no control metadata: {path}")
        return
    try:
        # Active workers briefly hold this lock while sampling; do not turn
        # ordinary contention into a launch failure or wait without a bound.
        deadline = time.monotonic() + REGISTRY_PREFLIGHT_WAIT_S
        while True:
            try:
                fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(REGISTRY_PREFLIGHT_POLL_S, remaining))
                if time.monotonic() >= deadline:
                    raise MemoryRegistryError(
                        f"memory registry is busy after {REGISTRY_PREFLIGHT_WAIT_S:g}s; "
                        f"retry launch preflight: {path}"
                    ) from exc
        _check_control(policy, control)
    finally:
        os.close(control)


def available_memory(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Use both host availability and visible cgroup headroom, without swap."""
    available = int(psutil.virtual_memory().available)
    # Include ancestor limits when the process is in a nested visible cgroup.
    roots = {cgroup_root}
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if controllers == "" or "memory" in controllers.split(","):
                base = cgroup_root if controllers == "" else cgroup_root / "memory"
                roots.add(base)
                relative = Path(path.lstrip("/"))
                if ".." not in relative.parts:
                    child = base / relative
                    while child != base:
                        roots.add(child)
                        child = child.parent
    roots.add(cgroup_root / "memory")
    for root in roots:
        for limit_name, usage_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            limit_file = root / limit_name
            if not limit_file.exists():
                continue
            raw = limit_file.read_text().strip()
            if raw != "max":
                available = min(
                    available, int(raw) - int((root / usage_name).read_text())
                )
    return max(0, available)


def markers_rss(markers: list[dict]) -> dict[str, tuple[int, bool]]:
    # Reuse cleanup's marker scan so detached children are counted as well.
    from proofstack.sandbox.subprocess import _find_processes_by_marker

    groups, complete, gaps = _find_processes_by_marker(SimpleNamespace(**marker) for marker in markers)
    if not complete:
        raise RuntimeError(f"cannot inspect all Compute processes for memory safety: {gaps}")
    result = {}
    for token, processes in groups.items():
        rss = 0
        alive = False
        for process in processes:
            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                rss += process.memory_info().rss
                alive = True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        result[token] = (rss, alive)
    return result


class MemoryLease:
    """A flock slot inherited by the worker, with crash-survivor detection.

    Read-only slot locks are never unlinked or replaced. Marker metadata uses
    separate, non-inherited files so writes to an inherited fd cannot corrupt
    accounting for every worker. Retained markers also prevent reusing a slot
    when a descendant survives its controller but closes inherited fds.
    """

    def __init__(self, policy: MemoryPolicy, marker) -> None:
        self.policy = policy
        self.marker = {"token": marker.token, "created_at": marker.created_at}
        self.fd: int | None = None
        self._slot_index: int | None = None
        self.marker_repairs = 0
        self.peak_rss = 0
        self.unreadable_slots: set[int] = set()
        self._reported_quarantine: set[int] = set()

    @staticmethod
    def _open(path: Path) -> int:
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)

    def quarantine_event(self) -> dict | None:
        new_slots = self.unreadable_slots - self._reported_quarantine
        if not new_slots:
            return None
        self._reported_quarantine.update(new_slots)
        return {
            "registry": str(self.policy.registry),
            "new_unreadable_slots": sorted(new_slots),
            "unreadable_slots": sorted(self.unreadable_slots),
            "reserved_bytes": len(self.unreadable_slots) * self.policy.worker_bytes,
            "max_workers": self.policy.max_workers,
        }

    def _read_marker(self, index: int, *, locked: bool) -> dict | None:
        path = self.policy.registry / f"slot-{index}.json"
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            marker = _read_metadata(fd, path)
        finally:
            os.close(fd)
        if marker == {"unused": True} and marker["unused"] is True and not locked:
            return None
        if marker is None:
            raise MemoryRegistryError(f"empty memory slot process marker: {path}")
        token, created_at = marker.get("token"), marker.get("created_at")
        if (set(marker) != {"token", "created_at"}
                or not isinstance(token, str) or not token
                or type(created_at) not in (int, float)
                or not 0 <= created_at <= sys.float_info.max):
            raise MemoryRegistryError(f"invalid memory slot process marker: {path}")
        return marker

    def _write_marker(self, index: int, marker: dict) -> None:
        path = self.policy.registry / f"slot-{index}.json"
        fd, temporary = tempfile.mkstemp(prefix=f".slot-{index}-", dir=self.policy.registry)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(json.dumps(marker).encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _slots(self) -> tuple[list[tuple[int, int, dict | None]], list[tuple[int, int]]]:
        slots = []
        free_fds = []
        opened = []
        retained = set()
        candidates = []
        try:
            for index in range(self.policy.max_workers):
                path = self.policy.registry / f"slot-{index}.lock"
                try:
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    except FileNotFoundError:
                        # Publish the sentinel first: a crash must not leave a
                        # newly created lock with an unknowable process marker.
                        if not (self.policy.registry / f"slot-{index}.json").exists():
                            self._write_marker(index, {"unused": True})
                        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    opened.append(fd)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = False
                    except BlockingIOError:
                        locked = True
                except OSError:
                    candidates.append((index, None, None, True, True))
                    continue
                try:
                    marker = self._read_marker(index, locked=locked)
                    unreadable = False
                except (MemoryRegistryError, OSError):
                    marker, unreadable = None, True
                candidates.append((index, fd, marker, locked, unreadable))
            self.unreadable_slots = {index for index, _, _, _, unreadable in candidates if unreadable}
            usage = markers_rss([marker for _, _, marker, _, _ in candidates if marker])
            for index, fd, marker, locked, unreadable in candidates:
                if unreadable:
                    # Even an unlocked slot may have surviving descendants.
                    # Quarantine it until its owner repairs the marker or an
                    # operator reconciles the registry with all workers stopped.
                    slots.append((index, self.policy.worker_bytes, None))
                    continue
                rss, alive = usage[marker["token"]] if marker else (0, False)
                if locked or alive:
                    slots.append((index, rss, marker))
                else:
                    free_fds.append((index, fd))
            retained = {fd for _, fd in free_fds}
            return slots, free_fds
        finally:
            for fd in opened:
                if fd not in retained:
                    os.close(fd)

    def try_acquire(self) -> bool:
        self.policy.registry.mkdir(parents=True, exist_ok=True, mode=0o700)
        control = self._open(self.policy.registry / "control.json")
        free_fds: list[tuple[int, int]] = []
        try:
            try:
                fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            _check_control(self.policy, control, initialize=True)
            slots, free_fds = self._slots()
            # Available RAM already excludes resident pages. Reserve only each
            # known worker's unused allowance. Unknown usage needs the full
            # allowance, even though it is also estimated in sampled totals.
            unused = sum(
                self.policy.worker_bytes if marker is None
                else max(0, self.policy.worker_bytes - rss)
                for _, rss, marker in slots
            )
            if not free_fds or available_memory() - unused < (
                self.policy.worker_bytes + self.policy.reserve_bytes
            ):
                return False
            index, fd = free_fds.pop()
            self.fd = fd
            self._slot_index = index
            self._write_marker(index, self.marker)
            return True
        except BaseException:
            self.close()
            raise
        finally:
            for _, fd in free_fds:
                os.close(fd)
            os.close(control)

    def sample(self) -> dict:
        # Slot ownership stays locked for the invocation. Sampling serializes
        # with admission so it never reads a half-written marker.
        control = self._open(self.policy.registry / "control.json")
        free_fds: list[tuple[int, int]] = []
        try:
            fcntl.flock(control, fcntl.LOCK_EX)
            _check_control(self.policy, control)
            if self.fd is None or self._slot_index is None:
                raise MemoryRegistryError("cannot sample memory without owning a slot")
            path = self.policy.registry / f"slot-{self._slot_index}.lock"
            try:
                held, current = os.fstat(self.fd), path.stat(follow_symlinks=False)
            except OSError as exc:
                raise MemoryRegistryError(f"memory lease lock disappeared: {path}") from exc
            if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
                raise MemoryRegistryError(f"memory lease lock was replaced: {path}")
            # Ownership is the retained lock, not untrusted on-disk metadata.
            # Never repair a peer's slot: its process identity may be unknown.
            try:
                own_marker = self._read_marker(self._slot_index, locked=True)
            except (MemoryRegistryError, OSError):
                own_marker = None
            if own_marker != self.marker:
                self._write_marker(self._slot_index, self.marker)
                self.marker_repairs += 1
            slots, free_fds = self._slots()
            if self._slot_index in self.unreadable_slots:
                # A peer may reserve its allowance, but the owner needs real
                # usage to enforce its limit. Let the accounting watchdog stop it.
                raise MemoryRegistryError(f"own memory slot is unreadable: {path}")
            own_rss = next(rss for index, rss, _ in slots if index == self._slot_index)
            self.peak_rss = max(self.peak_rss, own_rss)
            available = available_memory()
            total_rss = sum(rss for _, rss, _ in slots)
            reason = None
            if own_rss > self.policy.worker_bytes:
                reason = "worker_memory_limit"
            elif available < self.policy.reserve_bytes or total_rss > (
                self.policy.worker_bytes * self.policy.max_workers
            ):
                # Stop the largest worker first, not every worker at once.
                # An unknown slot cannot act on a stop decision. Elect a
                # known worker so quarantined slots cannot disable protection.
                victim = max((slot for slot in slots if slot[2] is not None),
                             key=lambda item: (item[1], item[0]), default=None)
                if victim is not None and victim[0] == self._slot_index:
                    reason = "shared_memory_pressure"
            return {
                "rss_bytes": own_rss,
                "peak_rss_bytes": self.peak_rss,
                "compute_rss_bytes": total_rss,
                "available_bytes": available,
                "worker_limit_bytes": self.policy.worker_bytes,
                "reserve_bytes": self.policy.reserve_bytes,
                "max_workers": self.policy.max_workers,
                "unreadable_slots": [index for index, _, marker in slots if marker is None],
                "marker_repairs": self.marker_repairs,
                "reason": reason,
            }
        finally:
            for _, fd in free_fds:
                os.close(fd)
            os.close(control)

    def close(self) -> None:
        fd, self.fd = self.fd, None
        self._slot_index = None
        if fd is not None:
            # No LOCK_UN: descendants may still own this open file description.
            # Forget first: close errors must not cause a second close after
            # another thread reuses the numeric descriptor.
            os.close(fd)
