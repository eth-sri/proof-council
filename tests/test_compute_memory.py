from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import psutil

from proofstack.sandbox import memory
from proofstack.sandbox import subprocess as sandbox_module
from proofstack.sandbox.base import SandboxSpawnError, SandboxSpec, WorkerStopState
from proofstack.sandbox.memory import GiB, MemoryLease, MemoryPolicy, MemoryRegistryError


def _marker(token="test"):
    return SimpleNamespace(token=token, created_at=time.time())


def _no_marked_processes(markers):
    return {marker["token"]: (0, False) for marker in markers}


@pytest.mark.parametrize("missing", ["uids", "create_time", "environ"])
@pytest.mark.parametrize("state", ["exited", "zombie", "running"])
@pytest.mark.parametrize("batch", [False, True])
def test_memory_scan_distinguishes_exited_processes_from_live_inspection_gaps(monkeypatch, missing, state, batch):
    def denied_environment():
        raise psutil.AccessDenied(pid=424242)

    info = {"pid": 424242, "uids": SimpleNamespace(real=os.getuid()),
            "create_time": time.time()}
    if missing in info:
        info[missing] = None
    process = SimpleNamespace(
        info=info, is_running=lambda: state != "exited",
        status=lambda: psutil.STATUS_ZOMBIE if state == "zombie" else psutil.STATUS_RUNNING,
        environ=denied_environment,
    )
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [process])
    if batch:
        found, complete, _ = sandbox_module._find_processes_by_marker([_marker("a"), _marker("b")])
        assert found == {"a": [], "b": []}
    else:
        found, complete = sandbox_module._find_marked_processes(_marker())
        assert found == []
    assert complete == (state != "running")


def _hold_slot(registry, connection):
    # Test real cross-process flock ownership independently of host RAM/access.
    memory.available_memory = lambda: 128 * GiB
    memory.markers_rss = _no_marked_processes
    lease = MemoryLease(MemoryPolicy(Path(registry), 6, 8 * GiB, 16 * GiB),
                        _marker(str(os.getpid())))
    try:
        connection.send(lease.try_acquire())
        if connection.poll(30):
            connection.recv()
    finally:
        lease.close()
        connection.close()


def test_six_slots_are_shared_across_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    ctx = multiprocessing.get_context("spawn")
    owners = []
    contender = MemoryLease(MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB), _marker())
    try:
        for _ in range(6):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_hold_slot, args=(str(tmp_path), child))
            process.start()
            child.close()
            owners.append((process, parent))
            assert parent.poll(15), "slot owner did not start"
            assert parent.recv() is True
        assert contender.try_acquire() is False
        owners[0][1].send("release")
        owners[0][0].join(timeout=5)
        assert owners[0][0].exitcode == 0
        assert contender.try_acquire() is True
    finally:
        contender.close()
        for process, connection in owners:
            if process.is_alive():
                try:
                    connection.send("release")
                except (BrokenPipeError, EOFError):
                    pass
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            connection.close()


def test_surviving_descendant_keeps_slot_after_owner_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    alive = set()
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        marker["token"]: (0, marker["token"] in alive) for marker in markers
    })
    policy = MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB)
    owner = MemoryLease(policy, _marker("owner"))
    contender = MemoryLease(policy, _marker("contender"))
    try:
        assert owner.try_acquire()
        alive.add("owner")
        owner.close()
        assert not contender.try_acquire()
        alive.clear()
        assert contender.try_acquire()
    finally:
        owner.close()
        contender.close()


def test_memory_headroom_can_reduce_admission_below_six(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    available = [63 * GiB]
    monkeypatch.setattr(memory, "available_memory", lambda: available[0])
    leases = [MemoryLease(MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB), _marker(str(i)))
              for i in range(6)]
    try:
        assert all(lease.try_acquire() for lease in leases[:5])
        assert not leases[5].try_acquire()
        available[0] = 64 * GiB
        assert leases[5].try_acquire()
    finally:
        for lease in leases:
            lease.close()


def test_incompatible_shared_policy_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    owner = MemoryLease(MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB), _marker())
    other = MemoryLease(MemoryPolicy(tmp_path, 4, 8 * GiB, 16 * GiB), _marker())
    try:
        assert owner.try_acquire()
        with pytest.raises(RuntimeError, match="same memory policy"):
            other.try_acquire()
    finally:
        owner.close()
        other.close()


def test_inherited_lock_is_read_only_and_cannot_poison_concurrent_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 2, 8 * GiB, 16 * GiB)
    leases = [MemoryLease(policy, _marker(str(i))) for i in range(3)]
    try:
        assert all(lease.try_acquire() for lease in leases[:2])
        before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
        for lease in leases[:2]:
            assert fcntl.fcntl(lease.fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
            # Exercise actual exec inheritance, not just the controller's fd.
            child = subprocess.run(
                [sys.executable, "-c", r"""
import errno, os, sys
fd = int(sys.argv[1])
for write in (lambda: os.write(fd, b'\xffbinary'),
              lambda: os.pwrite(fd, b'\xffbinary', 0),
              lambda: os.ftruncate(fd, 0)):
    try:
        write()
    except OSError as exc:
        assert exc.errno in (errno.EBADF, errno.EINVAL), exc
    else:
        raise AssertionError('worker inherited a writable registry fd')
""", str(lease.fd)],
                pass_fds=(lease.fd,), capture_output=True, text=True, timeout=10,
            )
            assert child.returncode == 0, child.stderr
        assert {path: path.read_bytes() for path in tmp_path.glob("*.json")} == before
        assert all(lease.sample()["reason"] is None for lease in leases[:2])
        assert not leases[2].try_acquire()
        assert all(path.stat().st_size == 0 for path in tmp_path.glob("*.lock"))
    finally:
        for lease in leases:
            lease.close()


def test_inherited_read_only_lock_outlives_controller_handle(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB)
    owner, contender = (MemoryLease(policy, _marker(name)) for name in ("owner", "contender"))
    child = None
    try:
        assert owner.try_acquire()
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            pass_fds=(owner.fd,), stdin=subprocess.PIPE,
        )
        owner.close()
        assert not contender.try_acquire()
        child.communicate(timeout=10)
        assert child.returncode == 0
        assert contender.try_acquire()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
        owner.close()
        contender.close()


@pytest.mark.parametrize("damaged", [
    b'{"token": "\xff"}', b'{"token":', b'[]', b'{}', b'null', b'',
    b'{"token":"worker","created_at":true}',
    b'{"token":"worker","created_at":NaN}',
    b'{"token":"worker","created_at":Infinity}',
    json.dumps({"token": "worker", "created_at": 10**400}).encode(),
    b'{"token":"worker","created_at":-1}',
    b'{"token":"","created_at":1}', b' ' * 4097, None,
])
def test_corrupt_peer_is_quarantined_until_owner_repairs_it(tmp_path, monkeypatch, damaged):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 2, 8 * GiB, 16 * GiB)
    owner, peer, contender = (MemoryLease(policy, _marker(name))
                             for name in ("owner", "peer", "contender"))
    try:
        assert owner.try_acquire()
        assert peer.try_acquire()
        path = tmp_path / "slot-1.json"
        if damaged is None:
            path.unlink()
        else:
            path.write_bytes(damaged)
        sample = peer.sample()
        assert sample["unreadable_slots"] == [1]
        assert sample["compute_rss_bytes"] == policy.worker_bytes
        assert sample["reason"] is None
        assert not contender.try_acquire()
        assert contender.fd is None
        assert (path.read_bytes() if path.exists() else None) == damaged
        sample = owner.sample()
        assert sample["marker_repairs"] == 1
        assert sample["unreadable_slots"] == []
        assert json.loads(path.read_bytes()) == owner.marker
        assert peer.sample()["reason"] is None
        assert not contender.try_acquire()
        owner.close()
        assert contender.try_acquire()
    finally:
        owner.close()
        peer.close()
        contender.close()


@pytest.mark.parametrize("missing", [False, True])
def test_unlocked_damaged_slot_stays_quarantined_and_reserves_full_allowance(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 2, 8 * GiB, 16 * GiB)
    owner, peer, contender = (MemoryLease(policy, _marker(name))
                             for name in ("owner", "peer", "contender"))
    try:
        assert owner.try_acquire()
        path = tmp_path / "slot-1.json"
        if missing:
            path.unlink()
        else:
            path.write_bytes(b'\xffbinary')
        owner.close()
        monkeypatch.setattr(memory, "available_memory", lambda: 31 * GiB)
        assert not peer.try_acquire()  # 8 GiB unknown + 8 GiB new + 16 GiB reserve.
        monkeypatch.setattr(memory, "available_memory", lambda: 32 * GiB)
        assert peer.try_acquire()
        assert peer._slot_index == 0
        assert peer.sample()["unreadable_slots"] == [1]
        assert not contender.try_acquire()  # Unlocked does not mean safe to reuse.
        monkeypatch.setattr(memory, "available_memory", lambda: 15 * GiB)
        # The unknown slot must not win the election and prevent a real stop.
        assert peer.sample()["reason"] == "shared_memory_pressure"
    finally:
        owner.close()
        peer.close()
        contender.close()


@pytest.mark.parametrize("failure", ["open", "flock", "symlink"])
def test_peer_lock_failure_is_quarantined_without_breaking_sampling(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 2, GiB, 4 * GiB)
    owner, contender = (MemoryLease(policy, _marker(name)) for name in ("owner", "contender"))
    try:
        assert owner.try_acquire() and owner._slot_index == 1
        path = tmp_path / "slot-0.lock"
        lock_stat = path.stat()
        real_open, real_flock = os.open, fcntl.flock
        if failure == "symlink":
            path.unlink()
            path.symlink_to(tmp_path / "slot-1.lock")
        elif failure == "open":
            def fail_open(target, *args):
                if Path(target) == path:
                    raise PermissionError("synthetic lock permission failure")
                return real_open(target, *args)
            monkeypatch.setattr(memory.os, "open", fail_open)
        else:
            def fail_flock(fd, flags):
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
                    raise OSError("synthetic flock failure")
                return real_flock(fd, flags)
            monkeypatch.setattr(memory.fcntl, "flock", fail_flock)
        sample = owner.sample()
        assert sample["reason"] is None
        assert sample["unreadable_slots"] == [0]
        assert sample["compute_rss_bytes"] == GiB
        assert not contender.try_acquire()
        assert contender.quarantine_event()["unreadable_slots"] == [0]
    finally:
        owner.close()
        contender.close()


@pytest.fixture(params=["permission", "emfile", "flock"])
def unreadable_owner_slot(tmp_path, monkeypatch, request):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        marker["token"]: (2 * GiB if marker["token"] == "owner" else 0, True)
        for marker in markers
    })
    policy = MemoryPolicy(tmp_path, 3, GiB, 4 * GiB, poll_seconds=0.05)
    owner, peer = (MemoryLease(policy, _marker(name)) for name in ("owner", "peer"))
    fault = SimpleNamespace(active=True)
    try:
        assert owner.try_acquire() and owner._slot_index == 2
        assert peer.try_acquire() and peer._slot_index == 1
        path = tmp_path / "slot-2.lock"
        info = path.stat()
        real_open, real_flock = os.open, fcntl.flock

        def fail_open(target, *args):
            if fault.active and Path(target) == path:
                code = errno.EMFILE if request.param == "emfile" else errno.EACCES
                raise OSError(code, "synthetic owner lock open failure")
            return real_open(target, *args)

        def fail_flock(fd, flags):
            held = os.fstat(fd)
            if fault.active and (held.st_dev, held.st_ino) == (info.st_dev, info.st_ino):
                raise OSError(errno.ENOLCK, "synthetic owner lock flock failure")
            return real_flock(fd, flags)

        if request.param == "flock":
            monkeypatch.setattr(memory.fcntl, "flock", fail_flock)
        else:
            monkeypatch.setattr(memory.os, "open", fail_open)
        yield owner, peer, fault
    finally:
        owner.close()
        peer.close()


def test_own_slot_failure_is_fatal_to_accounting_not_to_healthy_peers(unreadable_owner_slot):
    owner, peer, fault = unreadable_owner_slot
    with pytest.raises(MemoryRegistryError, match="own memory slot is unreadable"):
        owner.sample()
    assert owner.unreadable_slots == {2}
    assert owner.peak_rss == 0  # Never publish the allowance as measured usage.
    os.fstat(owner.fd)
    sample = peer.sample()
    assert sample["reason"] is None
    assert sample["unreadable_slots"] == [2]
    assert sample["compute_rss_bytes"] == owner.policy.worker_bytes
    # The failed sample must release the control lock and any free slot fds.
    contender = MemoryLease(owner.policy, _marker("contender"))
    try:
        assert contender.try_acquire() and contender._slot_index == 0
    finally:
        contender.close()
    fault.active = False
    sample = owner.sample()
    assert sample["unreadable_slots"] == []
    assert sample["rss_bytes"] == 2 * GiB
    assert sample["reason"] == "worker_memory_limit"


def test_own_slot_failure_stops_worker_after_accounting_grace(unreadable_owner_slot, monkeypatch):
    owner, peer, _ = unreadable_owner_slot

    async def exercise():
        clock, samples = [100.0], []
        monkeypatch.setattr(sandbox_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        monkeypatch.setattr(sandbox_module, "MEMORY_GAP_GRACE_S", 3.0)
        stream = object.__new__(sandbox_module._StreamingProcess)
        stream._process_group_stop_state = WorkerStopState.SURVIVING
        stream._memory_lease = owner
        stream._emit_memory = AsyncMock()
        stream.memory_failure = None
        real_sample = owner.sample

        def sample():
            clock[0] += 1
            samples.append(clock[0])
            return real_sample()

        async def terminate():
            assert clock[0] - samples[0] == 3.0
            stream._process_group_stop_state = WorkerStopState.STOPPED

        monkeypatch.setattr(owner, "sample", sample)
        stream.terminate = AsyncMock(side_effect=terminate)
        await asyncio.wait_for(stream._watch_memory(), 5)
        stream.terminate.assert_awaited_once()
        assert len(samples) == 4
        assert stream.memory_failure == "memory_accounting_unavailable"
        events = stream._emit_memory.await_args_list
        assert [event.args[0] for event in events] == [
            "cli.memory_accounting_gap", "cli.memory_limit_exceeded",
        ]
        assert all(event.args[1]["error_type"] == "MemoryRegistryError" for event in events)
        assert "own memory slot is unreadable" in events[-1].args[1]["error"]
        assert peer.sample()["reason"] is None

    asyncio.run(exercise())


@pytest.mark.parametrize("replacement", ["missing", "new_owner", "symlink"])
def test_lost_lease_inode_fails_before_repairing_any_marker(tmp_path, monkeypatch, replacement):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 1, GiB, 4 * GiB)
    owner, contender = (MemoryLease(policy, _marker(name)) for name in ("owner", "contender"))
    try:
        assert owner.try_acquire()
        path = tmp_path / "slot-0.lock"
        path.unlink()
        if replacement == "new_owner":
            assert contender.try_acquire()
        elif replacement == "symlink":
            path.symlink_to(tmp_path / "slot-0.json")
        marker = tmp_path / "slot-0.json"
        before = marker.read_bytes()
        with pytest.raises(MemoryRegistryError, match="memory lease lock"):
            owner.sample()
        assert marker.read_bytes() == before
        assert owner.marker_repairs == 0
        if replacement == "new_owner":
            assert contender.sample()["reason"] is None
    finally:
        owner.close()
        contender.close()


def test_crash_before_first_slot_lock_leaves_reusable_unused_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 1, GiB, 4 * GiB)
    lease = MemoryLease(policy, _marker())
    real_open = os.open
    path = tmp_path / "slot-0.lock"

    def crash_before_lock(target, flags, *args):
        if Path(target) == path and flags & os.O_CREAT:
            assert json.loads((tmp_path / "slot-0.json").read_bytes()) == {"unused": True}
            raise KeyboardInterrupt("synthetic crash before lock creation")
        return real_open(target, flags, *args)

    with monkeypatch.context() as patch:
        patch.setattr(memory.os, "open", crash_before_lock)
        with pytest.raises(KeyboardInterrupt):
            lease.try_acquire()
    assert lease.fd is None and not path.exists()
    try:
        assert lease.try_acquire()
        assert lease.sample()["unreadable_slots"] == []
    finally:
        lease.close()


def test_repaired_owner_still_obeys_its_memory_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        marker["token"]: (9 * GiB, True) for marker in markers
    })
    owner = MemoryLease(MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB), _marker())
    try:
        assert owner.try_acquire()
        (tmp_path / "slot-0.json").write_bytes(b'\xffbinary')
        sample = owner.sample()
        assert sample["marker_repairs"] == 1
        assert sample["rss_bytes"] == 9 * GiB
        assert sample["reason"] == "worker_memory_limit"
    finally:
        owner.close()


def test_memory_lease_does_not_close_reused_descriptor_after_close_error(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    owner = MemoryLease(MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB), _marker())
    assert owner.try_acquire()
    real_close = os.close
    replacement = []

    def close_then_reuse(target):
        real_close(target)
        opened = os.open(tmp_path / "replacement", os.O_CREAT | os.O_RDWR, 0o600)
        if opened != target:
            os.dup2(opened, target)
            real_close(opened)
        replacement.append(target)
        raise OSError("synthetic close error after reuse")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(memory.os, "close", close_then_reuse)
            with pytest.raises(OSError, match="after reuse"):
                owner.close()
            assert owner.fd is None
            owner.close()
        os.fstat(replacement[0])
    finally:
        for target in replacement:
            real_close(target)


@pytest.mark.parametrize("saved", [None, {}, {"version": 1}])
def test_legacy_registry_is_rejected_without_changing_metadata(tmp_path, monkeypatch, saved):
    monkeypatch.setattr(memory, "markers_rss", lambda markers: pytest.fail("must reject before scanning"))
    control = tmp_path / "control.json"
    expected = {"max_workers": 1, "worker_bytes": 8 * GiB, "reserve_bytes": 16 * GiB}
    control.write_text("" if saved is None else json.dumps(expected | saved))
    slot = tmp_path / "slot-0.json"
    slot.write_text(json.dumps({"token": "old-worker", "created_at": 1}))
    before = {path: path.read_bytes() for path in (slot, control)}
    lease = MemoryLease(MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB), _marker())
    with pytest.raises(MemoryRegistryError, match="registry version"):
        lease.try_acquire()
    assert lease.fd is None
    assert {path: path.read_bytes() for path in (slot, control)} == before
    assert not list(tmp_path.glob("*.lock"))


@pytest.mark.parametrize("existing_directory", [False, True])
def test_registry_preflight_leaves_fresh_registry_untouched(tmp_path, existing_directory):
    registry = tmp_path / "registry"
    if existing_directory:
        registry.mkdir()
    memory.check_memory_registry(MemoryPolicy(registry, 2, GiB, 4 * GiB))
    assert registry.exists() == existing_directory
    assert not list(registry.glob("*"))


@pytest.mark.parametrize("state", ["valid", "legacy", "corrupt", "missing", "empty", "policy"])
def test_registry_preflight_is_read_only_and_fail_closed(tmp_path, monkeypatch, state):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    monkeypatch.setattr(memory, "time", SimpleNamespace(
        monotonic=time.monotonic, sleep=lambda *_: pytest.fail("uncontended validation must not retry")))
    policy = MemoryPolicy(tmp_path, 2, GiB, 4 * GiB)
    owner = MemoryLease(policy, _marker())
    try:
        assert owner.try_acquire()
        control = tmp_path / "control.json"
        if state in {"legacy", "policy"}:
            saved = json.loads(control.read_bytes())
            if state == "legacy":
                saved.pop("version")
            else:
                saved["max_workers"] += 1
            control.write_text(json.dumps(saved))
        elif state == "corrupt":
            control.write_bytes(b"\xffbinary")
        elif state == "missing":
            control.unlink()
        elif state == "empty":
            control.write_bytes(b"")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        monkeypatch.setattr(memory, "markers_rss", lambda *_: pytest.fail("preflight must not scan or admit workers"))
        real_open = os.open
        opened = []

        def read_only_open(path, flags, *args):
            assert flags & os.O_ACCMODE == os.O_RDONLY
            assert not flags & os.O_CREAT
            fd = real_open(path, flags, *args)
            opened.append(fd)
            return fd

        with monkeypatch.context() as patch:
            patch.setattr(memory.os, "open", read_only_open)
            if state == "valid":
                memory.check_memory_registry(policy)
            else:
                with pytest.raises(MemoryRegistryError, match="control.json"):
                    memory.check_memory_registry(policy)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
        for fd in opened:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        owner.close()


@pytest.mark.parametrize("release_after_deadline", [False, True])
def test_registry_preflight_bounds_lock_wait_and_closes_fd(tmp_path, monkeypatch, release_after_deadline):
    control = os.open(tmp_path / "control.json", os.O_CREAT | os.O_RDWR, 0o600)
    clock, sleeps, attempts = [0.0], [], []
    real_flock = fcntl.flock

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay
        if release_after_deadline:
            real_flock(control, fcntl.LOCK_UN)
            clock[0] += memory.REGISTRY_PREFLIGHT_WAIT_S

    def flock(fd, flags):
        assert flags == fcntl.LOCK_EX | fcntl.LOCK_NB
        attempts.append((clock[0], fd))
        return real_flock(fd, flags)

    try:
        real_flock(control, fcntl.LOCK_EX)
        monkeypatch.setattr(memory, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
        monkeypatch.setattr(memory.fcntl, "flock", flock)
        with pytest.raises(MemoryRegistryError, match="busy after 3s; retry launch preflight"):
            memory.check_memory_registry(MemoryPolicy(tmp_path, 1, GiB, 4 * GiB))
        assert all(0 < delay <= memory.REGISTRY_PREFLIGHT_POLL_S for delay in sleeps)
        assert all(at < memory.REGISTRY_PREFLIGHT_WAIT_S for at, _ in attempts)
        if release_after_deadline:
            assert len(attempts) == 1
        else:
            assert sum(sleeps) == pytest.approx(memory.REGISTRY_PREFLIGHT_WAIT_S)
        assert len({fd for _, fd in attempts}) == 1
        with pytest.raises(OSError):
            os.fstat(attempts[0][1])
        assert (tmp_path / "control.json").read_bytes() == b""
    finally:
        os.close(control)


def _hold_control_lock(path, connection):
    control = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(control, fcntl.LOCK_EX)
        connection.send("locked")
        if connection.poll(30):
            connection.recv()
    finally:
        os.close(control)
    connection.send("released")
    connection.close()


def test_registry_preflight_retries_until_another_process_releases_control(tmp_path, monkeypatch):
    path = tmp_path / "control.json"
    before = json.dumps({"version": memory.REGISTRY_VERSION, "max_workers": 1,
                         "worker_bytes": GiB, "reserve_bytes": 4 * GiB})
    path.write_text(before)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_hold_control_lock, args=(str(path), child))
    process.start()
    child.close()
    clock, sleeps = [0.0], []

    def release_on_retry(delay):
        sleeps.append(delay)
        clock[0] += delay
        parent.send("release")
        assert parent.poll(5), "control owner did not release"
        assert parent.recv() == "released"

    try:
        assert parent.poll(15), "control owner did not start"
        assert parent.recv() == "locked"
        monkeypatch.setattr(memory, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=release_on_retry))
        memory.check_memory_registry(MemoryPolicy(tmp_path, 1, GiB, 4 * GiB))
        assert sleeps == [memory.REGISTRY_PREFLIGHT_POLL_S]
        assert path.read_text() == before
        assert list(tmp_path.iterdir()) == [path]
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()


def test_registry_preflight_does_not_retry_other_lock_errors(tmp_path, monkeypatch):
    path = tmp_path / "control.json"
    path.write_text("{}")
    attempts = []

    def fail_lock(fd, flags):
        attempts.append(fd)
        raise PermissionError("synthetic lock permission error")

    monkeypatch.setattr(memory.fcntl, "flock", fail_lock)
    monkeypatch.setattr(memory, "time", SimpleNamespace(
        monotonic=time.monotonic, sleep=lambda *_: pytest.fail("only contention should be retried")))
    with pytest.raises(PermissionError, match="synthetic lock permission error"):
        memory.check_memory_registry(MemoryPolicy(tmp_path, 1, GiB, 4 * GiB))
    assert len(attempts) == 1
    with pytest.raises(OSError):
        os.fstat(attempts[0])
    assert path.read_text() == "{}"


@pytest.mark.parametrize("damaged", [b'\xff', b'[]', b'', b' ' * 4097])
def test_control_corruption_is_not_reinitialized(tmp_path, monkeypatch, damaged):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 2, 8 * GiB, 16 * GiB)
    owner, contender = (MemoryLease(policy, _marker(name)) for name in ("owner", "contender"))
    try:
        assert owner.try_acquire()
        control = tmp_path / "control.json"
        control.write_bytes(damaged)
        for operation in (owner.sample, contender.try_acquire):
            with pytest.raises(MemoryRegistryError, match=r"control\.json"):
                operation()
        assert control.read_bytes() == damaged
        assert contender.fd is None
    finally:
        owner.close()
        contender.close()


def test_marker_replacement_failure_preserves_old_marker_and_releases_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 1, 8 * GiB, 16 * GiB)
    old = MemoryLease(policy, _marker("old"))
    assert old.try_acquire()
    old.close()
    path = tmp_path / "slot-0.json"
    before = path.read_bytes()
    contender = MemoryLease(policy, _marker("new"))
    replace = os.replace

    def fail_replace(source, destination):
        assert path.read_bytes() == before
        assert json.loads(Path(source).read_bytes()) == contender.marker
        raise OSError("synthetic interrupted publication")

    monkeypatch.setattr(memory.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        contender.try_acquire()
    assert contender.fd is None
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".slot-*"))
    monkeypatch.setattr(memory.os, "replace", replace)
    try:
        assert contender.try_acquire()
        assert json.loads(path.read_bytes()) == contender.marker
    finally:
        contender.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_queued_worker_can_timeout_or_cancel_without_spawning(tmp_path, monkeypatch, cancel):
    async def exercise():
        lease = SimpleNamespace(try_acquire=lambda: False, close=lambda: closed.append(True),
                                quarantine_event=lambda: None)
        closed = []
        monkeypatch.setattr(sandbox_module, "MemoryLease", lambda *args: lease)
        spawn = AsyncMock()
        monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", spawn)
        queued = asyncio.Event()

        async def emit(kind, payload):
            if kind == "cli.memory_queued":
                queued.set()

        sandbox = sandbox_module.SubprocessSandbox(
            SandboxSpec(backend="subprocess", memory_policy=MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB)),
            root=tmp_path / "workspace",
        )
        sandbox.emit_memory_event = emit
        task = asyncio.create_task(sandbox.stream_command(["never-spawn"], timeout_s=0.1))
        await asyncio.wait_for(queued.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(SandboxSpawnError, match="shared Compute slot"):
                await asyncio.wait_for(task, 2)
        assert closed == [True]
        spawn.assert_not_called()
        assert sandbox.worker_stop_state is WorkerStopState.STOPPED
        assert sandbox.worker_launch_settled

    asyncio.run(exercise())


def test_quarantined_pool_logs_before_queueing_without_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 1, GiB, 4 * GiB)
    owner = MemoryLease(policy, _marker())
    assert owner.try_acquire()
    owner.close()
    (tmp_path / "slot-0.json").write_bytes(b"\xffbinary")

    async def exercise():
        spawn = AsyncMock()
        monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", spawn)
        sandbox = sandbox_module.SubprocessSandbox(
            SandboxSpec(backend="subprocess", memory_policy=policy), root=tmp_path / "workspace")
        queued = asyncio.Event()
        events = []

        async def emit(kind, payload):
            events.append((kind, payload))
            if kind == "cli.memory_queued":
                queued.set()

        sandbox.emit_memory_event = emit
        task = asyncio.create_task(sandbox.stream_command(["never-spawn"], timeout_s=30))
        try:
            await asyncio.wait_for(queued.wait(), 3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        spawn.assert_not_called()
        assert [kind for kind, _ in events] == ["cli.memory_registry_quarantine", "cli.memory_queued"]
        assert events[0][1] == {
            "registry": str(tmp_path), "new_unreadable_slots": [0], "unreadable_slots": [0],
            "reserved_bytes": GiB, "max_workers": 1,
        }

    asyncio.run(exercise())


def test_quarantine_event_reports_each_new_slot_once_per_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    policy = MemoryPolicy(tmp_path, 3, GiB, 4 * GiB)
    owner = MemoryLease(policy, _marker())
    try:
        assert owner.try_acquire() and owner._slot_index == 2
        assert owner.quarantine_event() is None
        for index in (0, 1):
            (tmp_path / f"slot-{index}.json").write_bytes(b"\xffbinary")
            owner.sample()
            event = owner.quarantine_event()
            assert event["new_unreadable_slots"] == [index]
            assert event["unreadable_slots"] == list(range(index + 1))
            assert event["reserved_bytes"] == (index + 1) * GiB
            owner.sample()
            assert owner.quarantine_event() is None
    finally:
        owner.close()


def test_running_worker_emits_quarantine_once_without_termination(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    monkeypatch.setattr(memory, "markers_rss", _no_marked_processes)
    owner = MemoryLease(MemoryPolicy(tmp_path, 2, GiB, 4 * GiB, poll_seconds=0.05), _marker())
    assert owner.try_acquire() and owner._slot_index == 1
    (tmp_path / "slot-0.json").write_bytes(b"\xffbinary")

    async def exercise():
        stream = object.__new__(sandbox_module._StreamingProcess)
        stream._process_group_stop_state = WorkerStopState.SURVIVING
        stream._memory_lease = owner
        stream._emit_memory = AsyncMock()
        stream.terminate = AsyncMock()
        sample = owner.sample
        samples = []

        def bounded_sample():
            samples.append(sample())
            if len(samples) == 3:
                stream._process_group_stop_state = WorkerStopState.STOPPED
            return samples[-1]

        monkeypatch.setattr(owner, "sample", bounded_sample)
        await asyncio.wait_for(stream._watch_memory(), 3)
        stream.terminate.assert_not_called()
        events = [call for call in stream._emit_memory.await_args_list
                  if call.args[0] == "cli.memory_registry_quarantine"]
        assert len(events) == 1
        assert events[0].args[1]["unreadable_slots"] == [0]
        assert len(samples) == 3

    try:
        asyncio.run(exercise())
    finally:
        owner.close()


def test_sample_scans_process_table_once_for_all_six_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "available_memory", lambda: 128 * GiB)
    policy = MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB)
    leases = [MemoryLease(policy, _marker(str(i))) for i in range(6)]
    scans = []
    processes = []
    try:
        with monkeypatch.context() as setup:
            setup.setattr(memory, "markers_rss", _no_marked_processes)
            assert all(lease.try_acquire() for lease in leases)
        for i, lease in enumerate(leases):
            processes.append(SimpleNamespace(
                info={"pid": 900000 + i, "uids": SimpleNamespace(real=os.getuid()),
                      "create_time": lease.marker["created_at"]},
                environ=lambda token=lease.marker["token"]: {sandbox_module.PROCESS_MARKER_ENV: token},
                status=lambda: psutil.STATUS_RUNNING,
                memory_info=lambda size=i + 1: SimpleNamespace(rss=size),
            ))
        # A second process in one invocation must be included in its aggregate.
        processes.append(processes[0])

        def process_iter(**kwargs):
            scans.append(True)
            return iter(processes)

        monkeypatch.setattr(sandbox_module.psutil, "process_iter", process_iter)
        sample = leases[0].sample()
        assert scans == [True]
        assert sample["rss_bytes"] == 2
        assert sample["compute_rss_bytes"] == 22
        assert sample["reason"] is None
        assert policy.poll_seconds == 1.0
    finally:
        for lease in leases:
            lease.close()


def test_batched_scan_respects_each_markers_creation_time(monkeypatch):
    old = SimpleNamespace(token="old", created_at=100)
    new = SimpleNamespace(token="new", created_at=200)
    candidate = SimpleNamespace(
        info={"pid": 900000, "uids": SimpleNamespace(real=os.getuid()), "create_time": 150},
        environ=lambda: {sandbox_module.PROCESS_MARKER_ENV: "new"},
    )
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [candidate])
    groups, complete, _ = sandbox_module._find_processes_by_marker([old, new])
    assert complete
    assert groups == {"old": [], "new": []}


@pytest.mark.parametrize("wallclock_deadline,queue_timeout,expected_deadline", [
    (None, None, 12701.0),
    (4100.0, None, 4100.0),
    (3000.0, None, None),
    (None, 1000.0, None),
])
def test_queue_and_execution_have_separate_deadlines(
    tmp_path, monkeypatch, wallclock_deadline, queue_timeout, expected_deadline,
):
    async def exercise():
        clock = [100.0]
        closed = []

        def acquire():
            clock[0] += 3601
            return True

        lease = SimpleNamespace(try_acquire=acquire, close=lambda: closed.append(True), fd=123)
        monkeypatch.setattr(sandbox_module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
        monkeypatch.setattr(sandbox_module, "MemoryLease", lambda *args: lease)
        monkeypatch.setattr(sandbox_module, "_StreamingProcess", lambda **kwargs: SimpleNamespace(**kwargs))
        spawn = AsyncMock(return_value=SimpleNamespace())
        monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", spawn)
        sandbox = sandbox_module.SubprocessSandbox(
            SandboxSpec(backend="subprocess", memory_policy=MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB)),
            root=tmp_path / "workspace",
        )
        call = sandbox.stream_command(
            ["probe"], timeout_s=9000, queue_timeout_s=queue_timeout,
            wallclock_deadline=wallclock_deadline,
        )
        if expected_deadline is None:
            with pytest.raises(SandboxSpawnError):
                await call
            spawn.assert_not_called()
            assert closed == [True]
        else:
            stream = await call
            spawn.assert_awaited_once()
            assert stream.deadline == expected_deadline
            assert closed == []
            lease.close()

    asyncio.run(exercise())


def test_cancelled_admission_settles_thread_before_releasing_slot(tmp_path, monkeypatch):
    async def exercise():
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        closed = []

        def acquire():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return True

        lease = SimpleNamespace(try_acquire=acquire, close=lambda: closed.append(True))
        monkeypatch.setattr(sandbox_module, "MemoryLease", lambda *args: lease)
        spawn = AsyncMock()
        monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", spawn)
        sandbox = sandbox_module.SubprocessSandbox(
            SandboxSpec(backend="subprocess", memory_policy=MemoryPolicy(tmp_path, 6, 8 * GiB, 16 * GiB)),
            root=tmp_path / "workspace",
        )
        task = asyncio.create_task(sandbox.stream_command(["never-spawn"]))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert closed == []
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert closed == [True]
        spawn.assert_not_called()

    asyncio.run(exercise())


def test_memory_sampling_does_not_block_event_loop():
    async def exercise():
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        sampling_threads = []
        stream = object.__new__(sandbox_module._StreamingProcess)
        stream._process_group_stop_state = WorkerStopState.SURVIVING
        stream._emit_memory = AsyncMock()

        def sample():
            sampling_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return {"reason": "worker_memory_limit"}

        stream._memory_lease = SimpleNamespace(sample=sample)
        stream.terminate = AsyncMock()
        task = asyncio.create_task(stream._watch_memory())
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert len(sampling_threads) == 1
            assert sampling_threads[0] != threading.get_ident()
            stream.terminate.assert_not_called()
        finally:
            release.set()
            await asyncio.wait_for(task, 2)
        assert stream.memory_failure == "worker_memory_limit"
        stream.terminate.assert_awaited_once()
        stream._emit_memory.assert_awaited_once_with(
            "cli.memory_limit_exceeded", {"reason": "worker_memory_limit"},
        )

    asyncio.run(exercise())


def _chain(*procs):
    """Build a parent chain from (pid, uid) pairs, nearest parent first."""
    nodes = [SimpleNamespace(pid=pid, uids=lambda uid=uid: SimpleNamespace(real=uid),
                             environ=lambda: {}, create_time=time.time) for pid, uid in procs]
    for i, node in enumerate(nodes):
        upper = nodes[i + 1] if i + 1 < len(nodes) else None
        node.parent = lambda upper=upper: upper
        node.ppid = lambda upper=upper: upper.pid if upper else 0
    return nodes[0] if nodes else None


ME, ROOT = os.getuid(), 0


@pytest.mark.parametrize("parent, complete", [
    # codex-linux-sandbox or bwrap under another session's shell: a bystander.
    (_chain((5000, ME), (4000, ME), (1357, ME), (1, ROOT)), True),
    # sshd-session for a fresh login: adopted by root's sshd.
    (_chain((4242, ROOT), (1, ROOT)), True),
    # orphan reparented to pid 1, or to systemd --user (a same-uid subreaper): may be ours.
    (_chain((1, ROOT)), False),
    (_chain((1357, ME), (1, ROOT)), False),
    # our own descendant that entered a sandbox stays a gap.
    (_chain((6000, ME), (os.getpid(), ME), (1357, ME), (1, ROOT)), False),
    # docker exec: a tree not rooted in our view was injected from outside.
    (None, True),
])
def test_unreadable_process_is_a_gap_only_when_it_may_descend_from_us(monkeypatch, parent, complete):
    def denied_environment():
        raise psutil.AccessDenied(pid=424242)

    process = SimpleNamespace(
        info={"pid": 424242, "uids": SimpleNamespace(real=ME), "create_time": time.time()},
        pid=424242, is_running=lambda: True, status=lambda: psutil.STATUS_RUNNING,
        environ=denied_environment, parent=lambda: parent,
    )
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [process])
    found, got, _ = sandbox_module._find_processes_by_marker([_marker()])
    assert found == {"test": []}
    assert got == complete


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("ancestor_first", [False, True])
def test_unreadable_sibling_worker_child_respects_all_requested_markers(monkeypatch, registered, ancestor_first):
    parent = _chain((5000, ME), (4000, ME), (1357, ME), (1, ROOT))
    parent.environ = lambda: {sandbox_module.PROCESS_MARKER_ENV: "sibling"}
    parent.info = {"pid": parent.pid, "uids": SimpleNamespace(real=ME), "create_time": time.time()}
    parent.status = lambda: psutil.STATUS_RUNNING
    parent.memory_info = lambda: SimpleNamespace(rss=123)
    child = _unreadable_live_process()
    child.parent = lambda: parent
    child.ppid = lambda: parent.pid
    processes = [parent, child] if ancestor_first else [child, parent]
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: processes)
    markers = [{"token": "own", "created_at": time.time()}]
    if registered:
        markers.append({"token": "sibling", "created_at": time.time()})
        with pytest.raises(RuntimeError, match=str(child.pid)):
            memory.markers_rss(markers)
    else:
        assert memory.markers_rss(markers) == {"own": (0, False)}


def test_unreadable_ancestor_cannot_prove_sibling_is_unrelated(monkeypatch):
    parent = _chain((5000, ME), (4000, ME), (1, ROOT))
    child = _unreadable_live_process()
    child.parent = lambda: parent
    parent.environ = child.environ
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [child])
    with pytest.raises(RuntimeError, match=str(child.pid)):
        memory.markers_rss([{"token": "sibling", "created_at": time.time()}])


def _unreadable_live_process(pid=434343):
    def denied_environment():
        raise psutil.AccessDenied(pid=pid)

    return SimpleNamespace(
        info={"pid": pid, "uids": SimpleNamespace(real=ME), "create_time": time.time()},
        pid=pid, is_running=lambda: True, status=lambda: psutil.STATUS_RUNNING,
        name=lambda: "mystery", ppid=lambda: 1, environ=denied_environment,
        parent=lambda: _chain((1, ROOT)),
    )


def test_incomplete_scan_names_the_gap(monkeypatch):
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [_unreadable_live_process()])
    found, complete, gaps = sandbox_module._find_processes_by_marker([_marker()])
    assert not complete
    assert len(gaps) == 1
    assert gaps[0]["pid"] == 434343
    assert gaps[0]["name"] == "mystery"
    assert gaps[0]["ppid"] == 1
    assert gaps[0]["status"] == psutil.STATUS_RUNNING
    assert gaps[0]["error"].startswith("AccessDenied")


def test_markers_rss_error_names_the_gap(monkeypatch):
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [_unreadable_live_process()])
    with pytest.raises(RuntimeError, match="434343"):
        memory.markers_rss([{"token": "test", "created_at": time.time()}])


def test_candidate_exiting_during_parent_walk_is_not_a_gap(monkeypatch):
    process = _unreadable_live_process()
    running = [True]

    def vanished_parent():
        running[0] = False
        raise psutil.NoSuchProcess(pid=434343)

    process.parent = vanished_parent
    process.is_running = lambda: running[0]
    monkeypatch.setattr(sandbox_module.psutil, "process_iter", lambda **kwargs: [process])
    _, complete, gaps = sandbox_module._find_processes_by_marker([_marker()])
    assert complete and gaps == []


def _gap_watchdog(monkeypatch, samples):
    monkeypatch.setattr(sandbox_module, "MEMORY_GAP_GRACE_S", 0.05)
    stream = object.__new__(sandbox_module._StreamingProcess)
    stream._process_group_stop_state = WorkerStopState.SURVIVING
    stream._emit_memory = AsyncMock()
    stream.memory_failure = None

    def sample():
        result = samples.pop(0) if samples else {"reason": None}
        if isinstance(result, Exception):
            raise result
        return result

    stream._memory_lease = SimpleNamespace(sample=sample, policy=SimpleNamespace(poll_seconds=0.01))
    stream.terminate = AsyncMock()
    return stream


def test_transient_accounting_gap_does_not_terminate(monkeypatch):
    async def exercise():
        samples = [RuntimeError("gap pid 434343"), {"reason": None}]
        stream = _gap_watchdog(monkeypatch, samples)
        task = asyncio.create_task(stream._watch_memory())
        await asyncio.sleep(0.2)
        stream._process_group_stop_state = WorkerStopState.STOPPED
        await asyncio.wait_for(task, 2)
        stream.terminate.assert_not_called()
        assert stream.memory_failure is None
        kinds = [call.args[0] for call in stream._emit_memory.await_args_list]
        assert kinds[0] == "cli.memory_accounting_gap"
        assert "cli.memory_limit_exceeded" not in kinds

    asyncio.run(exercise())


def test_persistent_accounting_gap_terminates_with_error(monkeypatch):
    async def exercise():
        samples = [RuntimeError("gap pid 434343")] * 1000
        stream = _gap_watchdog(monkeypatch, samples)
        await asyncio.wait_for(stream._watch_memory(), 2)
        assert stream.memory_failure == "memory_accounting_unavailable"
        stream.terminate.assert_awaited_once()
        kinds = [call.args[0] for call in stream._emit_memory.await_args_list]
        assert kinds.count("cli.memory_accounting_gap") == 1
        assert kinds[-1] == "cli.memory_limit_exceeded"
        payload = stream._emit_memory.await_args_list[-1].args[1]
        assert payload["reason"] == "memory_accounting_unavailable"
        assert "434343" in payload["error"]

    asyncio.run(exercise())
