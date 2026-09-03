from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.storage_reservation import (
    StorageReservationError,
    StorageReservationLease,
)
from proofstack.kinds.cli import _release_storage_lease


def _lease(
    root: Path,
    *,
    owner: str,
    requested_bytes: int,
    retention_guard_path: Path | None = None,
    retention_guard_repair=None,
) -> StorageReservationLease:
    return StorageReservationLease(
        registry_dir=root / "leases",
        workspace_root=root / "workspace",
        requested_bytes=requested_bytes,
        minimum_free_bytes=50,
        owner=owner,
        retention_guard_path=retention_guard_path,
        retention_guard_repair=retention_guard_repair,
    )


def test_concurrent_leases_are_counted_and_released() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        first = _lease(root, owner="first", requested_bytes=100)
        denied = _lease(root, owner="denied", requested_bytes=101)

        first_status = first.acquire()
        assert first_status.active_reserved_bytes == 0
        assert first.active
        with pytest.raises(StorageReservationError, match="251 bytes are required"):
            denied.acquire()

        first.release()
        assert not first.active
        admitted_status = denied.acquire()
        assert admitted_status.active_reserved_bytes == 0
        denied.release()
        assert list((root / "leases").glob("lease-*.json")) == []


def test_guarded_lease_remains_reserved_after_unconfirmed_stop() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        guard = root / "recovery" / "worker.active.json"
        guard.parent.mkdir()
        guard.write_text("{}", encoding="utf-8")
        first = _lease(
            root,
            owner="possibly-running",
            requested_bytes=100,
            retention_guard_path=guard,
        )
        denied = _lease(root, owner="second", requested_bytes=101)

        first.acquire()
        lease_path = next((root / "leases").glob("lease-*.json"))
        first.retain()
        assert not first.active
        assert lease_path.exists()
        with pytest.raises(StorageReservationError, match="251 bytes are required"):
            denied.acquire()

        guard.unlink()
        admitted = denied.acquire()
        assert admitted.active_reserved_bytes == 0
        assert not lease_path.exists()
        denied.release()


def test_guarded_lease_remains_reserved_after_owner_crash() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        guard = root / "recovery" / "worker.active.json"
        guard.parent.mkdir()
        guard.write_text("{}", encoding="utf-8")
        first = _lease(
            root,
            owner="crashed",
            requested_bytes=100,
            retention_guard_path=guard,
        )
        denied = _lease(root, owner="second", requested_bytes=101)

        first.acquire()
        lease_path = next((root / "leases").glob("lease-*.json"))
        lease_file = first._lease_file
        assert lease_file is not None
        first._lease_file = None
        first._lease_path = None
        lease_file.close()

        with pytest.raises(StorageReservationError, match="251 bytes are required"):
            denied.acquire()
        assert lease_path.exists()

        guard.unlink()
        admitted = denied.acquire()
        assert admitted.active_reserved_bytes == 0
        assert not lease_path.exists()
        denied.release()


def test_retain_without_recovery_guard_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        lease = _lease(root, owner="unguarded", requested_bytes=100)
        lease.acquire()

        with pytest.raises(
            StorageReservationError,
            match="requires a recovery guard",
        ):
            lease.retain()

        assert lease.active
        lease.release()
        assert list((root / "leases").glob("lease-*.json")) == []


def test_missing_retention_guard_is_repaired_before_unlock() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        guard = root / "recovery" / "worker.active.json"

        def repair() -> None:
            guard.parent.mkdir(parents=True, exist_ok=True)
            guard.write_text("{}", encoding="utf-8")

        first = _lease(
            root,
            owner="possibly-running",
            requested_bytes=100,
            retention_guard_path=guard,
            retention_guard_repair=repair,
        )
        denied = _lease(root, owner="second", requested_bytes=101)

        first.acquire()
        assert first.retain() == "repaired"
        assert guard.exists()
        with pytest.raises(StorageReservationError, match="251 bytes are required"):
            denied.acquire()

        guard.unlink()
        admitted = denied.acquire()
        assert admitted.active_reserved_bytes == 0
        denied.release()


def test_missing_retention_guard_without_repair_keeps_lease_locked() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        guard = root / "missing.active.json"
        lease = _lease(
            root,
            owner="possibly-running",
            requested_bytes=100,
            retention_guard_path=guard,
        )
        lease.acquire()

        with pytest.raises(StorageReservationError, match="guard disappeared"):
            lease.retain()

        assert lease.active
        lease.release()


def test_unreadable_retention_guard_does_not_become_permanent() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=250),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        guard = root / "worker.active.json"
        guard.write_text("{}", encoding="utf-8")
        lease = _lease(
            root,
            owner="possibly-running",
            requested_bytes=100,
            retention_guard_path=guard,
        )
        lease.acquire()
        original_lstat = Path.lstat
        resolved_guard = guard.resolve()

        def unreadable(path: Path):
            if path == resolved_guard:
                raise OSError("transient metadata failure")
            return original_lstat(path)

        with patch.object(Path, "lstat", new=unreadable):
            assert lease.retain() == "unreadable"

        lease_path = next((root / "leases").glob("lease-*.json"))
        payload = json.loads(lease_path.read_text(encoding="utf-8"))
        assert "retained" not in payload
        guard.unlink()

        admitted = _lease(root, owner="second", requested_bytes=100)
        admitted.acquire()
        assert not lease_path.exists()
        admitted.release()


def test_legacy_permanent_fallback_is_pruned_when_guard_is_gone() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=200),
    ), patch(
        "proofstack.storage_reservation._workspace_allocated_bytes",
        return_value=0,
    ):
        root = Path(temp_dir)
        registry = root / "leases"
        registry.mkdir()
        stale = registry / "lease-retained.json"
        stale.write_text(
            json.dumps(
                {
                    "version": 3,
                    "filesystem_device": (root / "workspace").parent.stat().st_dev,
                    "reserved_bytes": 10_000,
                    "workspace": str(root / "workspace"),
                    "retention_guard": str(root / "missing.active.json"),
                    "retained": True,
                }
            ),
            encoding="utf-8",
        )

        lease = _lease(root, owner="current", requested_bytes=100)
        lease.acquire()
        assert not stale.exists()
        lease.release()


def test_reservations_count_only_each_workspaces_remaining_allowance() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        usage = {"first": 60, "second": 20}

        def allocated(path: Path) -> int:
            return usage[Path(path).name]

        first = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "first",
            requested_bytes=100,
            minimum_free_bytes=50,
            owner="first",
        )
        second = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "second",
            requested_bytes=100,
            minimum_free_bytes=50,
            owner="second",
        )
        with patch(
            "proofstack.storage_reservation.shutil.disk_usage",
            return_value=SimpleNamespace(free=200),
        ), patch(
            "proofstack.storage_reservation._workspace_allocated_bytes",
            side_effect=allocated,
        ):
            first_status = first.acquire()
            second_status = second.acquire()

        assert first_status.workspace_allocated_bytes == 60
        assert first_status.remaining_reserved_bytes == 40
        assert second_status.active_reserved_bytes == 40
        assert second_status.workspace_allocated_bytes == 20
        assert second_status.remaining_reserved_bytes == 80
        assert second_status.required_free_bytes == 170
        second.release()
        first.release()


def test_active_reservation_shrinks_as_workspace_consumes_capacity() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        usage = {"first": 0, "second": 0}

        def allocated(path: Path) -> int:
            return usage[Path(path).name]

        first = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "first",
            requested_bytes=100,
            minimum_free_bytes=50,
            owner="first",
        )
        second = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "second",
            requested_bytes=100,
            minimum_free_bytes=50,
            owner="second",
        )
        with patch(
            "proofstack.storage_reservation.shutil.disk_usage",
            return_value=SimpleNamespace(free=190),
        ), patch(
            "proofstack.storage_reservation._workspace_allocated_bytes",
            side_effect=allocated,
        ):
            first.acquire()
            usage["first"] = 70
            second_status = second.acquire()

        assert second_status.active_reserved_bytes == 30
        assert second_status.required_free_bytes == 180
        second.release()
        first.release()


def test_admission_uses_lower_free_space_sample_across_workspace_scans() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        usage = {"first": 80, "second": 0}

        def allocated(path: Path) -> int:
            return usage[Path(path).name]

        first = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "first",
            requested_bytes=100,
            minimum_free_bytes=50,
            owner="first",
        )
        second = StorageReservationLease(
            registry_dir=root / "leases",
            workspace_root=root / "second",
            requested_bytes=400,
            minimum_free_bytes=50,
            owner="second",
        )
        with patch(
            "proofstack.storage_reservation.shutil.disk_usage",
            return_value=SimpleNamespace(free=1_000),
        ), patch(
            "proofstack.storage_reservation._workspace_allocated_bytes",
            side_effect=allocated,
        ):
            first.acquire()

        # Simulate an active worker deleting data during the scan. Sampling
        # only afterwards would pair its old 20-byte remaining reservation
        # with the newer 500-byte free value and incorrectly admit this lease.
        with patch(
            "proofstack.storage_reservation.shutil.disk_usage",
            side_effect=[
                SimpleNamespace(free=420),
                SimpleNamespace(free=500),
            ],
        ), patch(
            "proofstack.storage_reservation._workspace_allocated_bytes",
            side_effect=allocated,
        ), pytest.raises(
            StorageReservationError,
            match="420 bytes free, but 470 bytes are required",
        ):
            second.acquire()

        first.release()


def test_stale_unlocked_lease_is_pruned() -> None:
    with tempfile.TemporaryDirectory() as temp_dir, patch(
        "proofstack.storage_reservation.shutil.disk_usage",
        return_value=SimpleNamespace(free=200),
    ):
        root = Path(temp_dir)
        registry = root / "leases"
        registry.mkdir()
        stale = registry / "lease-stale.json"
        stale.write_text(
            json.dumps({"filesystem_device": -1, "reserved_bytes": 10_000}),
            encoding="utf-8",
        )

        lease = _lease(root, owner="current", requested_bytes=100)
        lease.acquire()
        assert not stale.exists()
        lease.release()


def test_registry_inside_workspace_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        lease = StorageReservationLease(
            registry_dir=workspace / ".leases",
            workspace_root=workspace,
            requested_bytes=1,
            minimum_free_bytes=0,
            owner="bad-config",
        )

        with pytest.raises(StorageReservationError, match="outside the worker workspace"):
            lease.acquire()


def test_async_lease_release_does_not_block_event_loop() -> None:
    started = threading.Event()

    class BlockingLease:
        def release(self) -> None:
            started.set()
            time.sleep(0.15)

    async def exercise() -> int:
        release_task = asyncio.create_task(  # type: ignore[arg-type]
            _release_storage_lease(BlockingLease())
        )
        ticks = 0
        while not started.is_set():
            await asyncio.sleep(0)
        while not release_task.done():
            ticks += 1
            await asyncio.sleep(0.01)
        await release_task
        return ticks

    assert asyncio.run(exercise()) >= 3
