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


def _lease(root: Path, *, owner: str, requested_bytes: int) -> StorageReservationLease:
    return StorageReservationLease(
        registry_dir=root / "leases",
        workspace_root=root / "workspace",
        requested_bytes=requested_bytes,
        minimum_free_bytes=50,
        owner=owner,
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
