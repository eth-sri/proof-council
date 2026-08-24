from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.storage_reservation import (
    StorageReservationError,
    StorageReservationLease,
)


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
