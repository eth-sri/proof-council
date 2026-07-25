"""Private, run-scoped locations for temporary subscription credentials."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


def _auth_base() -> Path:
    uid = getattr(os, "getuid", lambda: "user")()
    return (
        Path(tempfile.gettempdir()) / f"proofstack-transient-auth-{uid}"
    ).resolve()


def codex_auth_run_root(run_path: Path | str) -> Path:
    """Return the deterministic external auth root associated with a run."""
    resolved = Path(run_path).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:32]
    return _auth_base() / f"codex-{digest}"


def create_codex_auth_parent(run_path: Path | str) -> Path:
    """Create a private, per-agent parent below the run's external auth root."""
    base = _auth_base()
    root = codex_auth_run_root(run_path)
    _ensure_private_dir(base)
    _ensure_private_dir(root)
    parent = Path(tempfile.mkdtemp(prefix="agent-", dir=root))
    _chmod_private(parent)
    return parent


def remove_codex_auth_parent(parent: Path, run_path: Path | str) -> None:
    """Remove one agent's auth parent without touching concurrent agents."""
    root = codex_auth_run_root(run_path)
    if parent.parent != root or not parent.name.startswith("agent-"):
        return
    _remove_without_following(parent)
    _prune_empty(root)
    _prune_empty(_auth_base())


def remove_codex_auth_for_run(run_path: Path | str) -> bool:
    """Remove all external Codex credential material associated with a run."""
    root = codex_auth_run_root(run_path)
    removed = _remove_without_following(root)
    _prune_empty(_auth_base())
    return removed


def _ensure_private_dir(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"cannot prepare transient authentication directory {path}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"transient authentication path is not a directory: {path}")
    getuid = getattr(os, "getuid", None)
    if getuid is not None and metadata.st_uid != getuid():
        raise RuntimeError(
            f"transient authentication directory has the wrong owner: {path}"
        )
    _chmod_private(path)


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError as exc:
        if os.name != "nt":
            raise RuntimeError(
                f"cannot restrict transient authentication directory {path}"
            ) from exc
        return
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise RuntimeError(
            f"transient authentication directory is not private: {path}"
        )


def _remove_without_following(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        return False
    return True


def _prune_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
