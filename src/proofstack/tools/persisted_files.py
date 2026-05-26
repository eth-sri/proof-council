"""Per-run text files for model tool loops."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


MAX_READ_CHARS = 200_000


def read_persisted_file(file_id: str, *, persisted_file_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(persisted_file_root)
    path = _path_for(root, file_id)
    if not path.exists():
        return {"ok": False, "file_id": file_id, "error": "file not found"}
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > MAX_READ_CHARS
    return {
        "ok": True,
        "file_id": file_id,
        "text": text[:MAX_READ_CHARS],
        "chars": len(text),
        "truncated": truncated,
    }


def append_persisted_file(
    file_id: str,
    text: str,
    *,
    persisted_file_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(persisted_file_root)
    path = _path_for(root, file_id)
    before = path.stat().st_size if path.exists() else 0
    with path.open("a", encoding="utf-8") as fh:
        fh.write(str(text))
    after_text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "ok": True,
        "file_id": file_id,
        "bytes_before": before,
        "bytes_after": path.stat().st_size,
        "chars": len(after_text),
    }


def edit_persisted_file(
    file_id: str,
    text_before: str,
    text_replace: str,
    *,
    persisted_file_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(persisted_file_root)
    path = _path_for(root, file_id)
    if not path.exists():
        return {"ok": False, "file_id": file_id, "error": "file not found"}
    body = path.read_text(encoding="utf-8", errors="replace")
    needle = str(text_before)
    if not needle:
        return {"ok": False, "file_id": file_id, "error": "text_before is empty"}
    if needle not in body:
        return {"ok": False, "file_id": file_id, "error": "text_before not found"}
    updated = body.replace(needle, str(text_replace), 1)
    path.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "file_id": file_id,
        "replacements": 1,
        "chars": len(updated),
    }


def list_persisted_files(*, persisted_file_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(persisted_file_root)
    manifest = _read_manifest(root)
    files = []
    for file_id, filename in sorted(manifest.items()):
        path = root / filename
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            files.append({"file_id": file_id, "chars": len(text), "bytes": path.stat().st_size})
    return {"ok": True, "files": files}


def _root(value: str | Path | None) -> Path:
    if value is None:
        raise RuntimeError("persisted_file_root was not provided by the agent runtime")
    root = Path(value)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path_for(root: Path, file_id: str) -> Path:
    manifest = _read_manifest(root)
    key = str(file_id or "").strip()
    if not key:
        raise ValueError("file_id cannot be empty")
    filename = manifest.get(key)
    if filename is None:
        filename = _filename_for(key)
        manifest[key] = filename
        _write_manifest(root, manifest)
    return root / filename


def _filename_for(file_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_id).strip("._-")[:80] or "file"
    digest = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}.txt"


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _read_manifest(root: Path) -> dict[str, str]:
    path = _manifest_path(root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _write_manifest(root: Path, manifest: dict[str, str]) -> None:
    _manifest_path(root).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "append_persisted_file",
    "edit_persisted_file",
    "list_persisted_files",
    "read_persisted_file",
]
