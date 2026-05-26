"""ContainerFileBridge — manage canonical workspace files inside an
OpenAI ``code_interpreter`` container.

The OpenAI Responses API has two relevant constraints:

  - Files attached to an ``auto`` container via ``container.file_ids``
    are mounted **read-only** at ``/mnt/data/file-{platform_id}-{name}``.
    The ``file-`` prefix is added by the platform and cannot be
    suppressed via this attachment mechanism.
  - Files written by the model into ``/mnt/data/`` from a ``code_
    interpreter`` cell (any path other than the read-only attachments)
    are writable and persist for the duration of that one Responses
    call.  After the call returns, ``client.containers.files.list``
    enumerates them and ``client.containers.files.content.retrieve``
    streams their contents.

So the design is: upload the workspace's *current* contents as
read-only attachments, instruct the model to copy/edit each one to a
canonical *unprefixed* path inside ``/mnt/data/``, then list+download
those canonical paths after the call.  Any file the model leaves
untouched at its canonical path falls back to the workspace's current
contents.

The bridge does not manage container creation — that is the
``code_interpreter`` tool's ``container: {type: auto}`` flow.  We pull
the container id out of the response output items after the call and
hand it to ``download``.

Cleanup is best-effort: uploaded platform files expire on their own,
but ``cleanup`` deletes them eagerly so the user's OpenAI files list
does not grow without bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from proofstack.agents.ac.blocks import CANONICAL_FILES


CONTAINER_DATA_ROOT = "/mnt/data"


@dataclass
class UploadedFile:
    name: str
    platform_file_id: str
    container_path: str  # e.g. /mnt/data/file-XXXX-answer.tex
    is_canonical: bool = True
    # Optional one-line note rendered alongside the listing entry (used
    # for extra attachments to explain what they are).
    note: str = ""


@dataclass
class ContainerFileBridge:
    """One-shot helper for a single Author turn's file lifecycle."""

    openai_client: object  # OpenAI client; left untyped to avoid hard import here
    workspace: Path
    names: tuple[str, ...] = CANONICAL_FILES
    # Extra read-only attachments to upload alongside the canonical
    # workspace files. Each entry is (path, note); the file's basename is
    # used as the listing name. These do NOT get a writable canonical path
    # — the Author cannot edit them and they will not be downloaded back.
    extra_attachments: list[tuple[Path, str]] = field(default_factory=list)
    uploaded: list[UploadedFile] = field(default_factory=list)

    # ----- upload -----------------------------------------------------------

    def upload(self) -> list[str]:
        """Upload each canonical workspace file (and any extra
        attachments) as platform user_data files.

        Missing or empty workspace files are still uploaded as a
        single-byte placeholder so they appear in ``/mnt/data/`` for
        the model to read.  Returns the list of platform file_ids in
        upload order (canonical files first, then extras).
        """
        ids: list[str] = []
        for name in self.names:
            path = self.workspace / name
            if not path.exists() or path.stat().st_size == 0:
                # Empty placeholder so the upload succeeds and the
                # model sees an attachment with the expected basename.
                path.write_text("(empty)\n", encoding="utf-8")
            ids.append(self._upload_one(path, name, is_canonical=True, note=""))
        for attach_path, note in self.extra_attachments:
            if not attach_path.exists():
                continue
            ids.append(
                self._upload_one(
                    attach_path,
                    attach_path.name,
                    is_canonical=False,
                    note=note,
                )
            )
        return ids

    def _upload_one(
        self, path: Path, name: str, *, is_canonical: bool, note: str
    ) -> str:
        with open(path, "rb") as fh:
            file_obj = self.openai_client.files.create(
                file=fh, purpose="user_data"
            )
        container_path = (
            f"{CONTAINER_DATA_ROOT}/file-{file_obj.id.split('-', 1)[1]}-{name}"
        )
        self.uploaded.append(
            UploadedFile(
                name=name,
                platform_file_id=file_obj.id,
                container_path=container_path,
                is_canonical=is_canonical,
                note=note,
            )
        )
        return file_obj.id

    # ----- prompt fragment --------------------------------------------------

    def render_workspace_listing(self) -> str:
        """Render a listing the Author can quote in its prompt.

        Canonical files have both a read-only input path and a writable
        canonical path. Extra attachments are read-only only — the
        listing includes whatever ``note`` was supplied so the Author
        knows what each attachment contains.
        """
        lines: list[str] = []
        for u in self.uploaded:
            if u.is_canonical:
                lines.append(
                    f"- {u.name}: read from `{u.container_path}` (read-only); "
                    f"write your edited version to `{CONTAINER_DATA_ROOT}/{u.name}`."
                )
            else:
                tail = f" {u.note}" if u.note else ""
                lines.append(
                    f"- {u.name} (read-only attachment): `{u.container_path}`.{tail}"
                )
        return "\n".join(lines)

    @property
    def platform_file_ids(self) -> list[str]:
        return [u.platform_file_id for u in self.uploaded]

    # ----- download ---------------------------------------------------------

    def download(self, container_id: str) -> dict[str, str]:
        """List the container's files and pull contents at the canonical
        write paths (``/mnt/data/<name>``).

        Returns ``{name: contents}`` only for the canonical files that
        were actually written by the model.  Caller treats missing
        names as "Author left this file unchanged".
        """
        target_paths = {f"{CONTAINER_DATA_ROOT}/{n}": n for n in self.names}
        out: dict[str, str] = {}
        files = list(self.openai_client.containers.files.list(container_id))
        for cf in files:
            path = str(getattr(cf, "path", ""))
            name = target_paths.get(path)
            if name is None:
                continue
            body = _read_container_file(self.openai_client, container_id, cf.id)
            out[name] = body
        return out

    # ----- cleanup ----------------------------------------------------------

    def cleanup(self) -> None:
        """Delete uploaded platform files. Failures are ignored."""
        for u in self.uploaded:
            try:
                self.openai_client.files.delete(u.platform_file_id)
            except Exception:
                pass
        self.uploaded.clear()


def find_container_id(conversation: Iterable[dict]) -> str | None:
    """Walk a matharena conversation log and return the first
    ``code_interpreter_call`` entry's ``container_id``.

    All ``code_interpreter_call`` items in a single Responses-API turn
    share the same container id (until OpenAI changes that), so the
    first hit is sufficient.
    """
    for msg in conversation:
        if isinstance(msg, dict) and msg.get("type") == "code_interpreter_call":
            cid = msg.get("container_id")
            if cid:
                return str(cid)
    return None


def _read_container_file(client: object, container_id: str, file_id: str) -> str:
    """Stream a container file's contents and decode as UTF-8."""
    resp = client.containers.files.content.retrieve(
        file_id, container_id=container_id
    )
    if hasattr(resp, "read"):
        body = resp.read()
    elif hasattr(resp, "content"):
        body = resp.content
    else:
        body = bytes(resp)
    if isinstance(body, bytes):
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return body.decode("utf-8", errors="replace")
    return str(body)


__all__ = [
    "CONTAINER_DATA_ROOT",
    "ContainerFileBridge",
    "UploadedFile",
    "find_container_id",
]
