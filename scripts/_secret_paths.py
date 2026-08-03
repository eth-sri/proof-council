"""Single source of truth for path names treated as credential/secret
material, shared by ``firstproof_entrypoint.py`` (output retrieval) and
``postscreen_run.py`` (audit bundling) so the two policies cannot drift."""
from __future__ import annotations

from pathlib import PurePath

SECRET_DIR_NAMES = frozenset(
    {".aws", ".codex", ".codex-home", ".compute_codex_home", ".ssh", "secrets"}
)
SECRET_FILE_NAMES = frozenset(
    {".env", "auth.json", "credentials", "credentials.json", "id_ed25519", "id_rsa"}
)


def is_secret_file_name(name: str) -> bool:
    return name in SECRET_FILE_NAMES or name.endswith(".env")


def is_secret_rel_path(rel: PurePath) -> bool:
    return (
        any(part in SECRET_DIR_NAMES for part in rel.parts)
        or is_secret_file_name(rel.name)
    )
