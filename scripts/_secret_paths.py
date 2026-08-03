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


def is_secret_dir_name(name: str) -> bool:
    return name.lower() in SECRET_DIR_NAMES


def is_secret_file_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in SECRET_FILE_NAMES
        # Dotenv variants: .env.local, .env.production, .envrc, prod.env, ...
        or lowered.startswith(".env")
        or lowered.endswith(".env")
    )


def is_secret_rel_path(rel: PurePath) -> bool:
    return (
        any(is_secret_dir_name(part) for part in rel.parts)
        or is_secret_file_name(rel.name)
    )
