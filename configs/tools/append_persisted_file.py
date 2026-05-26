from proofstack.tools.persisted_files import append_persisted_file as _append_persisted_file


def append_persisted_file(file_id: str, text: str, *, persisted_file_root=None) -> dict:
    """Append text to a file persisted for this workflow run."""
    return _append_persisted_file(file_id, text, persisted_file_root=persisted_file_root)
