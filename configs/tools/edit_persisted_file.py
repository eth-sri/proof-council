from proofstack.tools.persisted_files import edit_persisted_file as _edit_persisted_file


def edit_persisted_file(file_id: str, text_before: str, text_replace: str, *, persisted_file_root=None) -> dict:
    """Replace text in a file persisted for this workflow run."""
    return _edit_persisted_file(
        file_id,
        text_before,
        text_replace,
        persisted_file_root=persisted_file_root,
    )
