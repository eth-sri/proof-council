from proofstack.tools.persisted_files import list_persisted_files as _list_persisted_files


def list_persisted_files(*, persisted_file_root=None) -> dict:
    """List files persisted for this workflow run."""
    return _list_persisted_files(persisted_file_root=persisted_file_root)
