from proofstack.tools.persisted_files import read_persisted_file as _read_persisted_file


def read_persisted_file(file_id: str, *, persisted_file_root=None) -> dict:
    """Read a text file persisted for this workflow run."""
    return _read_persisted_file(file_id, persisted_file_root=persisted_file_root)
