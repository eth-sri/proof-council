def code_interpreter() -> None:
    """Provider-managed tool; no local Python function is executed."""
    raise RuntimeError("code_interpreter is provider-managed")
