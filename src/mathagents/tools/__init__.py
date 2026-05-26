from __future__ import annotations

from importlib import import_module
from pathlib import Path


_MODULE_EXPORTS = {
    "code_execution": ["execute_code", "execute_code_long"],
    "paper_search": [
        "check_and_prepare_paper",
        "download_paper_pdf",
        "find_in_paper",
        "ocr",
        "ocr_paper",
        "query_semantic_scholar",
        "read_pages",
        "read_paper",
    ],
    "query_knowledge": ["query_knowledge"],
}

_AVAILABLE_MODULES = {
    path.stem
    for path in Path(__file__).resolve().parent.glob("*.py")
    if path.stem != "__init__"
}

__all__ = [
    export
    for module_name, exports in _MODULE_EXPORTS.items()
    if module_name in _AVAILABLE_MODULES
    for export in exports
]


def __getattr__(name):
    for module_name, exports in _MODULE_EXPORTS.items():
        if name not in exports:
            continue
        if module_name not in _AVAILABLE_MODULES:
            raise AttributeError(
                f"{name} is unavailable because mathagents.tools.{module_name} is not present in this checkout."
            )
        module = import_module(f"mathagents.tools.{module_name}")
        return getattr(module, name)
    raise AttributeError(name)
