"""Regression tests for ``<compute_agent>`` parsing in
``proofstack.agents.ac.blocks``.

The Author may emit at most one ``<compute_agent>...</compute_agent>``
block per turn alongside the existing ``<council>`` and ``<ready>``
markup. The parser extracts the instructions string, strips the block
from the free-text ``thinking_summary``, and warns on duplicates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.blocks import (  # noqa: E402
    ComputeRequest,
    CouncilRequest,
    parse_author_output,
)


def test_compute_alongside_council_and_files():
    raw = (
        "Some thinking text...\n\n"
        "```file path=answer.tex\n\\documentclass{article}\n\\begin{document}body\\end{document}\n```\n\n"
        "<council>\nWhat is X?\n</council>\n\n"
        "<compute_agent>\n"
        "Run a Python script to enumerate degree-3 polys over F_5.\n"
        "Save results to data/enum.csv.\n"
        "</compute_agent>\n\n"
        "<ready>false</ready>\n"
    )
    p = parse_author_output(raw)

    assert isinstance(p.council, CouncilRequest)
    assert p.council.question == "What is X?"

    assert isinstance(p.compute, ComputeRequest)
    assert "Run a Python script" in p.compute.instructions
    assert "data/enum.csv" in p.compute.instructions

    assert "answer.tex" in p.files
    assert not p.ready
    assert "Some thinking text" in p.thinking_summary
    assert "<compute_agent>" not in p.thinking_summary
    assert "<council>" not in p.thinking_summary


def test_duplicate_compute_blocks_first_wins_with_warning():
    raw = (
        "<compute_agent>first</compute_agent>"
        " middle text "
        "<compute_agent>second</compute_agent>"
    )
    p = parse_author_output(raw)

    assert p.compute is not None
    assert p.compute.instructions == "first"
    assert any("extra <compute_agent>" in w for w in p.parse_warnings)


def test_empty_compute_block_yields_no_request():
    p = parse_author_output("<compute_agent>   </compute_agent>")
    assert p.compute is None


def test_compute_case_insensitive():
    p = parse_author_output("<COMPUTE_AGENT>X</COMPUTE_AGENT>")
    assert p.compute is not None
    assert p.compute.instructions == "X"


def test_absent_compute_does_not_affect_other_fields():
    raw = "just text and <council>q</council> and <ready>true</ready>"
    p = parse_author_output(raw)
    assert p.compute is None
    assert p.council is not None and p.council.question == "q"
    assert p.ready is True


def test_ready_tag_inside_inline_code_is_ignored():
    raw = "I am not declaring `<ready>true</ready>` yet."
    p = parse_author_output(raw)

    assert p.ready is False
    assert any("ignored <ready>" in w for w in p.parse_warnings)


def test_ready_tag_inside_fenced_code_is_ignored():
    raw = "```text\n<ready>true</ready>\n```\n<ready>false</ready>\n"
    p = parse_author_output(raw)

    assert p.ready is False


def test_last_visible_ready_tag_wins():
    raw = "<ready>false</ready>\nAfter final fixes:\n<ready>true</ready>\n"
    p = parse_author_output(raw)

    assert p.ready is True
