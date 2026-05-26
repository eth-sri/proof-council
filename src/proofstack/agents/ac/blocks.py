"""Parsers for Author response markup.

The Author writes its turn output as a free-form assistant message
that can contain four kinds of structured block, all extracted by
regex here:

- ``file path=<name>`` fenced code blocks with the new contents of one
  of the three canonical workspace files (``answer.tex``,
  ``research_notes.tex``, ``references.bib``). Files not present in
  the response are left unchanged on disk.

- A ``<council>...</council>`` envelope holding a single sub-question
  for the Advisory Council to answer in parallel with the Critic.

- A ``<compute_agent>...</compute_agent>`` envelope holding
  instructions for an out-of-band codex CLI worker with a persistent
  workspace and a 60 minute soft timeout. Runs in parallel with the
  Critic and Council; its reply lands in the next Author turn.

- A ``<ready>true</ready>`` flag the Author may set when it believes
  the answer is shippable. The workflow's early-stop gate also
  requires the Critic to concur.

Anything outside these blocks is treated as the Author's free-text
``thinking_summary`` and is preserved for the archival event log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


CANONICAL_FILES = ("answer.tex", "research_notes.tex", "references.bib")


# ``` followed by ``file`` info string, optional space, ``path=NAME``.
# Matches the entire fenced block including the closing fence.
# Filename pattern allows alphanumerics, dot, underscore, hyphen, slash.
_FILE_BLOCK_RE = re.compile(
    r"```file\s+path=(?P<path>[A-Za-z0-9._/-]+)\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)

_COUNCIL_RE = re.compile(
    r"<council(?:\s+to=\"(?P<to>[^\"]*)\")?>(?P<body>.*?)</council>",
    re.DOTALL | re.IGNORECASE,
)

_COMPUTE_RE = re.compile(
    r"<compute_agent>(?P<body>.*?)</compute_agent>",
    re.DOTALL | re.IGNORECASE,
)

_READY_RE = re.compile(
    r"<ready>\s*(?P<value>true|false)\s*</ready>",
    re.IGNORECASE,
)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")


@dataclass
class CouncilRequest:
    question: str
    to: list[str] = field(default_factory=list)


@dataclass
class ComputeRequest:
    instructions: str


@dataclass
class ParsedAuthorOutput:
    files: dict[str, str] = field(default_factory=dict)
    council: CouncilRequest | None = None
    compute: ComputeRequest | None = None
    ready: bool = False
    thinking_summary: str = ""
    parse_warnings: list[str] = field(default_factory=list)


def parse_author_output(raw_text: str) -> ParsedAuthorOutput:
    """Extract file rewrites, optional council request, and ready flag.

    Files outside ``CANONICAL_FILES`` are recorded with a parse warning
    and ignored. Multiple ``<council>`` blocks: only the first wins
    (the workflow caps invocations at one per round); a warning is
    emitted for any extras.
    """
    out = ParsedAuthorOutput()

    files: dict[str, str] = {}
    file_block_spans: list[tuple[int, int]] = []
    for m in _FILE_BLOCK_RE.finditer(raw_text):
        path = m.group("path").strip()
        body = m.group("body")
        file_block_spans.append((m.start(), m.end()))
        if path not in CANONICAL_FILES:
            out.parse_warnings.append(
                f"ignored non-canonical file block path={path!r}"
            )
            continue
        if path in files:
            out.parse_warnings.append(
                f"duplicate file block for {path!r}; keeping last"
            )
        files[path] = body
    out.files = files

    council_matches = list(_COUNCIL_RE.finditer(raw_text))
    council_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in council_matches
    ]
    if council_matches:
        first = council_matches[0]
        question = first.group("body").strip()
        to_raw = (first.group("to") or "").strip()
        to_list = [s.strip() for s in to_raw.split(",") if s.strip()]
        if question:
            out.council = CouncilRequest(question=question, to=to_list)
        if len(council_matches) > 1:
            out.parse_warnings.append(
                f"{len(council_matches) - 1} extra <council> block(s) ignored"
            )

    compute_matches = list(_COMPUTE_RE.finditer(raw_text))
    compute_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in compute_matches
    ]
    if compute_matches:
        instructions = compute_matches[0].group("body").strip()
        if instructions:
            out.compute = ComputeRequest(instructions=instructions)
        if len(compute_matches) > 1:
            out.parse_warnings.append(
                f"{len(compute_matches) - 1} extra <compute_agent> block(s) ignored"
            )

    visible_text = _strip_markdown_code(raw_text)
    ready_matches = list(_READY_RE.finditer(visible_text))
    if ready_matches:
        ready_m = ready_matches[-1]
        out.ready = ready_m.group("value").lower() == "true"
    elif _READY_RE.search(raw_text):
        out.parse_warnings.append("ignored <ready> tag inside Markdown code")

    out.thinking_summary = _strip_spans(
        raw_text, file_block_spans + council_spans + compute_spans
    ).strip()
    return out


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans_sorted = sorted(spans)
    parts: list[str] = []
    cursor = 0
    for start, end in spans_sorted:
        if start > cursor:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])
    return "".join(parts)


def _strip_markdown_code(text: str) -> str:
    return _INLINE_CODE_RE.sub("", _FENCED_CODE_RE.sub("", text))


__all__ = [
    "CANONICAL_FILES",
    "CouncilRequest",
    "ComputeRequest",
    "ParsedAuthorOutput",
    "parse_author_output",
]
