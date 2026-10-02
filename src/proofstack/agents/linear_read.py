"""Prefix-only reading of a manuscript, for define-before-use.

Splits a LaTeX document into passages and asks a cheap model, for each
passage, what a reader who has read ONLY the preceding text cannot yet
understand. A whole-document reviewer cannot un-know a later definition; a
prefix reader genuinely cannot see it, which makes this a measurement rather
than an opinion. All calls go through mathagents.APIClient.
"""

from __future__ import annotations

import json
import re

SYSTEM = """You are a careful mathematician reading a research note strictly from the beginning, one passage at a time. You have read exactly the text shown under TEXT READ SO FAR and nothing else; you cannot look ahead.

For the CURRENT PASSAGE, report every symbol, term, notation, or named result that the passage uses but that has NOT been introduced in the text read so far or in the passage itself, and that a research mathematician in this area would not know from standard background. Flag silent use only:
- Do not flag items the passage itself introduces (defined, or clear from immediate context).
- Do not flag standard notation or terminology of the field, or a named classical object that comes with a citation.
- Do not flag explicit forward pointers (a \\ref to a later result, "defined below", "see Section 4"), unless the sentence cannot be understood at all without the referenced content.
- An introduction or overview may name results and objects it will treat later in loose terms; flag an item there only when the sentence is unintelligible without its definition.
- Do not flag LaTeX macros: the PREAMBLE shows their expansions.
- Do not flag items merely because you would have introduced them differently.

Separately, if a sentence of the passage cannot be understood from what has been read so far, for a reason other than an undefined item, report it under "unclear". Be sparing: precision matters more than recall.

Answer with JSON only, no prose:
{"undefined": [{"item": "<symbol or term>", "quote": "<short quote from the passage>", "why": "<one sentence>"}],
 "unclear": [{"quote": "<short quote>", "why": "<one sentence>"}]}
Use empty lists when there is nothing to report."""

ENV_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
MIN_BLOCK_CHARS = 300


def split_document(tex: str) -> tuple[str, list[str]]:
    """Return (preamble, passages): body split at blank lines and \\section outside environments."""
    head, sep, body = tex.partition("\\begin{document}")
    if not sep:
        head, body = "", tex
    body = body.split("\\end{document}")[0]
    blocks: list[str] = []
    current: list[str] = []
    depth = 0
    for line in body.splitlines():
        if depth == 0 and (not line.strip() or line.lstrip().startswith("\\section")) and current:
            blocks.append("\n".join(current).strip())
            current = []
        if not line.strip() and not current:
            continue
        current.append(line)
        for kind, _ in ENV_RE.findall(line):
            depth += 1 if kind == "begin" else -1
        depth = max(depth, 0)
    if current:
        blocks.append("\n".join(current).strip())
    merged: list[str] = []
    for block in blocks:
        if merged and len(merged[-1]) < MIN_BLOCK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + block
        else:
            merged.append(block)
    return head.strip(), merged


def build_query(preamble: str, seen: list[str], passage: str, index: int) -> list[dict]:
    so_far = "\n\n".join(seen) if seen else "(nothing yet: this is the first passage)"
    user = (f"PREAMBLE (macro definitions only; not part of the text):\n{preamble}\n\n"
            f"TEXT READ SO FAR:\n{so_far}\n\n"
            f"CURRENT PASSAGE (number {index + 1}):\n{passage}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def reply_text(conversation: list[dict]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "assistant" and message.get("type") != "cot":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def parse_reply(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {"undefined": [], "unclear": [], "parse_error": text[:200]}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"undefined": [], "unclear": [], "parse_error": text[:200]}
    return {"undefined": data.get("undefined") or [], "unclear": data.get("unclear") or []}


def render_report(title: str, model: str, blocks: list[str], results: dict[int, dict],
                  cost: float) -> tuple[str, list[dict]]:
    findings: list[dict] = []
    lines = [f"# Linear read of `{title}`", "",
             f"model `{model}`, {len(blocks)} passages, cost ${cost:.3f}", "", "",
             "Each finding is what a reader who has seen only the preceding text could not "
             "understand. Standard notation and cited objects were excluded; judge borderline cases yourself.", ""]
    for i, block in enumerate(blocks):
        res = results.get(i, {"undefined": [], "unclear": []})
        if not res["undefined"] and not res["unclear"] and "parse_error" not in res:
            continue
        first_line = block.splitlines()[0][:90]
        lines.append(f"## Passage {i + 1}: `{first_line}`")
        for item in res["undefined"]:
            lines.append(f"- **{item.get('item', '?')}** — {item.get('why', '')}  \n  > {item.get('quote', '')}")
            findings.append({"passage": i + 1, "kind": "undefined", **item})
        for item in res["unclear"]:
            lines.append(f"- *unclear* — {item.get('why', '')}  \n  > {item.get('quote', '')}")
            findings.append({"passage": i + 1, "kind": "unclear", **item})
        if "parse_error" in res:
            lines.append(f"- parse error: `{res['parse_error']}`")
        lines.append("")
    lines[3] = (f"{sum(f['kind'] == 'undefined' for f in findings)} undefined-item findings, "
                f"{sum(f['kind'] == 'unclear' for f in findings)} unclear-sentence findings")
    return "\n".join(lines) + "\n", findings


def make_client(model_ref: str):
    from mathagents import APIClient, load_solver_config

    cfg = {k: v for k, v in load_solver_config(model_ref).items() if not k.startswith("__")}
    return APIClient(**cfg)


def linear_read(tex: str, model_ref: str, *, title: str = "answer.tex", max_blocks: int = 0,
                client=None, before_batch=None, on_result=None) -> dict:
    """Blocking; bounded batches let the supervisor stop admission and meter replies."""
    preamble, blocks = split_document(tex)
    if max_blocks:
        blocks = blocks[:max_blocks]
    client = client if client is not None else make_client(model_ref)
    results: dict[int, dict] = {}
    cost = 0.0
    for start in range(0, len(blocks), 4):
        if before_batch is not None:
            before_batch()
        if getattr(client, "terminated", False):
            raise RuntimeError("linear read interrupted before all passages were reviewed")
        batch = blocks[start:start + 4]
        queries = [build_query(preamble, blocks[:start + i], block, start + i)
                   for i, block in enumerate(batch)]
        # ProviderTrace totals are keyed by query index across the whole read,
        # not by batch. Reusing 0..3 would charge previous batches again.
        for idx, conversation, detailed in client.run_queries(
                queries, no_tqdm=True, custom_indices=list(range(start, start + len(batch)))):
            cost += detailed.get("cost") or 0.0
            if on_result is not None:
                on_result(detailed)
            results[idx] = parse_reply(reply_text(conversation))
    if getattr(client, "terminated", False) or len(results) != len(blocks):
        raise RuntimeError("linear read interrupted before all passages were reviewed")
    report, findings = render_report(title, client.model, blocks, results, cost)
    return {"report": report, "findings": findings, "passages": len(blocks),
            "cost_usd": cost, "model": client.model, "blocks": blocks}
