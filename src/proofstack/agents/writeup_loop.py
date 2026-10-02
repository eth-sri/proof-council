"""WriteupLoop — guide-driven one-shot rewrite with a cold-referee/repair loop.

Legacy ProofCouncil composite Agent, intended to slot between the
Author/Critic loop's ``answer.tex`` and FirstProof submission. A cheaper
fallback: ``CleanupSession`` is the default write-up node. See
``writeup_prompts/APPROVALS.md`` for prompt provenance and maintenance.

Loop design (v2.1):
  1. Rewrite the document via the approved wrapper around Johannes
     Schmitt's writing guide; mechanical gate (compiles, and within the
     page limit when one is set), one re-roll, else ship the original.
  2. Up to ``rounds`` times: cold referee (fresh seat, document only),
     then a repair-prompt pass on every report. A clean (NO ERRORS)
     report's minor suggestions are applied as a "polish" step and
     shipped without re-check; an ERRORS FOUND report's repair continues
     the loop. Repaired documents that fail the gate are discarded
     (their UNABLE: declarations are NOT discarded — a broken repair
     does not neutralize a catastrophe signal), and a polish that
     declares UNABLE: is discarded too (a clean report has no critical
     errors for it to be unable to fix).
  3. Ship the final document — unless the final repair declared UNABLE:
     on a critical error (the catastrophe condition), in which case the
     original ships unchanged.

Reliability contract: ``run()`` never raises — including on task
cancellation, deliberately caught so a harness timeout still yields a
document — and never ships a blank document when its input was nonblank.
Any seat failure degrades along a defined path (re-roll, ship unchecked,
ship pre-repair); ``BudgetExhausted``, seat deadlines, and unexpected
errors ship the best version so far, the original if none. Improved
documents are always gated on EXACTLY the bytes that ship, unmutated;
the original input, as the trust anchor, ships ungated. Decisions come
from mechanical signals — gate results and exact sentinel-line matches.
(The UNABLE:/FIXED: declaration is a model self-report; trusting it is
a deliberate policy choice, not independent verification.)

Seat calls are wallclock-bounded by the MINIMUM remaining across the
node's whole budget-scope chain, enforced primarily INSIDE the API
client (``max_wallclock_per_call_s`` and SDK ``timeout`` overrides), so
a timed-out background call is cancelled server-side and surfaces as an
ordinary seat error (with an ``agent.error`` event). Known accounting
gap, accepted: the client's timeout path raises before registering the
cancelled call's usage, so a timed-out seat's spend is not reflected in
``usd_total`` or the budget tracker. The outer ``wait_for`` with slack
is only a backstop against a hung client.

What the framework does NOT guarantee (integration requirement): input
validation, resume-cache reads, and event-log writes happen in
``Agent.__call__`` outside this net, so the HARNESS invocation of this
node must sit in its own try/except with a ship-the-original fallback.

Seats default to gpt-5.6-sol at max effort (xhigh stays available as an
override) with server-side
web_search, as a single user turn with no system prompt.

Two seat implementations, selected by the ``seat`` component config
("api", the default, or "cli"); the loop, the gates and the frozen
prompts are identical either way:

  api  ``_OneShotSeat`` -> OpenAI Responses API, background+poll, billed
       to an API key.
  cli  ``codex exec`` on the ChatGPT subscription (see
       ``writeup_codex_seat``), same model family and web search, $0 —
       and refused outright unless the codex login really is a ChatGPT
       one, since $0 is recorded rather than measured.
       Blocking rather than background: the wallclock bound is enforced
       by killing the process group, and a timeout surfaces as an
       ordinary seat failure, so every degradation path is unchanged.
       Usage is metered in tokens; dollars are 0. Unlike an API reply
       it can echo a local credential back, so a reply carrying one is
       failed outright — rather than scrubbed and kept, which would edit
       the document silently — and the ordinary degradation path takes
       over.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.agents.writeup_codex_seat import (
    CodexSeatError,
    DEFAULT_CODEX_BIN,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    looks_like_credential,
    redact_seat_text,
    run_codex_seat,
)
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.kinds.api_call import APICallAgent, Message, ModelSpec

_PROMPTS_DIR = Path(__file__).resolve().parent / "writeup_prompts"

# (run_id, node name) pairs already claimed by a stage in this process.
# Only used to spot a chain whose stages were left sharing one name; see
# ``WriteupLoop._stage_tag``.
_STAGE_CLAIMED: set[tuple[str, str]] = set()
_STAGE_CLAIMED_MAX = 4096

_GUIDE_PLACEHOLDER = "{{full text of WRITING_GUIDANCE.md}}"
_NOTES_BLOCK_PLACEHOLDER = "{{research notes block}}"


def _template(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _assemble(template_name: str, subs: dict[str, str],
              research_notes: str | None = None,
              writing_constraints: str | None = None) -> str:
    """Fill the approved template. Placeholders are replaced literally;
    the surrounding words are never touched (standing rule: model-facing
    prose is approved byte-exact). Substitution is single-pass over the
    template's own placeholder positions (every occurrence), so
    placeholder-like text inside a substituted VALUE is never itself
    substituted, and a document containing '{{' trips nothing.

    The research-notes block is optional (David 2026-08-31: let the
    rewriter see them if present): when notes are given, the placeholder
    expands to the approved research-notes-block.txt fragment; when
    absent, placeholder and following blank line are removed, leaving
    the previously approved wrapper byte-exact. Optional caller-supplied
    submission constraints are appended only after template substitution."""
    text = _template(template_name)
    if _GUIDE_PLACEHOLDER in text:
        text = text.replace(_GUIDE_PLACEHOLDER, _template("WRITING_GUIDANCE.md"))
    if _NOTES_BLOCK_PLACEHOLDER in text:
        if research_notes:
            # Insert the fragment with its {{research notes}} placeholder
            # intact and resolve everything in ONE _splice below, so a
            # value (e.g. notes containing a literal '{{document}}') is
            # never rescanned for placeholders (codex r3 #14).
            block = _template("research-notes-block.txt").rstrip("\n")
            text = text.replace(_NOTES_BLOCK_PLACEHOLDER, block)
            subs = {**subs, "research notes": research_notes}
        else:
            text = text.replace(_NOTES_BLOCK_PLACEHOLDER + "\n\n", "")
    prompt = _splice(text, subs)
    if writing_constraints:
        prompt += "\n\n# Submission constraints\n\n" + writing_constraints
    return prompt


def _splice(text: str, subs: dict[str, str]) -> str:
    """Single-pass positional substitution over EVERY occurrence of each
    placeholder; spans are located in the original text before any value
    is inserted."""
    spans: list[tuple[int, int, str]] = []
    for key, value in subs.items():
        placeholder = "{{" + key + "}}"
        found = False
        start = 0
        while True:
            idx = text.find(placeholder, start)
            if idx < 0:
                break
            found = True
            spans.append((idx, idx + len(placeholder), value))
            start = idx + len(placeholder)
        if not found:
            raise RuntimeError(f"template lacks placeholder {placeholder}")
    spans.sort()
    parts: list[str] = []
    cursor = 0
    for start, end, value in spans:
        parts.append(text[cursor:start])
        parts.append(value)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)\n```", flags=re.DOTALL)

_VERBATIM_RE = re.compile(
    r"\\begin\s*\{(verbatim\*?|lstlisting)\}.*?\\end\s*\{\1\}",
    flags=re.DOTALL)
_VERBATIM_OPEN_RE = re.compile(r"\\begin\s*\{(?:verbatim\*?|lstlisting)\}")


def _extract_tex(output: str) -> tuple[str, str]:
    """Split a model reply into (document, declaration text).

    The document is the first fenced block CONTAINING ``\\documentclass``
    (models sometimes fence an explanation before the real document),
    falling back to the first fence, else everything up to and including
    ``\\end{document}``. Declarations (FIXED:/UNABLE: lines) are
    collected from AFTER the document — including trailing lines a model
    left inside the fence after ``\\end{document}``, which TeX ignores
    but the loop must not."""
    fences = list(_FENCE_RE.finditer(output))
    m = (next((f for f in fences
               if "\\documentclass" in f.group(1)
               and "\\end{document}" in f.group(1)), None)
         or next((f for f in fences if "\\documentclass" in f.group(1)), None)
         or (fences[0] if fences else None))
    if m:
        doc, trailing = m.group(1) + "\n", output[m.end():]
    else:
        e = re.search(r"\\end\{document\}", output)
        if e:
            doc, trailing = output[: e.end()] + "\n", output[e.end():]
        else:
            # No fence and no \\end{document}: there is no document here.
            # The reply is still scanned for declarations — a repair that
            # answers only 'UNABLE: ...' must not lose its catastrophe
            # signal (codex r5 #1). The full text is also returned as the
            # doc candidate; without \\documentclass it cannot pass the
            # gate, so nothing unvetted ships.
            return output, _strip_listings(output)
    e = re.search(r"\\end\{document\}", doc)
    if e:
        inside_tail = doc[e.end():]
        doc = doc[: e.end()] + "\n"
        # NOT stripped: this is after \end{document}, so it is the model
        # talking, not the document quoting output — a declaration left
        # there is exactly the case r8 #3 exists for, and stripping it
        # hid one (codex r14 #2).
        trailing = inside_tail + "\n" + trailing
    else:
        # No \end{document} in the chosen fence: it holds either no
        # document at all (a fenced 'UNABLE: ...' reply, codex r6 #1) or a
        # truncated one (codex r8 #3). Both fail the gate; both are
        # scanned for declarations, because neither fencing nor truncation
        # may hide a catastrophe signal — but the body's LISTINGS are the
        # document quoting output, not the model declaring anything.
        trailing = _strip_listings(doc) + "\n" + trailing
    return doc, trailing


def _clean_line(line: str) -> str:
    return line.strip().strip("*_`#").strip()


def _verdict_line(text: str, sentinel: str) -> bool:
    """Exact-line sentinel match (markdown emphasis tolerated), so prose
    mentioning a sentinel ('I cannot certify NO ERRORS because...')
    cannot drive control flow."""
    return any(_clean_line(line) == sentinel for line in text.splitlines())


def _declares(text: str, prefix: str) -> bool:
    """A declaration is a line BEGINNING with the prefix (FIXED:/UNABLE:),
    not a substring anywhere in prose."""
    return any(line.lstrip().startswith(prefix) for line in text.splitlines())


def _strip_listings(text: str) -> str:
    """Verbatim/lstlisting spans removed, including an UNTERMINATED one.

    Applied to DOCUMENT BODY text on its way into the declaration scan,
    and only there — never to what follows ``\\end{document}``. A reply truncated inside a listing had the listing's
    own contents read as declarations, so ``UNABLE: output of the example
    algorithm`` could taint a perfectly good document (codex r12 #3). The
    text that FOLLOWS the document is never stripped — cutting everything
    after an unterminated listing swallowed a genuine declaration written
    after the fence, which is the dangerous direction of the same mistake
    (codex r13 #2)."""
    text = _VERBATIM_RE.sub("", text)
    m = _VERBATIM_OPEN_RE.search(text)
    return text[: m.start()] if m else text


def _gate_detail(gate: "GateResult") -> str:
    base = f"compiled={gate.compiled} pages={gate.pages}"
    return f"{base} {gate.note}" if gate.note else base


def _completed_reply_text(exc: BaseException) -> str:
    """Text of a seat reply that arrived and was then discarded by the
    failure the exception reports.

    Two carriers, same reason. ``completed_output``: api_call.py runs its
    budget check after the call is charged but before the reply reaches
    the caller, so without this a completed repair's UNABLE: would vanish
    and the catastrophe fallback would be bypassed by a budget crossing
    (codex r7 #3). ``partial``: a CLI turn refused for carrying a
    credential, or one that exited nonzero after writing its message, also
    completed as far as the model is concerned — its UNABLE: is a real
    declaration and must not be lost to the discard (codex r11 #2)."""
    for attr in ("completed_output", "partial"):
        text = getattr(getattr(exc, attr, None), "text", None)
        if isinstance(text, str) and text:
            return text
    return ""


def _flag_lines(tex: str) -> list[str]:
    return [ln for ln in tex.splitlines() if ln.lstrip().startswith("%% FLAG:")]


_PROOF_ENV_RE = re.compile(r"\\begin\s*\{proof\}.*?\\end\s*\{proof\}",
                           flags=re.DOTALL)
_PARAGRAPH_CMD_RE = re.compile(r"\\(paragraph|subparagraph)(\*?)\s*\{")
# secnumdepth above which each command is numbered (article and friends:
# paragraph is sectioning level 4, subparagraph 5).
_SECNUMDEPTH_OF = {"paragraph": 3, "subparagraph": 4}


def _run_in_heading(cmd: str, star: str, title: str) -> str:
    """The bold run-in replacement, carrying the ORIGINAL heading's
    counter and label semantics. An unstarred \\paragraph steps its
    counter — and so retargets a following \\label — exactly when
    secnumdepth exceeds 3; dropping that made two labelled proof steps
    both resolve to the enclosing section under
    ``\\setcounter{secnumdepth}{4}`` (codex r7 #6). The test below is
    LaTeX's own, so the default (unnumbered) case renders as before."""
    plain = "\\textbf{" + title + "}"
    if star:
        return "\\medskip\\par\\noindent" + plain + "\\quad"
    return ("\\medskip\\par\\noindent"
            "\\ifnum\\value{secnumdepth}>" + str(_SECNUMDEPTH_OF[cmd])
            + "\\relax\\refstepcounter{" + cmd + "}"
            "\\textbf{\\the" + cmd + "\\quad " + title + "}"
            "\\else" + plain + "\\fi\\quad")


def _replace_paragraph_cmds(segment: str) -> str:
    """Replace \\paragraph{...}/\\subparagraph{...} with a bold run-in
    heading, parsing the argument with a real brace counter (regex
    alternation caps nesting depth; \\paragraph{Step \\textbf{a \\emph{b}}}
    must match)."""
    out: list[str] = []
    pos = 0
    while True:
        m = _PARAGRAPH_CMD_RE.search(segment, pos)
        if not m:
            out.append(segment[pos:])
            return "".join(out)
        depth = 1
        i = m.end()
        while i < len(segment) and depth:
            c = segment[i]
            if c == "\\" :
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced: leave untouched, let the gate decide
            out.append(segment[pos:])
            return "".join(out)
        title = segment[m.end(): i - 1]
        out.append(segment[pos:m.start()])
        out.append(_run_in_heading(m.group(1), m.group(2), title))
        pos = i


def _defuse_proof_sectioning(tex: str) -> str:
    """\\paragraph inside amsthm proof environments is illegal LaTeX
    (proof is a trivlist; TeX Live 2025 dies with 'Improper
    \\prevdepth' / 'missing \\item'), and the rewriter produces
    '\\paragraph{Step N}' proof signposting constantly — the guide asks
    for signposted steps. Deterministic transform to a bold run-in
    heading, applied BEFORE the gate so the gate still checks exactly
    the bytes that ship. Deliberately narrow: only the paragraph-level
    commands, whose counter/label semantics the replacement reproduces
    (see ``_run_in_heading``); higher numbered sectioning inside a proof
    is left to fail the gate; verbatim and lstlisting spans are
    protected; arguments may nest one brace level."""
    protected: list[str] = []

    def shield(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f"\x00WRITEUP_VERB_{len(protected) - 1}\x00"

    tex = _VERBATIM_RE.sub(shield, tex)

    tex = _PROOF_ENV_RE.sub(lambda m: _replace_paragraph_cmds(m.group(0)), tex)
    for i, span in enumerate(protected):
        tex = tex.replace(f"\x00WRITEUP_VERB_{i}\x00", span)
    return tex


_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?!s)")
_BIBTEX_RE = re.compile(r"\\bibliography\s*\{([^}]+)\}")
# \addbibresource takes an optional argument (\addbibresource[location=local]
# {refs.bib}); missing it meant the .bib was never written and no backend ran
# (codex r8 #2).
_BIBLATEX_RE = re.compile(r"\\addbibresource\s*(?:\[[^]]*\])?\s*\{([^}]+)\}")
_BIBLATEX_PKG_RE = re.compile(
    r"\\usepackage\s*\[([^]]*)\]\s*\{([^}]*)\}")
_BIBLATEX_BACKEND_RE = re.compile(r"\bbackend\s*=\s*([A-Za-z0-9]+)")
_LOG_PAGES_RE = re.compile(rb"Output written on main\.pdf \((\d+) pages?")


def _biblatex_backend(tex: str) -> str:
    """The tool biblatex was told to use. ``backend=bibtex`` (or bibtex8)
    is a legitimate configuration whose documents still declare their
    bibliography with \\addbibresource, and sending those to biber failed
    the gate on a host where only bibtex was installed (codex r8 #5).
    biblatex's own default is biber, so that is the default here."""
    for m in _BIBLATEX_PKG_RE.finditer(tex):
        if "biblatex" not in [p.strip() for p in m.group(2).split(",")]:
            continue
        opt = _BIBLATEX_BACKEND_RE.search(m.group(1))
        if opt:
            return "bibtex" if opt.group(1).startswith("bibtex") else "biber"
    return "biber"


def _count_pdf_pages(pdf_path: Path, last_stdout: bytes) -> int:
    """Page count, in decreasing order of trust: PyMuPDF reads the PDF
    itself; pdflatex's anchored 'Output written' line; the raw object
    regex (undercounts on compressed PDFs). A loose '(N pages' scan is
    spoofable by \\typeout in the document — do not reintroduce it."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import fitz  # type: ignore[import-not-found]

        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception:
        pass
    matches = _LOG_PAGES_RE.findall(last_stdout)
    if matches:
        # Last occurrence: pdflatex's own closing line. An earlier
        # \typeout can fabricate a matching line but cannot appear
        # after the genuine one (codex r3 #5).
        return int(matches[-1])
    try:
        return len(_PDF_PAGE_RE.findall(pdf_path.read_bytes()))
    except Exception:
        return 0


def _kill_compiler(proc: "subprocess.Popen[bytes]") -> None:
    """Kill a compiler pass and anything it spawned. The passes are started
    in their own session, so the group can be signalled as a unit."""
    with contextlib.suppress(Exception):
        os.killpg(proc.pid, signal.SIGKILL)
        return
    with contextlib.suppress(Exception):
        proc.kill()


class _GateCanceller:
    """A stop channel from the event loop into the gate's worker thread.

    Cancelling ``asyncio.to_thread`` only abandons the future: the thread
    runs on, so the pdflatex passes kept going against the original
    deadline after ``_ship`` had already returned, and ``asyncio.run``
    then blocked on executor shutdown (codex r9 #3). The flag is checked
    between passes and the pass in flight is killed outright.
    """

    def __init__(self) -> None:
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._live: subprocess.Popen[bytes] | None = None

    def stopped(self) -> bool:
        return self._stopped.is_set()

    def track(self, proc: "subprocess.Popen[bytes]") -> None:
        """Register the pass in flight (worker thread)."""
        with self._lock:
            if self._stopped.is_set():
                # Cancelled between the check and the spawn.
                _kill_compiler(proc)
                return
            self._live = proc

    def untrack(self) -> None:
        with self._lock:
            self._live = None

    def cancel(self) -> None:
        """Stop the gate (event loop). Safe to call more than once."""
        self._stopped.set()
        with self._lock:
            proc = self._live
        if proc is not None:
            _kill_compiler(proc)


def _compile_raw(tex: str, bib_text: str | None,
                 pass_timeout: int = 300,
                 deadline: float | None = None,
                 canceller: _GateCanceller | None = None,
                 bbl_output: Path | None = None,
                 secure: bool = False,
                 ) -> tuple[bool, int, str]:
    """Compile EXACTLY the given bytes — no wrapping, no normalization,
    no repair — so gate-passing text is text that ships working. Returns
    (compiled, pages, note); any failure mode (error, missing binary,
    hang, blank input) is (False, 0, note), never an exception. ``note``
    is a short diagnosis for the step record, "" when there is nothing
    to say.

    Bibliography: bib_text is written both as references.bib and under
    every name the document's \\bibliography/\\addbibresource actually
    references; when the bibliography tool runs, its exit code is
    enforced. A document citing a bibliography we were not given
    compiles with unresolved references — a warning, not a gate failure,
    matching the AC pipeline's stance. A bibliography we WERE given but
    cannot process (backend not installed) is the opposite case and IS a
    gate failure: pdflatex still exits 0, so the document would ship with
    an empty bibliography, unresolved citations and an undercounted page
    total (codex r7 #5)."""
    try:
        if not tex.strip() or "\\documentclass" not in tex:
            return False, 0, ""
        with tempfile.TemporaryDirectory(prefix="writeup_gate_") as work_str:
            work = Path(work_str)
            (work / "main.tex").write_text(tex, encoding="utf-8")
            def _safe(name: str) -> str | None:
                # Bib names come from model output: only plain basenames
                # are honored — a path-qualified or absolute name must
                # not escape the temp dir (codex r3 #6).
                name = name.strip()
                if name.endswith(".bib"):
                    name = name[:-4]
                return name if re.fullmatch(r"[A-Za-z0-9._-]+", name) else None

            bib_names = [s for m in _BIBTEX_RE.finditer(tex)
                         for n in m.group(1).split(",")
                         if (s := _safe(n))]
            biblatex_names = [s for m in _BIBLATEX_RE.finditer(tex)
                              if (s := _safe(Path(m.group(1).strip()).name))]
            if bib_text:
                # A document loading several resources loads them ALL, so
                # writing the merged bibliography into each one made bibtex
                # (and biber) fail on repeated entry keys and sank the gate
                # for a perfectly good document (codex r9 #5). The entries
                # go to the FIRST resource the document names; the rest are
                # created empty, so the loads still resolve.
                ordered = list(dict.fromkeys([*bib_names, *biblatex_names]))
                primary = ordered[0] if ordered else "references"
                for name in dict.fromkeys([primary, *ordered, "references"]):
                    (work / f"{name}.bib").write_text(
                        bib_text if name == primary else "", encoding="utf-8")

            last_stdout = b""
            failure = ""

            def diagnosis(stage):
                log = work / "main.log"
                text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                return f"{stage}: {failure}\n{text[-4000:]}\n{last_stdout.decode('utf-8', errors='replace')[-2000:]}".strip()

            def _run(cmd: list[str], timeout: int | None = None) -> int:
                nonlocal last_stdout, failure
                if canceller is not None and canceller.stopped():
                    failure = "compilation cancelled"
                    return 1
                timeout = min(timeout or 300, pass_timeout)
                if deadline is not None:
                    # Recompute against the LIVE deadline: successive
                    # passes must not each spend a stale snapshot
                    # (codex r5 #3), and an expired deadline launches
                    # nothing at all (codex r6 #3).
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        failure = "compile deadline reached"
                        return 1
                    timeout = min(timeout, rem)
                # Popen rather than subprocess.run, and its own session, so
                # a cancelled gate can kill the pass in flight and whatever
                # it spawned (codex r9 #3).
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=work, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, start_new_session=True,
                        **({"env": {
                            "PATH": os.environ.get("PATH", os.defpath),
                            "HOME": str(work), "LANG": "C.UTF-8",
                            "openin_any": "p", "openout_any": "p", "shell_escape": "f",
                        }} if secure else {}))
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    return 1
                if canceller is not None:
                    canceller.track(proc)
                try:
                    out, _err = proc.communicate(timeout=timeout)
                    last_stdout = (out or b"") + (_err or b"")
                    failure = f"exit code {proc.returncode}" if proc.returncode else ""
                    return proc.returncode
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    _kill_compiler(proc)
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=10)
                    return 1
                finally:
                    if canceller is not None:
                        canceller.untrack()

            latex = ["pdflatex", "-interaction=nonstopmode",
                     "-halt-on-error", "main.tex"]
            if secure:
                latex.insert(1, "-no-shell-escape")
            if _run(latex) != 0:
                return False, 0, diagnosis("pdflatex failed")
            backend = (_biblatex_backend(tex) if (bib_text and biblatex_names)
                       else "bibtex" if (bib_text and bib_names) else None)
            if backend is not None:
                if not shutil.which(backend):
                    return False, 0, (f"bibliography backend {backend!r} "
                                      "not installed")
                timeout = 120 if backend == "biber" else 60
                if _run([backend, "main"], timeout=timeout) != 0:
                    return False, 0, diagnosis(f"{backend} failed")
                if _run(latex) != 0:
                    return False, 0, diagnosis("pdflatex failed")
            if _run(latex) != 0:
                return False, 0, diagnosis("pdflatex failed")
            pdf = work / "main.pdf"
            if not pdf.exists():
                return False, 0, diagnosis("pdflatex produced no PDF")
            if bbl_output is not None and (work / "main.bbl").exists():
                shutil.copyfile(work / "main.bbl", bbl_output)
            return True, _count_pdf_pages(pdf, last_stdout), ""
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Seats: one prompt, one call, raw text back. The full user message is
# assembled by the composite (from the approved templates) and passed in
# whole, so the framework's ``USER_PROMPT.format`` templating is bypassed
# on purpose — .format would choke on LaTeX braces.


class _SeatInputs(BaseModel):
    prompt: str


class _SeatOutputs(BaseModel):
    text: str


class _OneShotSeat(APICallAgent):
    execution_mode: ClassVar[str] = "agent"
    # Default to max effort; xhigh remains available as an override
    # (models/openai/gpt-56-sol-xhigh).
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-56-sol-max"
    Inputs: ClassVar[type[BaseModel]] = _SeatInputs
    Outputs: ClassVar[type[BaseModel]] = _SeatOutputs

    # Set per-instance by WriteupLoop._call_seat: which chain stage this
    # seat belongs to. Mixed into the resume-cache key (and nothing else),
    # so two stages running the same class on the same document sample
    # independently instead of replaying each other (see
    # WriteupLoop._stage_tag).
    stage: str = ""

    # Set per-instance by WriteupLoop._call_seat: the node's remaining
    # wallclock. Enforced INSIDE the API client so a timeout cancels the
    # background response server-side and an agent.error event flows. NOT
    # with cost accounted: the client raises before registering the
    # cancelled call's usage — the accepted gap in the module docstring.
    wallclock_cap_s: float | None = None
    completed_reply: str | None = None

    def _on_response(self, raw_text: str, inp: BaseModel) -> None:
        self.completed_reply = raw_text

    def _coerce_output(self, value: Any) -> BaseModel:
        out = super()._coerce_output(value)
        # Cache-hit logging may fail too; retain the validated cached reply.
        self.completed_reply = out.text
        return out

    def _cache_key(self, inp: BaseModel) -> str:
        key = super()._cache_key(inp)
        if not self.stage:
            return key
        return hashlib.sha256(
            f"{self.stage}\x00{key}".encode()).hexdigest()

    def render_messages(self, inp: BaseModel) -> list[Message]:
        # The template supplies the complete user turn; no extra system prompt.
        return [{"role": "user", "content": inp.prompt}]

    def parse_output(self, raw_text: str, inp: BaseModel) -> BaseModel:
        return _SeatOutputs(text=raw_text or "")

    def extra_client_kwargs(self) -> dict[str, Any]:
        # Server-side web search (runs at the provider, needs no
        # container egress beyond the API itself). David 2026-08-30:
        # all three seats get it — trust the model, don't handicap it.
        # APIClient.tools takes (local_func, tool_desc) pairs; func=None
        # means no local execution — the desc goes straight into the
        # Responses API payload (api_client.py:2347).
        kwargs: dict[str, Any] = {"tools": [(None, {"type": "web_search"})]}
        if self.wallclock_cap_s is not None:
            cap = float(self.wallclock_cap_s)
            kwargs["max_wallclock_per_call_s"] = cap
            # Also cap the SDK per-request timeout: without this a single
            # responses.create/retrieve can hang to the config's 11400s,
            # sailing past the polling-loop deadline (codex r3 #1).
            kwargs["timeout"] = cap
        return kwargs


class RewriteSeat(_OneShotSeat):
    description = "Guide-driven one-shot rewrite of a mathematics write-up."


class ColdRefereeSeat(_OneShotSeat):
    description = "Independent minimal-prompt correctness check of a write-up."


class RepairSeat(_OneShotSeat):
    description = "Minimal targeted fixes to a write-up from a referee report."


# --------------------------------------------------------------------------


class WriteupLoopInputs(BaseModel):
    document_text: str
    bib_text: str | None = Field(
        default=None,
        description="references.bib content, used only for the compile gate")
    research_notes_text: str | None = Field(
        default=None,
        description="research_notes.tex content, shown to the rewriter "
                    "as background when present")
    page_limit: int | None = Field(
        default=None,
        description="page cap enforced by the gate when set; None (the "
                    "general-use default) gates on compilation only. "
                    "Use writing_constraints to also communicate the cap "
                    "to rewrite/repair seats; batch 3's limit is 16.")
    rounds: int = Field(default=2, ge=0)
    writing_constraints: str | None = Field(
        default=None,
        description="Optional submission requirements for rewrite and repair; omitted in the general-use workflow.")


class GateResult(BaseModel):
    ok: bool
    compiled: bool
    pages: int
    note: str = ""  # short diagnosis, surfaced in the step record


class StepRecord(BaseModel):
    step: str
    round: int | None = None
    ok: bool | None = None
    detail: str = ""


class WriteupLoopOutputs(BaseModel):
    document_text: str  # never blank when the input was nonblank
    shipped: str        # decision-trail reason, mirrors the standalone loop
    improved: bool      # False iff document_text is the input unchanged
    steps: list[StepRecord] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    error: str | None = None
    usd_total: float = 0.0


@dataclass
class _WriteupInvocation:
    deadline: float = 0.0
    discarded_reply: tuple[str, str] | None = None


class WriteupLoop(Agent):
    """Composite node: rewrite -> (cold referee -> repair) x rounds."""

    description = ("Rewrite a correct-but-dense write-up per the writing "
                   "guide, then cold-referee and repair; always ships a "
                   "document — the original input as the fallback.")
    PALETTE = {
        "id": "writeup_loop",
        "label": "Writeup Loop",
        "group": "Proof Work",
        "description": ("Guide-driven rewrite of a finished write-up, with "
                        "cold-referee/repair rounds; ships the best gated "
                        "version, the untouched input on total failure."),
        "keywords": "writeup rewrite polish referee repair exposition",
    }
    execution_mode: ClassVar[str] = "workflow"
    cache_enabled: ClassVar[bool] = False
    default_budget: ClassVar[BudgetSpec | None] = BudgetSpec(
        max_usd=20.0, max_wallclock_s=7200)

    # Floor below which no further seat call is attempted, and the
    # backstop slack the outer wait_for allows the client's own
    # deadline before declaring the client hung.
    _MIN_SEAT_SECONDS = 120.0
    _WAITFOR_SLACK_SECONDS = 300.0

    # ---- seat implementation (component config; see module docstring) ----
    # These are plain class attributes on purpose: Agent's component-config
    # application only assigns keys that already exist on the type, so a
    # typo in a preset stays inert instead of silently configuring nothing.
    # "api" is the default so no existing caller changes behavior.
    seat: ClassVar[str] = "api"
    cli_model: ClassVar[str] = DEFAULT_MODEL
    cli_reasoning_effort: ClassVar[str] = DEFAULT_REASONING_EFFORT
    cli_codex_bin: ClassVar[str] = DEFAULT_CODEX_BIN
    cli_web_search: ClassVar[bool] = True
    # Default True: keeps ~/.codex/config.toml's personality directive out
    # of a frozen-prompt writing task, and pins the seat to the ChatGPT
    # login so codex-billing's API-provider switch cannot make it billable.
    cli_ignore_user_config: ClassVar[bool] = True

    Inputs: ClassVar[type[BaseModel]] = WriteupLoopInputs
    Outputs: ClassVar[type[BaseModel]] = WriteupLoopOutputs

    # ---- seams (monkeypatchable in control-flow tests) ----

    def _invocation_var(self) -> ContextVar[_WriteupInvocation | None]:
        var = getattr(self, "_invocation_context", None)
        if var is None:
            var = ContextVar("writeup_invocation", default=None)
            self._invocation_context = var
        return var

    @property
    def _deadline(self) -> float:
        invocation = self._invocation_var().get()
        if invocation is None:
            raise RuntimeError("writeup invocation has not started")
        return invocation.deadline

    @_deadline.setter
    def _deadline(self, value: float) -> None:
        var = self._invocation_var()
        invocation = var.get()
        if invocation is None:
            var.set(_WriteupInvocation(deadline=value))
        else:
            invocation.deadline = value

    @property
    def _discarded_reply(self) -> tuple[str, str] | None:
        invocation = self._invocation_var().get()
        return invocation.discarded_reply if invocation is not None else None

    def _remaining_wallclock(self) -> float:
        """Minimum remaining wallclock across the whole budget-scope
        chain — a node entering with 10s left in its parent's budget
        must not grant a seat two hours."""
        candidates: list[float] = []
        try:
            for node in self.tracker.chain():
                spec = getattr(node, "spec", None)
                limit = (getattr(spec, "max_wallclock_s", None)
                         if spec is not None else None)
                if limit:
                    used = float(node.counters.wallclock_s())
                    candidates.append(float(limit) - used)
        except Exception:
            pass
        return min(candidates) if candidates else 7200.0

    def _note_discarded_reply(self, name: str, text: str) -> None:
        """Remember a reply that arrived and is about to be discarded.

        Written SYNCHRONOUSLY, the moment the text is in hand and before
        any await — the same rule as registering an UNABLE before the gate
        (codex r6 #2). The failure that discards a reply is accounted for
        over several awaits (usage, events, transcripts), and a
        cancellation delivered in one of them replaces the seat's
        exception with CancelledError: the repair handler never runs, and
        a genuine UNABLE: would be lost on the way out (codex r12 #4).
        ``_ship`` reads this on every exit path instead."""
        invocation = self._invocation_var().get()
        if invocation is not None:
            invocation.discarded_reply = (name, str(text))

    def _absorb_discarded_unable(self, state: dict[str, Any]) -> None:
        """Apply a discarded repair's UNABLE: to the state, if the seat
        that produced it is the repair the loop is waiting on. Best
        effort and synchronous: ``_ship`` must never raise."""
        try:
            invocation = self._invocation_var().get()
            if invocation is None:
                return
            name, text = invocation.discarded_reply or ("", "")
            if not name or name != state.get("awaiting_repair"):
                return
            invocation.discarded_reply = None
            _, declaration = _extract_tex(text)
            if _declares(declaration, "UNABLE:"):
                state["standing_unable"] = True
        except Exception:  # noqa: BLE001
            pass

    def _stage_tag(self) -> str:
        """This stage's identity, mixed into its seats' resume-cache keys.

        The framework keys that cache on (agent class, agent NAME, config,
        inputs), and every stage of a chain runs the same class with the
        same seat names — so when a stage ships its input unchanged (its
        rewrite failed the gate twice, say) the next stage hands its seats
        an identical prompt, hits the cache, and replays the candidates
        that just failed. The re-roll that is the loop's whole defence
        against a bad draw then re-serves the bad draw, and a four-stage
        chain makes four seat calls instead of eight (codex r10 #3).

        The node's own name is the stage identity; the shipped chain
        preset gives each stage a distinct one (``writeup_loop_1``..
        ``_4``). The FIRST claim of a name keeps the name alone, so a
        resume of an ordinary run still hits the cache exactly as before.

        A name claimed a second time within a run — stages left unnamed
        are all ``WriteupLoop`` — cannot be told apart from the first by
        anything the framework records: the DAG's node ids, which would
        distinguish them, never reach the agent. Such a stage is given an
        identity unique to the instance, which no cache entry can match:
        it samples afresh and never replays, at the price of not resuming.

        That is a mitigation, not a fix, and the limit is worth stating
        plainly: which of two same-named stages gets the bare identity is
        decided by claim ORDER, so a resume that runs them in a different
        order hands the second one the first one's cached candidates
        (codex r11 #5, r12 #5). Nothing available to this node closes
        that. The remedy is one line of config — a distinct ``name:`` per
        stage, as the chain preset has and its test enforces — so a
        duplicate raises a ``config.warn`` where the run's events are
        read.

        The tag is a JSON pair, not a concatenation, so that
        (``polish``, second claim) cannot encode to the same string as a
        stage genuinely named ``polish#1`` (codex r11 #4). It deliberately
        does NOT go into the seat's ``name``:
        ``model_for``/``component_config_for`` look overrides up by
        instance name, so renaming the seats would silently drop a
        configured ``rewrite-a1`` override (codex r11 #6).
        """
        tag = getattr(self, "_stage_tag_value", None)
        if tag is None:
            name = str(getattr(self, "name", "") or "")
            run_id = str(getattr(getattr(self, "ctx", None), "run_id", "") or "")
            discriminator = ""
            if name:
                key = (run_id, name)
                if key in _STAGE_CLAIMED or (
                        len(_STAGE_CLAIMED) >= _STAGE_CLAIMED_MAX):
                    # Claimed already, or the registry is full. The register
                    # is never pruned: dropping another run's claims because
                    # its id differs assumes it has finished, and a run
                    # still in flight would then reissue a bare tag it has
                    # already used (codex r12 #6). Refusing to issue more
                    # bare tags costs a resume; forgetting costs a replay.
                    discriminator = uuid.uuid4().hex
                    self._stage_duplicate = key in _STAGE_CLAIMED
                else:
                    _STAGE_CLAIMED.add(key)
            tag = json.dumps([name, discriminator])
            self._stage_tag_value = tag
        return tag

    async def _call_seat(self, seat_cls: type[_OneShotSeat], name: str,
                         prompt: str) -> str:
        """One seat call, bounded by the node's remaining wallclock. The
        bound is enforced inside the API client (server-side cancel; an
        agent.error event flows, though the cancelled call's spend is
        NOT registered — the accepted accounting gap in the module
        docs); the outer wait_for is a hung-client backstop with slack.
        Timeout surfaces as an ordinary seat failure so the caller's
        degradation path decides. BudgetExhausted is never SWALLOWED
        below run()'s dedicated handler (the loop re-raises it after
        reading any completed declaration off it)."""
        await self._budget_check()
        remaining = self._deadline - time.monotonic()
        if remaining < self._MIN_SEAT_SECONDS:
            raise TimeoutError("node wallclock exhausted")
        if str(getattr(self, "seat", "api")).strip().lower() == "cli":
            return await self._call_cli_seat(name, prompt, remaining)
        seat = seat_cls(self.ctx, name=name,
                        parent_budget_scope=self.tracker.scope)
        seat.stage = self._stage_tag()
        seat.wallclock_cap_s = remaining
        try:
            out = await asyncio.wait_for(
                seat(prompt=prompt),
                timeout=remaining + self._WAITFOR_SLACK_SECONDS)
        except (asyncio.CancelledError, TimeoutError):
            # APICallAgent runs the call in a worker thread whose poll loop
            # stops only on APIClient.terminate(); cancelling this await
            # leaves the background response live, and billing, to its own
            # deadline (codex r7 #2). terminate() is synchronous, so it is
            # safe here on the cancellation path, which must not await.
            client = getattr(seat, "_client", None)
            if client is not None:
                with contextlib.suppress(Exception):
                    client.terminate()
            raise
        finally:
            # wait_for runs the seat in a child task. Copy its synchronous
            # latch before the loop's failure/cancellation handlers can ship.
            reply = getattr(seat, "completed_reply", None)
            if reply is not None:
                self._note_discarded_reply(name, reply)
        return out.text

    async def _budget_check(self) -> None:
        """Ask the tracker whether there is headroom left. Enforcement is
        cooperative (budget.py): add_usd/add_tokens only accumulate, and
        APICallAgent is the only caller of check() — so without this the
        CLI seat, which never goes through APICallAgent, runs on past an
        exhausted max_tokens, the documented subscription backstop
        (configs/workflows/instructions.md) (codex r7 #4). Warnings are
        emitted rather than dropped because check() latches ``warned`` per
        scope: swallowing them here would silence the API seat's own
        budget.warn events."""
        for scope, kind, used, limit in self.tracker.check():
            try:
                await self.events.emit(
                    "budget.warn",
                    {"scope": scope, "kind": kind, "used": used,
                     "limit": limit})
            except Exception:  # noqa: BLE001 — an event-log write is not a
                pass          # reason to fail a budget-clean seat call

    async def _call_cli_seat(self, name: str, prompt: str,
                             remaining: float) -> str:
        """Subscription seat: one blocking ``codex exec`` turn. The bound
        is enforced by the child-process kill in ``run_codex_seat`` (no
        server-side background call exists to cancel), and any failure —
        missing binary, nonzero exit, empty final message, timeout —
        raises, so the caller's existing degradation path decides. Usage
        is recorded here rather than by an APICallAgent; dollars stay 0,
        so the node's ``max_usd`` gate is never tripped by a $0 run, and
        the token backstop is enforced by the post-call ``_budget_check``
        below (``add_tokens`` alone only accumulates)."""
        # Each spawn is a tool call, charged the way CLIAgent charges its
        # own (kinds/cli.py) so a shared max_tool_calls constrains this
        # seat too — and charged BEFORE the spawn, so a failed attempt
        # still counts against it (codex r8 #4).
        self.tracker.add_tool_call()
        try:
            result = await run_codex_seat(
                prompt,
                model=self.cli_model,
                reasoning_effort=self.cli_reasoning_effort,
                codex_bin=self.cli_codex_bin,
                web_search=bool(self.cli_web_search),
                ignore_user_config=bool(self.cli_ignore_user_config),
                timeout_s=remaining,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — record, then degrade
            # A turn that reported usage and then failed still spent real
            # subscription throughput; charge it, or a failing seat is free
            # and max_tokens never stops the retries (codex r9 #2).
            partial = getattr(e, "partial", None)
            if partial is not None:
                # Before any await below: see _note_discarded_reply.
                self._note_discarded_reply(name, getattr(partial, "text", ""))
                # Its reply too, redacted: a turn refused for carrying a
                # credential leaves nothing else to look at, and the
                # failure message deliberately does not quote it.
                self._persist_cli_seat(name, prompt, partial)
                await self._record_cli_usage(name, partial)
            await self._record_cli_failure(name, prompt, e)
            raise
        # Registered before the accounting awaits, not only on the failure
        # path: a cancellation delivered while this turn's usage event is
        # being emitted discards a reply that HAS arrived, and a repair's
        # UNABLE: in it would go with it (codex r13 #5). Harmless when the
        # call goes on to succeed — the loop clears ``awaiting_repair`` the
        # moment it has the reply in hand, so _ship absorbs nothing.
        self._note_discarded_reply(name, result.text)
        self._persist_cli_seat(name, prompt, result)
        await self._record_cli_usage(name, result)
        # Checked at the LOOP boundary, not only in the transcripts: this
        # text becomes the document and the next seat's prompt, so a
        # credential the turn echoed would otherwise reach the referee and
        # could ship inside the document itself (codex r10 #1).
        # ``run_codex_seat`` already refuses such a reply at the source;
        # this is the seam a substituted seat implementation cannot slip
        # past. A reply that needs scrubbing is failed rather than
        # returned scrubbed, because a substitution inside the document's
        # mathematics is one the compile gate cannot see (codex r11 #5).
        text = result.text
        if redact_seat_text(text) != text or looks_like_credential(text):
            err = CodexSeatError(
                "codex seat reply contained a credential; candidate "
                "discarded (see the seat transcript)")
            # Scrubbed, and carried both ways: on the exception for the
            # ordinary path, and on the node for the path where a
            # cancellation eats the exception (codex r11 #2, r12 #4).
            scrubbed = redact_seat_text(text)
            self._note_discarded_reply(name, scrubbed)
            err.completed_output = _SeatOutputs(text=scrubbed)
            raise err
        try:
            await self._budget_check()
        except BudgetExhausted as e:
            # Same reason api_call.py attaches its own: this turn has
            # already completed and been charged, and a repair's UNABLE:
            # must survive the crossing (codex r7 #3).
            e.completed_output = _SeatOutputs(text=text)
            raise
        return text

    async def _record_cli_failure(self, name: str, prompt: str,
                                  exc: BaseException) -> None:
        """A failed CLI seat left nothing behind: no APICallAgent artifact,
        no agent.error of its own, and (before r11) no transcript either —
        so only the exception TYPE reached the step record and the run
        directory kept no reason for the failure (codex r8 #6).
        Write one, and emit the event Agent.__call__ would have emitted for
        an API seat. The message quotes a codex log tail, which can echo a
        credential, so it goes through redact_seat_text first. Best effort
        throughout: a failed seat must still degrade, never raise here."""
        detail = redact_seat_text(f"{type(exc).__name__}: {exc}")
        try:
            out = Path(self.workdir) / "cli-seats"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.prompt.txt").write_text(prompt, encoding="utf-8")
            (out / f"{name}.error.txt").write_text(detail, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        try:
            emit = self.events.emit(
                "agent.error",
                {"type": type(exc).__name__, "msg": detail, "seat": name})
        except Exception:  # noqa: BLE001
            return
        if asyncio.iscoroutine(emit):
            with contextlib.suppress(Exception):
                await emit

    def _persist_cli_seat(self, name: str, prompt: str, result: Any) -> None:
        """The API seat is an APICallAgent, so the framework writes its
        prompt and reply into the run directory. The CLI seat is a plain
        subprocess and gets no such artifact, which would leave a bad
        polish undebuggable after the fact — so write the same material
        by hand. Best effort: never a reason to fail a seat call.

        Redacted like the failure path: a SUCCESSFUL turn can echo a
        credential too — through a diagnostic or a tool result — and these
        artifacts get read and shared (codex r9 #1)."""
        try:
            out = Path(self.workdir) / "cli-seats"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.prompt.txt").write_text(prompt, encoding="utf-8")
            (out / f"{name}.reply.txt").write_text(
                redact_seat_text(result.text), encoding="utf-8")
            if result.log_tail:
                (out / f"{name}.log-tail.txt").write_text(
                    redact_seat_text(result.log_tail), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    async def _record_cli_usage(self, name: str, result: Any) -> None:
        """Best-effort accounting; never the reason a seat call fails.
        Tokens are the real subscription limit, so they are charged to the
        tracker; usd stays 0. Charging is not enforcement — the
        ``max_tokens`` gate is applied by ``_budget_check``, which the
        caller runs immediately after this."""
        try:
            tokens = int(result.metered_tokens)
        except Exception:  # noqa: BLE001
            return
        try:
            self.tracker.add_tokens(tokens)
        except Exception:  # noqa: BLE001
            pass
        try:
            emit = self.events.emit(
                "model.call",
                {
                    "model": self.cli_model,
                    "seat": name,
                    "in_tokens": result.usage.input_tokens,
                    "cached_in_tokens": result.usage.cached_input_tokens,
                    "out_tokens": result.usage.output_tokens,
                    "reasoning_out_tokens": result.usage.reasoning_output_tokens,
                    "metered_tokens": tokens,
                    "cost_usd": 0.0,
                    "duration_s": round(result.duration_s, 1),
                    "via": "codex_exec_json",
                    "billing": "chatgpt_subscription",
                },
            )
        except Exception:  # noqa: BLE001
            return
        if asyncio.iscoroutine(emit):
            try:
                await emit
            except Exception:  # noqa: BLE001 — an event-log write is not
                pass          # a reason to fail a completed seat call

    def _gate(self, tex: str, inp: WriteupLoopInputs,
              pass_timeout: int = 300,
              deadline: float | None = None,
              canceller: _GateCanceller | None = None) -> GateResult:
        compiled, pages, note = _compile_raw(tex, inp.bib_text,
                                             pass_timeout=pass_timeout,
                                             deadline=deadline,
                                             canceller=canceller)
        ok = compiled and (inp.page_limit is None or pages <= inp.page_limit)
        return GateResult(ok=ok, compiled=compiled, pages=pages, note=note)

    async def _gate_async(self, tex: str, inp: WriteupLoopInputs) -> GateResult:
        # pdflatex can spin for minutes on pathological input; keep the
        # event loop (and cancellation delivery) alive meanwhile. With
        # the deadline (nearly) exhausted the gate is not started at
        # all: an ungated candidate is never adopted, so this reads as
        # a gate failure and best-so-far ships (codex r3 #4).
        remaining = self._deadline - time.monotonic()
        if remaining < 60.0:
            return GateResult(ok=False, compiled=False, pages=0)
        # Up to ~4 tool passes per gate; each is capped so the whole
        # gate cannot outlive the deadline by more than one pass.
        pass_timeout = int(min(300.0, max(30.0, remaining / 4)))
        canceller = _GateCanceller()
        try:
            return await asyncio.to_thread(self._gate, tex, inp, pass_timeout,
                                           self._deadline, canceller)
        except asyncio.CancelledError:
            # The thread is not cancellable; tell it to stop and kill the
            # pass in flight, or it compiles on after run() has shipped and
            # asyncio.run then blocks on executor shutdown (codex r9 #3).
            canceller.cancel()
            raise

    # ---- the loop ----

    async def run(self, inp: WriteupLoopInputs) -> WriteupLoopOutputs:
        # DAG map_chain reuses instances concurrently. Child seat tasks
        # inherit this invocation object, never another document's state.
        var = self._invocation_var()
        token = var.set(_WriteupInvocation(
            deadline=time.monotonic() + max(0.0, self._remaining_wallclock())))
        state: dict[str, Any] = {"document": None, "steps": [], "error": None}
        try:
            return await self._run_inner(inp, state)
        except BudgetExhausted as e:
            return self._ship(
                inp, state,
                f"best-effort: budget exhausted (scope={e.scope})")
        except asyncio.CancelledError:
            # Deliberate contract deviation from asyncio convention: a
            # harness timeout must still yield a document, so the node
            # completes with best-effort outputs instead of propagating
            # the cancellation. _ship is synchronous — no further awaits.
            state["error"] = "cancelled"
            return self._ship(inp, state, "best-effort: cancelled")
        except Exception as e:  # noqa: BLE001 — the node must never raise
            name = type(e).__name__
            try:
                state["error"] = f"{name}: {e}"
            except Exception:  # a __str__ that itself raises
                state["error"] = name
            return self._ship(inp, state, f"best-effort after error: {name}")
        finally:
            var.reset(token)

    def _ship(self, inp: WriteupLoopInputs, state: dict[str, Any],
              reason: str) -> WriteupLoopOutputs:
        # Every exit path, including the ones that bypass the loop's own
        # handlers (cancellation, an unexpected error): a repair whose
        # reply was received and then discarded still declared what it
        # declared (codex r12 #4).
        self._absorb_discarded_unable(state)
        document = state.get("document")
        if state.get("standing_unable"):
            # An unresolved UNABLE: on a critical error taints the
            # current document on EVERY exit path — including budget
            # exhaustion and cancellation, which bypass the loop's own
            # catastrophe check (codex r5 #2). Only the original ships.
            document = None
            reason += " [unresolved UNABLE: original shipped]"
        if not (isinstance(document, str) and document.strip()):
            document = inp.document_text
        try:
            usd = round(float(self.tracker.counters.usd), 4)
        except Exception:
            usd = 0.0
        try:
            flags = _flag_lines(document)
            steps = list(state.get("steps") or [])
        except Exception:
            flags, steps = [], []
        return WriteupLoopOutputs(
            document_text=document,
            shipped=reason,
            improved=document != inp.document_text,
            steps=steps,
            flags=flags,
            error=state.get("error"),
            usd_total=usd,
        )

    async def _run_inner(self, inp: WriteupLoopInputs,
                         state: dict[str, Any]) -> WriteupLoopOutputs:
        steps: list[StepRecord] = state["steps"]

        self._stage_tag()
        if getattr(self, "_stage_duplicate", False):
            # The node cannot give two stages sharing one name a stable
            # identity — it can only stop them replaying each other, at
            # the cost of this stage's resume. Say so where a run's
            # events are read, since the remedy is one line of config.
            with contextlib.suppress(Exception):
                await self.events.emit(
                    "config.warn",
                    {"node": str(getattr(self, "name", "")),
                     "msg": "another stage of this run already uses this "
                            "node name; give each chain stage a distinct "
                            "name: or this stage cannot resume"})

        if not inp.document_text.strip():
            return self._ship(inp, state, "original: input document is blank")

        # 1. Rewrite; gate; one re-roll (distinct seat name => fresh
        # sample, not a resume-cache replay of the failed attempt).
        gates_run = 0
        for attempt in (1, 2):
            name = f"rewrite-a{attempt}"
            try:
                raw = await self._call_seat(
                    RewriteSeat, name,
                    _assemble("rewrite-wrapper.txt",
                              {"document": inp.document_text},
                              research_notes=inp.research_notes_text,
                              writing_constraints=inp.writing_constraints))
            except BudgetExhausted:
                raise
            except Exception as e:  # noqa: BLE001 — degrade, don't die
                steps.append(StepRecord(step=name, ok=False,
                                        detail=f"seat error: {type(e).__name__}"))
                continue
            tex, _ = _extract_tex(raw)
            tex = _defuse_proof_sectioning(tex)
            gate = await self._gate_async(tex, inp)
            gates_run += 1
            steps.append(StepRecord(
                step=name, ok=gate.ok,
                detail=_gate_detail(gate)))
            if gate.ok:
                state["document"] = tex
                break
        if state["document"] is None:
            reason = ("original: rewrite failed the gate"
                      f" {'twice' if gates_run >= 2 else 'once'}"
                      if gates_run else
                      "original: rewrite seats unavailable")
            return self._ship(inp, state, reason)

        # 2. (referee, repair) rounds. standing_unable tracks an
        # unresolved catastrophe signal: SET by any completed repair
        # declaring UNABLE:, CLEARED only by a completed clean referee
        # verdict (the ship paths below) or by a later ADOPTED repair
        # declaring no UNABLE. Incomplete rounds — a referee or repair
        # that errors, or a repair whose gate fails without its own
        # UNABLE — never clear it (codex r4 #1/#2).
        state["standing_unable"] = False
        for round_no in range(1, inp.rounds + 1):
            try:
                report = await self._call_seat(
                    ColdRefereeSeat, f"referee-r{round_no}",
                    _assemble("cold-referee.txt",
                              {"document": state["document"]}))
            except BudgetExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                steps.append(StepRecord(
                    step=f"referee-r{round_no}", round=round_no, ok=False,
                    detail=f"seat error: {type(e).__name__}"))
                if state["standing_unable"]:
                    # _ship enforces the fallback; the reason names it.
                    return self._ship(
                        inp, state,
                        "original: referee unavailable with an unresolved "
                        "UNABLE on a critical error")
                # No referee => no basis to edit further. Trust the
                # rewriter (the room's standing ethos) and ship what we
                # have, marked as unchecked this round.
                return self._ship(
                    inp, state,
                    f"polished: referee unavailable in round {round_no}, "
                    f"shipped without further checks")
            clean = _verdict_line(report, "NO ERRORS")
            found = _verdict_line(report, "ERRORS FOUND")
            no_errors = clean and not found
            if no_errors:
                # A completed clean referee verdict is the authority
                # that supersedes any standing declaration — cleared
                # NOW, so a later polish failure or budget exit cannot
                # resurrect it (codex r6 #4).
                state["standing_unable"] = False
            steps.append(StepRecord(
                step=f"referee-r{round_no}", round=round_no, ok=True,
                detail=("NO ERRORS" if no_errors else
                        "ERRORS FOUND" if found else
                        "verdict unparsed; treated as ERRORS FOUND")))

            repair_name = (f"repair-r{round_no}" if not no_errors
                           else f"polish-r{round_no}")
            # Named BEFORE the call: if this reply is received and then
            # discarded, _ship must know whose declaration it was, on
            # every exit path including the ones that never come back
            # here (codex r12 #4). A polish is not a catastrophe carrier
            # — a clean referee verdict is the authority — so it is not
            # watched.
            state["awaiting_repair"] = repair_name if not no_errors else None
            try:
                raw = await self._call_seat(
                    RepairSeat, repair_name,
                    _assemble("repair.txt",
                              {"document": state["document"],
                               "referee findings": report},
                              writing_constraints=inp.writing_constraints))
            except BudgetExhausted as e:
                # The post-call check raises AFTER the reply is back, so a
                # repair can complete with UNABLE: and still land here.
                # Inspect it before the exit, or a budget crossing silently
                # bypasses the catastrophe fallback (codex r7 #3).
                if not no_errors:
                    _, done_decl = _extract_tex(_completed_reply_text(e))
                    if _declares(done_decl, "UNABLE:"):
                        state["standing_unable"] = True
                        steps.append(StepRecord(
                            step=repair_name, round=round_no, ok=False,
                            detail="budget exhausted after a completed "
                                   "UNABLE: declaration"))
                raise
            except Exception as e:  # noqa: BLE001
                detail = f"seat error: {type(e).__name__}"
                # A seat can fail with the model's reply already in hand —
                # a turn refused for carrying a credential, or one that
                # exited nonzero after writing its final message. The
                # candidate is discarded either way, but a repair's
                # UNABLE: inside it is a completed declaration and must
                # still taint the document (codex r11 #2).
                if not no_errors:
                    _, done_decl = _extract_tex(_completed_reply_text(e))
                    if _declares(done_decl, "UNABLE:"):
                        state["standing_unable"] = True
                        detail += " after a completed UNABLE: declaration"
                steps.append(StepRecord(
                    step=repair_name, round=round_no, ok=False,
                    detail=detail))
                if no_errors:
                    return self._ship(
                        inp, state,
                        f"round {round_no} referee: NO ERRORS "
                        f"(polish step failed, shipped without it)")
                # NB: standing_unable is otherwise deliberately untouched —
                # a failed repair seat must not CLEAR an earlier round's
                # surviving UNABLE: catastrophe signal (codex r3 #10).
                continue
            # The reply is in hand and is not being discarded: nothing
            # left for _ship to absorb on its behalf.
            state["awaiting_repair"] = None
            tex, declaration = _extract_tex(raw)
            tex = _defuse_proof_sectioning(tex)
            unable = _declares(declaration, "UNABLE:")
            if unable and not no_errors:
                # Registered BEFORE the gate await: a cancellation while
                # gating must not ship over a received UNABLE (codex r6
                # #2). After a clean verdict, a contradictory polish
                # UNABLE does not taint — the referee is the authority.
                state["standing_unable"] = True
            gate = await self._gate_async(tex, inp)
            steps.append(StepRecord(
                step=repair_name, round=round_no, ok=gate.ok,
                detail=f"{_gate_detail(gate)} unable={unable}"))
            adopted = False
            if gate.ok and not (no_errors and unable):
                # A polish declaring UNABLE contradicts its clean report;
                # trust the referee's verdict and keep the cleared text.
                state["document"] = tex
                adopted = True
            if adopted and not unable:
                # A completed, adopted repair with no UNABLE resolves
                # any earlier declaration; a discarded one resolves
                # nothing (codex r4 #2).
                state["standing_unable"] = False
            if no_errors:
                return self._ship(
                    inp, state,
                    f"round {round_no} referee: NO ERRORS, "
                    + ("polish applied" if adopted else "polish discarded"))

        # 3. Catastrophe check: an unresolved UNABLE at the round cap.
        if state["standing_unable"]:
            return self._ship(
                inp, state,
                "original: final repair declared UNABLE on a critical error")
        return self._ship(inp, state,
                          "polished: round cap reached, no catastrophe")
