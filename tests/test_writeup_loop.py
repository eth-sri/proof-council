"""Control-flow and parsing tests for WriteupLoop.

Seats are monkeypatched at the `_call_seat` seam; the gate is patched in
control-flow tests but exercised for real (tiny pdflatex compiles) in
TestGate, since the 2026-08-31 codex review showed the previous gate
accepted text it would never ship working. Live behavior is validated
separately via the writeup_loop workflow preset."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.writeup_loop import (  # noqa: E402
    GateResult,
    RewriteSeat,
    _OneShotSeat,
    _STAGE_CLAIMED,
    _SeatInputs,
    _SeatOutputs,
    _BIBLATEX_RE,
    _GateCanceller,
    _biblatex_backend,
    _defuse_proof_sectioning,
    WriteupLoop,
    WriteupLoopInputs,
    _assemble,
    _compile_raw,
    _declares,
    _extract_tex,
    _flag_lines,
    _verdict_line,
)
from proofstack.agents.writeup_codex_seat import (  # noqa: E402
    BLOCKED_AUTH_ENVS,
    SEAT_MARKER_ENV,
    auth_secrets,
    kill_seat_survivors,
    looks_like_credential,
    scrub_credentials,
    seat_survivors,
    CodexSeatError,
    CodexSeatResult,
    _kill_process_group,
    assert_subscription_login,
    build_codex_cmd,
    child_env,
    redact_seat_text,
    resolve_codex_bin,
    run_codex_seat,
)
from proofstack.budget import BudgetExhausted, BudgetSpec  # noqa: E402
from proofstack.kinds.api_call import APICallAgent  # noqa: E402
from proofstack.cli_usage import CodexUsage  # noqa: E402


DOC = ("\\documentclass{article}\n\\begin{document}\nx\n\\end{document}\n")
GOOD_GATE = GateResult(ok=True, compiled=True, pages=3)
BAD_GATE = GateResult(ok=False, compiled=False, pages=0)
HAS_PDFLATEX = shutil.which("pdflatex") is not None


def _reap(proc):
    """Cleanup for a test-spawned child, whether or not it was killed."""
    if proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)


def make_loop(max_wallclock: float | None = None,
              spec: BudgetSpec | None = None) -> WriteupLoop:
    """Instance without framework wiring — bypass __init__ like the
    writeup-nodes tests do, but use the REAL BudgetTracker so the
    deadline chain is exercised, not stubbed (codex r3 #2)."""
    from proofstack.budget import BudgetTracker

    loop = WriteupLoop.__new__(WriteupLoop)
    if spec is None and max_wallclock is not None:
        spec = BudgetSpec(max_wallclock_s=max_wallclock)
    loop.tracker = BudgetTracker(scope="test", spec=spec)
    return loop


def run_loop(loop, seat_script, gates=None, rounds=2, doc="original text"):
    """Drive run() with scripted seat replies. seat_script is a list of
    either strings (seat raw text) or Exceptions (seat failure), consumed
    in call order. gates likewise, GateResult per gate call."""
    calls = {"seats": [], "gates": 0}

    async def fake_call_seat(seat_cls, name, prompt):
        calls["seats"].append(name)
        item = seat_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def fake_gate(tex, inp, *args):
        calls["gates"] += 1
        if gates:
            return gates.pop(0)
        return GOOD_GATE

    loop._call_seat = fake_call_seat
    loop._gate = fake_gate
    inp = WriteupLoopInputs(document_text=doc, rounds=rounds)
    out = asyncio.run(loop.run(inp))
    return out, calls


class TestHelpers(unittest.TestCase):
    def test_extract_fenced_with_trailing_declaration(self):
        raw = "preamble\n```latex\n" + DOC.strip() + "\n```\nFIXED: x\nUNABLE: y\n"
        tex, decl = _extract_tex(raw)
        self.assertIn("\\documentclass", tex)
        self.assertTrue(_declares(decl, "UNABLE:"))

    def test_extract_unfenced_ends_at_end_document(self):
        raw = DOC + "UNABLE: something\n"
        tex, decl = _extract_tex(raw)
        self.assertTrue(tex.rstrip().endswith("\\end{document}"))
        self.assertTrue(_declares(decl, "UNABLE:"))

    def test_extract_declaration_inside_fence(self):
        # Codex finding 8: a declaration left INSIDE the fence after
        # \end{document} compiles fine (TeX ignores it) but must still
        # reach the catastrophe logic.
        raw = "```latex\n" + DOC.strip() + "\nUNABLE: critical gap\n```\n"
        tex, decl = _extract_tex(raw)
        self.assertTrue(tex.rstrip().endswith("\\end{document}"))
        self.assertTrue(_declares(decl, "UNABLE:"))
        self.assertNotIn("UNABLE", tex)

    def test_verdict_is_exact_line_not_substring(self):
        # Codex finding 9: prose mentioning a sentinel must not count.
        self.assertFalse(_verdict_line(
            "I cannot certify NO ERRORS because Lemma 2 is false", "NO ERRORS"))
        self.assertTrue(_verdict_line("fine\nNO ERRORS\n", "NO ERRORS"))
        self.assertTrue(_verdict_line("**NO ERRORS**", "NO ERRORS"))

    def test_declares_requires_line_start(self):
        self.assertFalse(_declares("we were UNABLE: to proceed", "UNABLE:"))
        self.assertTrue(_declares("  UNABLE: the error", "UNABLE:"))

    def test_remaining_wallclock_uses_real_tracker_chain(self):
        loop = make_loop(max_wallclock=600.0)
        remaining = loop._remaining_wallclock()
        self.assertLess(remaining, 600.01)
        self.assertGreater(remaining, 590.0)

    def test_page_fallback_takes_last_anchored_match(self):
        # codex r3 #5: a \typeout-forged line BEFORE the real one must
        # lose to the genuine closing line.
        from proofstack.agents.writeup_loop import _LOG_PAGES_RE
        stdout = (b"blah\nOutput written on main.pdf (1 pages)\nmore\n"
                  b"Output written on main.pdf (13 pages, 1234 bytes).\n")
        self.assertEqual(_LOG_PAGES_RE.findall(stdout)[-1], b"13")

    def test_defusal_protects_verbatim_and_nested_braces(self):
        # codex r3 #7/#9
        tex = ("\\begin{proof}\\paragraph{Step \\textit{one}.}x\\end{proof}"
               "\\begin{verbatim}\\begin{proof}\\paragraph{demo}\\end{proof}"
               "\\end{verbatim}")
        fixed = _defuse_proof_sectioning(tex)
        self.assertIn("\\textbf{Step \\textit{one}.}", fixed)
        self.assertIn("\\begin{verbatim}\\begin{proof}\\paragraph{demo}", fixed)

    def test_defusal_leaves_numbered_sectioning_alone(self):
        # codex r3 #8: numbered sectioning has real semantics; leave it
        # to the gate.
        tex = "\\begin{proof}\\subsection{Case}\\end{proof}"
        self.assertEqual(_defuse_proof_sectioning(tex), tex)

    def test_notes_containing_document_placeholder_not_substituted(self):
        # codex r3 #14
        text = _assemble("rewrite-wrapper.txt", {"document": "REALDOC"},
                         research_notes="notes with literal {{document}}")
        self.assertIn("notes with literal {{document}}", text)
        self.assertEqual(text.count("REALDOC"), 1)

    def test_repair_seat_failure_keeps_prior_unable(self):
        # codex r3 #10: round-1 UNABLE survives a round-2 seat failure.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical gap\n",
             "still bad\nERRORS FOUND\n",
             RuntimeError("repair seat down")])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE on a critical error", out.shipped)

    def test_referee_failure_with_standing_unable_ships_original(self):
        # codex r4 #1: an adopted UNABLE repair followed by a referee
        # outage must ship the original, not the tainted document.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             RuntimeError("referee down")])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_garbage_second_repair_keeps_catastrophe(self):
        # codex r4 #2: a failed-gate round-2 repair with no declaration
        # must not clear round 1's UNABLE.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             "still bad\nERRORS FOUND\n",
             "```\nGARBAGE\n```\n"],
            gates=[GOOD_GATE, GOOD_GATE, BAD_GATE])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE on a critical error", out.shipped)

    def test_adopted_clean_repair_resolves_standing_unable(self):
        # The resolution rule: a later adopted repair with no UNABLE
        # clears the earlier declaration; cap ships polished.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             "better\nERRORS FOUND\n",
             "```\nREPAIRED2\n```\nFIXED: the critical error\n"])
        self.assertEqual(out.document_text, "REPAIRED2\n")
        self.assertIn("no catastrophe", out.shipped)

    def test_defusal_deep_nesting_and_spaced_verbatim(self):
        # codex r4 #5/#6
        tex = ("\\begin{proof}\\paragraph{Step \\textbf{a \\emph{b}}.}x\\end{proof}"
               "\\begin {verbatim}\\begin{proof}\\paragraph{demo}\\end{proof}"
               "\\end {verbatim}")
        fixed = _defuse_proof_sectioning(tex)
        self.assertIn("\\textbf{Step \\textbf{a \\emph{b}}.}", fixed)
        self.assertIn("\\begin {verbatim}\\begin{proof}\\paragraph{demo}", fixed)

    def test_fence_prefers_complete_over_truncated_draft(self):
        # codex r4 #4: a truncated draft fence (no \end{document}) must
        # lose to a later complete fence.
        truncated = "\\documentclass{article}\n\\begin{document}\ndraft cut off"
        raw = ("```latex\n" + truncated + "\n```\nfull version:\n```latex\n"
               + DOC.strip() + "\n```\n")
        tex, _ = _extract_tex(raw)
        self.assertIn("\\end{document}", tex)
        self.assertNotIn("draft cut off", tex)

    def test_declaration_only_repair_is_catastrophe(self):
        # codex r5 #1: a repair answering ONLY 'UNABLE: ...' (no
        # document) must still register the catastrophe signal.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "UNABLE: cannot fix this at all\n",
             "still bad\nERRORS FOUND\n",
             "UNABLE: still cannot\n"],
            gates=[GOOD_GATE, BAD_GATE, BAD_GATE])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE", out.shipped)

    def test_budget_exhaustion_after_unable_ships_original(self):
        # codex r5 #2: BudgetExhausted must not bypass a standing UNABLE.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             BudgetExhausted("run", "max_usd", 20.0, 21.0)])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_cancellation_after_unable_ships_original(self):
        # codex r5 #2, cancellation variant.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             asyncio.CancelledError()])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_extract_no_document_reply_is_all_declaration(self):
        tex, decl = _extract_tex("UNABLE: cannot fix\n")
        self.assertTrue(_declares(decl, "UNABLE:"))

    def test_fenced_declaration_only_reply_registers(self):
        # codex r6 #1
        tex, decl = _extract_tex("```\nUNABLE: cannot repair\n```\n")
        self.assertTrue(_declares(decl, "UNABLE:"))

    def test_cancel_during_gate_after_unable_ships_original(self):
        # codex r6 #2: UNABLE received, cancellation arrives while
        # gating — the flag must already be registered.
        loop = make_loop()
        script = ["```\nREWRITTEN\n```",
                  "bad\nERRORS FOUND\n",
                  "```\nREPAIRED\n```\nUNABLE: critical\n"]

        async def fake_call_seat(seat_cls, name, prompt):
            return script.pop(0)

        gates = iter([GOOD_GATE])

        async def fake_gate_async(tex, inp):
            try:
                return next(gates)
            except StopIteration:
                raise asyncio.CancelledError()

        loop._call_seat = fake_call_seat
        loop._gate_async = fake_gate_async
        out = asyncio.run(loop.run(WriteupLoopInputs(document_text="orig")))
        self.assertEqual(out.document_text, "orig")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_polish_failure_after_clean_verdict_ships_rewrite(self):
        # codex r6 #4: a clean verdict clears a prior UNABLE even if
        # the polish then fails.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             "fine now\nNO ERRORS\n",
             RuntimeError("polish seat down")])
        self.assertEqual(out.document_text, "REPAIRED\n")
        self.assertIn("polish step failed", out.shipped)

    def test_expired_deadline_launches_no_compiles(self):
        # codex r6 #3
        import time as _t
        start = _t.monotonic()
        compiled, pages, _note = _compile_raw(DOC, None,
                                              deadline=_t.monotonic() - 1)
        self.assertFalse(compiled)
        self.assertLess(_t.monotonic() - start, 2.0)

    def test_flag_lines(self):
        tex = "a\n%% FLAG: one\n  %% FLAG: two\n% not a flag\n"
        self.assertEqual(len(_flag_lines(tex)), 2)

    def test_editing_stages_receive_the_same_complete_guidance(self):
        guidance = (ROOT / "src/proofstack/agents/writeup_prompts"
                    / "WRITING_GUIDANCE.md").read_text(encoding="utf-8")
        for template, subs in (
            ("rewrite-wrapper.txt", {"document": DOC}),
            ("repair.txt", {"document": DOC, "referee findings": "REPORT"}),
        ):
            with self.subTest(template=template):
                text = _assemble(template, subs)
                self.assertEqual(text.count(guidance), 1)
                self.assertIn(
                    f"<writing_guidance>\n{guidance}\n</writing_guidance>", text)
                self.assertIn(f"<document>\n{DOC}\n</document>", text)
                self.assertNotIn("{{full text of WRITING_GUIDANCE.md}}", text)

    def test_cold_referee_does_not_receive_editor_guidance(self):
        text = _assemble("cold-referee.txt", {"document": DOC})
        self.assertNotIn("<writing_guidance>", text)
        self.assertNotIn("# Writing Guidance for Research Mathematics", text)
        self.assertIn(DOC, text)

    def test_assemble_tolerates_braces_in_document(self):
        text = _assemble("cold-referee.txt", {"document": "x_{{i}} {{weird}}"})
        self.assertIn("x_{{i}}", text)
        self.assertNotIn("{{document}}", text)

    def test_assemble_value_containing_placeholder_not_resubstituted(self):
        # Codex finding 13: a document containing the literal text
        # '{{referee findings}}' must not have the report spliced into it.
        text = _assemble("repair.txt",
                         {"document": "evil {{referee findings}} doc",
                          "referee findings": "REPORT"})
        self.assertIn("evil {{referee findings}} doc", text)
        self.assertEqual(text.count("REPORT"), 1)

    def test_assemble_notes_absent_is_byte_exact_previous_wrapper(self):
        text = _assemble("rewrite-wrapper.txt", {"document": "D"})
        self.assertNotIn("research notes", text)
        self.assertIn("</document>\n\nRewrite the document above", text)

    def test_assemble_notes_present_inserts_approved_block(self):
        text = _assemble("rewrite-wrapper.txt", {"document": "D"},
                         research_notes="NOTES CONTENT")
        self.assertIn("<research_notes>\nNOTES CONTENT\n</research_notes>", text)
        self.assertNotIn("{{research notes", text)

    def test_splice_repeated_placeholder_all_occurrences(self):
        # Codex r2 finding 8: every occurrence must be substituted.
        from proofstack.agents.writeup_loop import _splice
        self.assertEqual(_splice("{{x}} / {{x}}", {"x": "v"}), "v / v")

    def test_extract_prefers_fence_with_documentclass(self):
        # Codex r2 finding 7: an explanatory fence before the real
        # document must not be mistaken for the document.
        raw = ("```\njust an explanation\n```\nnow the doc:\n```latex\n"
               + DOC.strip() + "\n```\nFIXED: y\n")
        tex, decl = _extract_tex(raw)
        self.assertIn("\\documentclass", tex)
        self.assertNotIn("explanation", tex)
        self.assertTrue(_declares(decl, "FIXED:"))

    def test_assemble_missing_placeholder_raises(self):
        with self.assertRaises(RuntimeError):
            _assemble("cold-referee.txt", {"nonexistent": "x"})


@unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
class TestGate(unittest.TestCase):
    # Codex finding 1: the gate must compile exactly what ships.
    def test_bare_text_rejected(self):
        self.assertEqual(_compile_raw("BARE MODEL TEXT\n", None), (False, 0, ""))

    def test_blank_rejected(self):
        self.assertEqual(_compile_raw("", None), (False, 0, ""))
        self.assertEqual(_compile_raw("\n", None), (False, 0, ""))

    def test_minimal_document_compiles_with_page_count(self):
        compiled, pages, _note = _compile_raw(DOC, None)
        self.assertTrue(compiled)
        self.assertEqual(pages, 1)

    def test_broken_document_rejected(self):
        compiled, _pages, _note = _compile_raw(
            "\\documentclass{article}\\begin{document}\\badmacro\\end{document}",
            None)
        self.assertFalse(compiled)

    def test_typeout_cannot_spoof_page_count(self):
        # Codex r2 finding 4: a document printing "(1 pages,)" via
        # \typeout must not be counted as one page.
        tex = ("\\documentclass{article}\\begin{document}"
               "\\typeout{(1 pages,)}x\\newpage y\\newpage z"
               "\\end{document}")
        compiled, pages, _note = _compile_raw(tex, None)
        self.assertTrue(compiled)
        self.assertEqual(pages, 3)

    def test_proof_sectioning_defused_and_compiles(self):
        # \paragraph inside amsthm proof is illegal LaTeX (TeX Live 2025
        # crashes); the recurring rewriter construct is transformed to a
        # bold run-in heading, outside-proof sectioning left untouched.
        tex = ("\\documentclass[11pt]{article}"
               "\\usepackage{amsmath,amssymb,amsthm,mathtools}"
               "\\begin{document}\\paragraph{Outside.} ok\n"
               "\\begin{proof}[Proof]\n\\paragraph{Step 1.}\nText.\n"
               "\\paragraph{Step 2: Remove labels.}\nMore.\n\\end{proof}"
               "\\end{document}\n")
        fixed = _defuse_proof_sectioning(tex)
        self.assertIn("\\paragraph{Outside.}", fixed)
        self.assertNotIn("\\paragraph{Step", fixed)
        self.assertIn("\\textbf{Step 2: Remove labels.}", fixed)
        compiled, _pages, _note = _compile_raw(fixed, None)
        self.assertTrue(compiled)

    def test_bibtex_dance_resolves_citations(self):
        # Codex r2 finding 5: bib under the referenced name, bibtex exit
        # code enforced.
        tex = ("\\documentclass{article}\\begin{document}"
               "cites~\\cite{onlyentry}"
               "\\bibliographystyle{plain}\\bibliography{mybib}"
               "\\end{document}")
        bib = ("@article{onlyentry, author={A. Author}, title={T},"
               " journal={J}, year={2020}}")
        compiled, pages, _note = _compile_raw(tex, bib)
        self.assertTrue(compiled)
        self.assertEqual(pages, 1)


class TestControlFlow(unittest.TestCase):
    def test_happy_path_clean_first_round(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nPOLISHED\n```\nFIXED: none needed\n"])
        self.assertEqual(out.document_text, "POLISHED\n")
        self.assertIn("NO ERRORS, polish applied", out.shipped)
        self.assertTrue(out.improved)
        self.assertIsNone(out.error)

    def test_rewrite_gate_fails_twice_ships_original(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nBAD1\n```", "```\nBAD2\n```"],
            gates=[BAD_GATE, BAD_GATE])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("rewrite failed the gate twice", out.shipped)
        self.assertFalse(out.improved)

    def test_rewrite_seat_error_then_success(self):
        out, calls = run_loop(
            make_loop(),
            [RuntimeError("boom"),
             "```\nREWRITTEN\n```",
             "ok\nNO ERRORS\n",
             "```\nPOLISHED\n```"])
        self.assertEqual(out.document_text, "POLISHED\n")
        self.assertEqual(
            calls["seats"],
            ["rewrite-a1", "rewrite-a2", "referee-r1", "polish-r1"])

    def test_failed_gate_repair_keeps_unable_catastrophe(self):
        # Codex r2 finding 6: a final repair that is broken TeX AND
        # declares UNABLE must still trigger the catastrophe fallback.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nFIXED: a\n",
             "still bad\nERRORS FOUND\n",
             "```\nBROKEN\n```\nUNABLE: critical\n"],
            gates=[GOOD_GATE, GOOD_GATE, BAD_GATE])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE on a critical error", out.shipped)

    def test_rewrite_seats_unavailable_reason_is_honest(self):
        # Codex r2 finding 9: two seat errors must not be reported as
        # gate failures.
        out, calls = run_loop(
            make_loop(),
            [RuntimeError("a"), RuntimeError("b")])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("rewrite seats unavailable", out.shipped)

    def test_catastrophe_ships_original(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad stuff\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: the critical error\n",
             "still bad\nERRORS FOUND\n",
             "```\nREPAIRED2\n```\nUNABLE: still\n"])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE on a critical error", out.shipped)

    def test_round_cap_no_catastrophe_ships_polished(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "issues\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nFIXED: it\n",
             "more\nERRORS FOUND\n",
             "```\nREPAIRED2\n```\nFIXED: that\n"])
        self.assertEqual(out.document_text, "REPAIRED2\n")
        self.assertIn("round cap reached, no catastrophe", out.shipped)

    def test_polish_declaring_unable_is_discarded(self):
        # Codex finding 7: a polish contradicting its clean report must
        # not ship; the referee-cleared text ships instead.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nSNEAKY EDIT\n```\nUNABLE: critical gap\n"])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertIn("polish discarded", out.shipped)

    def test_ambiguous_verdict_takes_repair_path(self):
        # Prose mentioning NO ERRORS mid-sentence is not a clean verdict.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "I cannot certify NO ERRORS because Lemma 2 is false\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nFIXED: lemma\n",
             "ok\nNO ERRORS\n",
             "```\nFINAL\n```"])
        self.assertEqual(out.document_text, "FINAL\n")

    def test_referee_failure_ships_unchecked(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```", RuntimeError("referee down")])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertIn("referee unavailable", out.shipped)

    def test_budget_exhausted_ships_best_so_far(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             BudgetExhausted("run", "max_usd", 20.0, 21.0)])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertIn("budget exhausted", out.shipped)

    def test_cancellation_ships_best_so_far(self):
        # Codex finding 3: CancelledError is BaseException; the node
        # deliberately completes with best-effort outputs.
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```", asyncio.CancelledError()])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertIn("cancelled", out.shipped)

    def test_blank_input_ships_immediately_without_seats(self):
        out, calls = run_loop(make_loop(), [], doc="   \n")
        self.assertEqual(calls["seats"], [])
        self.assertIn("input document is blank", out.shipped)

    def test_whitespace_document_falls_back_to_input(self):
        # Codex finding 2: whitespace must not pass the `or` fallback.
        out, calls = run_loop(
            make_loop(),
            ["```\n \n```", "```\n \n```"],
            gates=[GOOD_GATE, GOOD_GATE])
        # gate faked as good, but _ship must still refuse to ship blank
        self.assertEqual(out.document_text, "original text")

    def test_raising_str_exception_still_ships(self):
        # Codex finding 4: an exception whose __str__ raises must not
        # escape the fallback handler.
        class EvilError(Exception):
            def __str__(self):
                raise RuntimeError("nope")

        loop = make_loop()

        async def broken_inner(inp, state):
            raise EvilError()

        loop._run_inner = broken_inner
        out = asyncio.run(loop.run(WriteupLoopInputs(document_text="orig")))
        self.assertEqual(out.document_text, "orig")
        self.assertEqual(out.error, "EvilError")

    def test_non_compiling_repair_discarded(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "issues\nERRORS FOUND\n",
             "```\nBROKEN REPAIR\n```\nFIXED: sure\n",
             "ok now\nNO ERRORS\n",
             "```\nFINAL POLISH\n```"],
            gates=[GOOD_GATE, BAD_GATE, GOOD_GATE])
        self.assertEqual(out.document_text, "FINAL POLISH\n")

    def test_every_seat_exploding_still_ships_original(self):
        loop = make_loop()

        async def explode(*a, **k):
            raise ValueError("totally unexpected")

        loop._call_seat = explode
        loop._gate = lambda tex, inp: GOOD_GATE
        out = asyncio.run(loop.run(WriteupLoopInputs(document_text="orig")))
        self.assertEqual(out.document_text, "orig")
        self.assertIn("rewrite seats unavailable", out.shipped)

    def test_outer_net_catches_loop_body_error(self):
        loop = make_loop()

        async def broken_inner(inp, state):
            raise KeyError("loop body bug")

        loop._run_inner = broken_inner
        out = asyncio.run(loop.run(WriteupLoopInputs(document_text="orig")))
        self.assertEqual(out.document_text, "orig")
        self.assertIn("best-effort after error", out.shipped)
        self.assertIn("KeyError", out.error)


TURN_COMPLETED = (
    '{"type":"turn.completed","usage":{"input_tokens":1200,'
    '"cached_input_tokens":900,"output_tokens":300,'
    '"reasoning_output_tokens":250}}'
)


class FakeProc:
    """Stands in for an asyncio subprocess: writes what a real
    ``codex exec --output-last-message FILE --json`` would leave behind."""

    def __init__(self, cmd, *, message, stdout, returncode, hang=False,
                 exit_then_hang=False):
        self._cmd = list(cmd)
        self._message = message
        self._stdout = stdout
        self.returncode = None if hang else returncode
        self._final_rc = returncode
        self._hang = hang
        # The leader exits and is reaped while a detached descendant keeps
        # the pipe open, so communicate() hangs on with returncode already
        # set — the r9 #4 shape, which r12 #7 is about.
        self._exit_then_hang = exit_then_hang
        self.pid = 424242
        self.killed = False

    def _out_path(self):
        i = self._cmd.index("--output-last-message")
        return Path(self._cmd[i + 1])

    async def communicate(self, payload=None):
        if self._hang:
            if self._exit_then_hang:
                self.returncode = self._final_rc
            await asyncio.sleep(3600)
        if self._message is not None:
            self._out_path().write_text(self._message, encoding="utf-8")
        self.returncode = self._final_rc
        return self._stdout.encode("utf-8"), None

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


SUBSCRIPTION_AUTH = (
    '{"auth_mode":"chatgpt","tokens":{"access_token":"tok-aaaaaaaa",'
    '"refresh_token":"tok-bbbbbbbb","id_token":"tok-cccccccc"}}'
)
APIKEY_AUTH = '{"OPENAI_API_KEY":"sk-fake-key-value","auth_mode":"apikey"}'


def fake_codex_home(test, auth_json=SUBSCRIPTION_AUTH):
    """A throwaway CODEX_HOME with a login of the requested class, so the
    seat's auth guard is exercised for real without depending on whatever
    `codex login` the host happens to hold."""
    import os
    import tempfile
    from unittest import mock

    home = Path(tempfile.mkdtemp(prefix="writeup_codex_home_"))
    test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    if auth_json is not None:
        (home / "auth.json").write_text(auth_json, encoding="utf-8")
    patcher = mock.patch.dict(os.environ, {"CODEX_HOME": str(home)})
    patcher.start()
    test.addCleanup(patcher.stop)
    return home


def patch_subprocess(test, *, message="OK", stdout=TURN_COMPLETED,
                     returncode=0, hang=False, exit_then_hang=False):
    """Patch asyncio.create_subprocess_exec; return a box holding the
    launched argv and the fake process."""
    from unittest import mock

    fake_codex_home(test)
    box = {"killpg_calls": [], "survivor_calls": []}

    async def fake_exec(*cmd, **kwargs):
        box["cmd"] = list(cmd)
        box["kwargs"] = kwargs
        proc = FakeProc(cmd, message=message, stdout=stdout,
                        returncode=returncode, hang=hang,
                        exit_then_hang=exit_then_hang)
        box["proc"] = proc
        return proc

    patcher = mock.patch("asyncio.create_subprocess_exec", fake_exec)
    patcher.start()
    test.addCleanup(patcher.stop)
    # The fake process's pid is invented; cleanup must not be allowed to
    # reach whatever really owns that number on this host (codex r11 #7).
    def no_such_group(pgid, sig):
        box["killpg_calls"].append(pgid)
        raise ProcessLookupError(pgid)

    kill_patcher = mock.patch(
        "proofstack.agents.writeup_codex_seat.os.killpg", no_such_group)
    kill_patcher.start()
    test.addCleanup(kill_patcher.stop)

    def fake_sweep(pgid, marker):
        box["survivor_calls"].append((pgid, marker))
        return 0

    sweep_patcher = mock.patch(
        "proofstack.agents.writeup_codex_seat.kill_seat_survivors",
        fake_sweep)
    sweep_patcher.start()
    test.addCleanup(sweep_patcher.stop)
    # Skip the real-binary check: these tests never launch codex.
    bin_patcher = mock.patch(
        "proofstack.agents.writeup_codex_seat.resolve_codex_bin",
        lambda raw=None: "/fake/codex")
    bin_patcher.start()
    test.addCleanup(bin_patcher.stop)
    return box


class TestCodexSeatCommand(unittest.TestCase):
    def test_command_carries_model_effort_web_search_and_stdin(self):
        cmd = build_codex_cmd(
            codex_bin="/fake/codex", model="gpt-5.6-sol",
            reasoning_effort="max", workdir="/tmp/w",
            last_message_path="/tmp/w/last.md")
        self.assertEqual(cmd[:2], ["/fake/codex", "exec"])
        self.assertEqual(cmd[cmd.index("-m") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="max"', cmd)
        self.assertIn("tools.web_search=true", cmd)
        self.assertIn("--json", cmd)
        self.assertIn("--output-last-message", cmd)
        # '-' means: read the prompt from stdin, never from argv.
        self.assertEqual(cmd[-1], "-")

    def test_user_config_ignored_by_default_but_optional(self):
        # Default on: no personality injection, and codex-billing's
        # API-provider switch cannot make this seat billable.
        base = dict(codex_bin="/fake/codex", model="m",
                    reasoning_effort="low", workdir="/tmp/w",
                    last_message_path="/tmp/w/last.md")
        self.assertIn("--ignore-user-config", build_codex_cmd(**base))
        self.assertNotIn("--ignore-user-config",
                         build_codex_cmd(**base, ignore_user_config=False))

    def test_web_search_can_be_disabled(self):
        cmd = build_codex_cmd(
            codex_bin="/fake/codex", model="m", reasoning_effort="low",
            workdir="/tmp/w", last_message_path="/tmp/w/last.md",
            web_search=False)
        self.assertNotIn("tools.web_search=true", cmd)

    def test_missing_binary_raises_codex_seat_error(self):
        with self.assertRaises(CodexSeatError):
            resolve_codex_bin("/nonexistent/definitely-not-codex-xyz")


class TestCodexSeatSubprocess(unittest.TestCase):
    def test_success_returns_final_message_and_usage(self):
        box = patch_subprocess(self, message="```latex\nDOC\n```\n")
        result = asyncio.run(run_codex_seat("prompt", reasoning_effort="low"))
        self.assertIn("DOC", result.text)
        self.assertEqual(result.usage.input_tokens, 1200)
        self.assertEqual(result.usage.output_tokens, 300)
        self.assertEqual(result.metered_tokens, 1500)
        self.assertIn('model_reasoning_effort="low"', box["cmd"])
        # Own process group, so a timeout can kill codex's whole tree.
        self.assertTrue(box["kwargs"].get("start_new_session"))

    def test_nonzero_exit_raises(self):
        patch_subprocess(self, message=None, stdout="boom\n", returncode=1)
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_empty_final_message_raises(self):
        patch_subprocess(self, message="   \n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_timeout_kills_process_and_raises(self):
        box = patch_subprocess(self, hang=True)
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt", timeout_s=0.1))
        # os.killpg on a fake pid fails and falls through to proc.kill().
        self.assertTrue(box["proc"].killed)

    def test_tokens_used_line_is_the_usage_fallback(self):
        patch_subprocess(self, stdout="no json here\ntokens used: 12,345\n")
        result = asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(result.usage.n_turns, 0)
        self.assertEqual(result.metered_tokens, 12345)


def mock_run_codex_seat(fake):
    from unittest import mock

    return mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake)


def run_cli_loop(test, replies, *, rounds=1, doc="original text",
                 gates=None, loop=None):
    """Drive run() with seat='cli' and a scripted run_codex_seat."""
    from unittest import mock

    loop = loop if loop is not None else make_loop()
    loop.seat = "cli"
    loop.cli_reasoning_effort = "low"
    script = list(replies)
    seen = []

    async def fake_run_codex_seat(prompt, **kwargs):
        seen.append(kwargs)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return CodexSeatResult(
            text=item,
            usage=CodexUsage(input_tokens=1000, output_tokens=200, n_turns=1),
            returncode=0, duration_s=1.0)

    gate_queue = list(gates or [])

    def fake_gate(tex, inp, *args):
        return gate_queue.pop(0) if gate_queue else GOOD_GATE

    loop._gate = fake_gate
    patcher = mock.patch(
        "proofstack.agents.writeup_loop.run_codex_seat", fake_run_codex_seat)
    patcher.start()
    test.addCleanup(patcher.stop)
    out = asyncio.run(loop.run(
        WriteupLoopInputs(document_text=doc, rounds=rounds)))
    return out, loop, seen


class TestCLISeatSelection(unittest.TestCase):
    def test_default_seat_is_api(self):
        # Nothing existing changes behavior unless a preset says so.
        self.assertEqual(WriteupLoop.seat, "api")

    def test_api_seat_does_not_shell_out(self):
        from unittest import mock

        loop = make_loop()
        loop.ctx = None
        loop._deadline = __import__("time").monotonic() + 3600

        class FakeSeat:
            def __init__(self, ctx, name=None, parent_budget_scope=None):
                self.wallclock_cap_s = None

            async def __call__(self, prompt):
                return type("O", (), {"text": "from-api"})()

        async def explode(*a, **k):
            raise AssertionError("api seat must not call the codex CLI")

        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat",
                        explode):
            text = asyncio.run(loop._call_seat(FakeSeat, "rewrite-a1", "p"))
        self.assertEqual(text, "from-api")

    def test_cli_seat_runs_the_loop_to_completion(self):
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nPOLISHED\n```\nFIXED: none needed\n"])
        self.assertEqual(out.document_text, "POLISHED\n")
        self.assertIn("NO ERRORS, polish applied", out.shipped)
        self.assertTrue(out.improved)
        self.assertIsNone(out.error)
        # Configurable knobs actually reach the CLI.
        self.assertEqual(seen[0]["reasoning_effort"], "low")
        self.assertEqual(seen[0]["model"], "gpt-5.6-sol")
        self.assertTrue(seen[0]["web_search"])

    def test_cli_seat_meters_tokens_and_no_dollars(self):
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nPOLISHED\n```\n"])
        self.assertEqual(out.usd_total, 0.0)
        self.assertEqual(loop.tracker.counters.tokens, 3 * 1200)

    def test_seat_transcripts_are_persisted(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            loop = make_loop()
            loop.seat = "cli"
            loop._gate = lambda tex, inp, *a: GOOD_GATE
            with mock.patch.object(
                    WriteupLoop, "workdir",
                    property(lambda self: Path(tmp))):
                async def fake(prompt, **kwargs):
                    return CodexSeatResult(text="```\nX\n```",
                                           log_tail="tail")

                with mock.patch(
                        "proofstack.agents.writeup_loop.run_codex_seat", fake):
                    asyncio.run(loop.run(WriteupLoopInputs(
                        document_text="orig", rounds=0)))
            seats = Path(tmp) / "cli-seats"
            self.assertTrue((seats / "rewrite-a1.prompt.txt").exists())
            self.assertTrue((seats / "rewrite-a1.reply.txt").exists())
            self.assertTrue((seats / "rewrite-a1.log-tail.txt").exists())

    def test_persist_failure_does_not_break_the_seat(self):
        from unittest import mock

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        async def fake(prompt, **kwargs):
            return CodexSeatResult(text="```\nX\n```")

        with mock.patch.object(
                WriteupLoop, "workdir",
                property(lambda self: Path("/proc/nonexistent/nope"))), \
             mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))
        self.assertEqual(out.document_text, "X\n")

    def test_codex_failure_on_every_call_ships_original(self):
        # The failure contract: a broken CLI degrades to the untouched
        # input, it never raises out of run().
        out, loop, seen = run_cli_loop(
            self,
            [CodexSeatError("codex exec exited 1"),
             CodexSeatError("codex exec exited 1")])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("rewrite seats unavailable", out.shipped)
        self.assertFalse(out.improved)

    def test_codex_timeout_mid_loop_ships_best_so_far(self):
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             CodexSeatError("codex exec exceeded its 600s wallclock bound")])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertIn("referee unavailable", out.shipped)

    def test_codex_failure_preserves_catastrophe_path(self):
        # A standing UNABLE must survive a CLI outage exactly as it does
        # an API one: the original ships.
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             "```\nREPAIRED\n```\nUNABLE: critical\n",
             CodexSeatError("codex exec exited 1")],
            rounds=2)
        self.assertEqual(out.document_text, "original text")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_cli_seat_bounded_by_remaining_wallclock(self):
        loop = make_loop(max_wallclock=600.0)
        loop.seat = "cli"
        captured = {}

        async def fake(prompt, **kwargs):
            captured.update(kwargs)
            return CodexSeatResult(text="```\nX\n```", returncode=0)

        from unittest import mock

        loop._gate = lambda tex, inp, *a: GOOD_GATE
        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))
        self.assertLessEqual(captured["timeout_s"], 600.0)
        self.assertGreater(captured["timeout_s"], 500.0)


class TestBudgetEnforcement(unittest.TestCase):
    """codex r7 #4: add_tokens only accumulates; the CLI seat never goes
    through APICallAgent, so nothing asked the tracker for headroom."""

    def test_cli_seats_stop_once_max_tokens_is_exhausted(self):
        loop = make_loop(spec=BudgetSpec(max_tokens=1))
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nPOLISHED\n```\n"],
            loop=loop)
        # The rewrite runs (the tracker starts clean) and is charged; the
        # referee is refused, and the node degrades rather than raising.
        self.assertEqual(len(seen), 1)
        self.assertEqual(out.document_text, "original text")
        self.assertIn("budget exhausted", out.shipped)

    def test_api_seat_also_asks_before_calling(self):
        loop = make_loop(spec=BudgetSpec(max_tokens=1))
        loop.tracker.add_tokens(5000)

        async def never(*a, **k):
            raise AssertionError("seat must not be constructed")

        loop._call_cli_seat = never
        loop._deadline = __import__("time").monotonic() + 3600
        with self.assertRaises(BudgetExhausted):
            asyncio.run(loop._call_seat(None, "rewrite-a1", "p"))

    def test_headroom_leaves_the_loop_untouched(self):
        out, loop, seen = run_cli_loop(
            self,
            ["```\nREWRITTEN\n```",
             "fine\nNO ERRORS\n",
             "```\nPOLISHED\n```\n"],
            loop=make_loop(spec=BudgetSpec(max_tokens=10_000_000)))
        self.assertEqual(len(seen), 3)
        self.assertEqual(out.document_text, "POLISHED\n")


class TestCompletedDeclarationSurvivesBudget(unittest.TestCase):
    """codex r7 #3: api_call.py's POST-call check raises after the reply is
    back, so a completed repair's UNABLE: was being thrown away."""

    def test_api_call_agent_attaches_the_completed_reply(self):
        import tempfile
        from unittest import mock

        from proofstack.budget import BudgetTracker
        from proofstack.context import RunContext
        from proofstack.kinds.api_call import APICallAgent

        class _In(BaseModel):
            prompt: str

        class _Out(BaseModel):
            text: str

        class _Seat(APICallAgent):
            Inputs = _In
            Outputs = _Out

            def render_messages(self, inp):
                return [{"role": "user", "content": inp.prompt}]

            def parse_output(self, raw_text, inp):
                return _Out(text=raw_text or "")

        class _Events:
            async def emit(self, *a, **kw):
                return None

        seat = _Seat.__new__(_Seat)
        seat.MODEL = "models/openai/gpt-56-sol-max"
        seat.name = "repair-r1"
        seat.events = _Events()
        seat._client = object()
        seat._fallback_workdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, seat._fallback_workdir,
                        ignore_errors=True)
        seat.ctx = RunContext.create(root_workdir=seat._fallback_workdir, flat=True)
        seat.tracker = BudgetTracker(scope="agent:repair",
                                     spec=BudgetSpec(max_usd=20.0))
        reply = "```latex\nREPAIRED\n```\nUNABLE: Lemma 4 is false\n"

        def fake_one_shot(client, messages):
            return (0,
                    [{"role": "assistant", "content": reply}],
                    {"cost": 21.0, "input_tokens": 10, "output_tokens": 10})

        with mock.patch("proofstack.kinds.api_call._one_shot_query",
                        fake_one_shot):
            with self.assertRaises(BudgetExhausted) as cm:
                asyncio.run(seat.run(_In(prompt="p")))
        self.assertEqual(cm.exception.completed_output.text, reply)

    def test_loop_registers_an_unable_lost_to_the_budget(self):
        exhausted = BudgetExhausted("run", "usd", 20.0, 21.0)
        exhausted.completed_output = type(
            "O", (), {"text": "```\nREPAIRED\n```\nUNABLE: critical\n"})()
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```", "bad\nERRORS FOUND\n", exhausted])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("unresolved UNABLE", out.shipped)

    def test_a_budget_exit_with_no_declaration_still_ships_the_rewrite(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```", "bad\nERRORS FOUND\n",
             BudgetExhausted("run", "usd", 20.0, 21.0)])
        self.assertEqual(out.document_text, "REWRITTEN\n")
        self.assertNotIn("unresolved UNABLE", out.shipped)

    def test_cli_seat_attaches_its_completed_reply_too(self):
        from unittest import mock

        loop = make_loop(spec=BudgetSpec(max_tokens=1))
        loop.seat = "cli"
        loop._deadline = __import__("time").monotonic() + 3600

        async def fake(prompt, **kwargs):
            return CodexSeatResult(
                text="```\nR\n```\nUNABLE: critical\n",
                usage=CodexUsage(input_tokens=10, output_tokens=1, n_turns=1))

        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            with self.assertRaises(BudgetExhausted) as cm:
                asyncio.run(loop._call_cli_seat("repair-r1", "p", 600.0))
        self.assertIn("UNABLE:", cm.exception.completed_output.text)


class TestCancellationTerminatesClient(unittest.TestCase):
    """codex r7 #2: the API call runs in a worker thread whose poll loop
    stops only on APIClient.terminate(); cancelling the await alone leaves
    a background response live and billing."""

    def _run_with(self, seat_error):
        loop = make_loop()
        loop.ctx = None
        loop._deadline = __import__("time").monotonic() + 3600
        box = {"terminated": False}

        class FakeClient:
            def terminate(self):
                box["terminated"] = True

        class FakeSeat:
            def __init__(self, ctx, name=None, parent_budget_scope=None):
                self.wallclock_cap_s = None
                self._client = FakeClient()

            async def __call__(self, prompt):
                raise seat_error

        with self.assertRaises(type(seat_error)):
            asyncio.run(loop._call_seat(FakeSeat, "rewrite-a1", "p"))
        return box

    def test_cancellation_terminates_the_client(self):
        self.assertTrue(self._run_with(asyncio.CancelledError())["terminated"])

    def test_wait_for_timeout_terminates_the_client(self):
        self.assertTrue(self._run_with(TimeoutError("hung"))["terminated"])

    def test_a_clean_call_leaves_the_client_alone(self):
        loop = make_loop()
        loop.ctx = None
        loop._deadline = __import__("time").monotonic() + 3600
        box = {"terminated": False}

        class FakeSeat:
            def __init__(self, ctx, name=None, parent_budget_scope=None):
                self.wallclock_cap_s = None
                self._client = type(
                    "C", (),
                    {"terminate": lambda _s: box.__setitem__("terminated", True)})()

            async def __call__(self, prompt):
                return type("O", (), {"text": "ok"})()

        self.assertEqual(
            asyncio.run(loop._call_seat(FakeSeat, "rewrite-a1", "p")), "ok")
        self.assertFalse(box["terminated"])


class TestCodexSeatAuthGuard(unittest.TestCase):
    """codex r7 #1: --ignore-user-config pins the provider, not the
    credential. An API-key login would bill dollars against a $0 seat."""

    def test_api_key_login_is_refused(self):
        home = fake_codex_home(self, APIKEY_AUTH)
        with self.assertRaises(CodexSeatError) as cm:
            assert_subscription_login({"CODEX_HOME": str(home)})
        self.assertIn("api_key", str(cm.exception))

    def test_absent_login_is_refused(self):
        home = fake_codex_home(self, auth_json=None)
        with self.assertRaises(CodexSeatError):
            assert_subscription_login({"CODEX_HOME": str(home)})

    def test_chatgpt_login_is_accepted(self):
        home = fake_codex_home(self)
        assert_subscription_login({"CODEX_HOME": str(home)})

    def test_run_codex_seat_refuses_before_spawning(self):
        from unittest import mock

        patch_subprocess(self, message="```\nDOC\n```\n")
        fake_codex_home(self, APIKEY_AUTH)   # innermost patch wins
        spawned = []

        async def tripwire(*cmd, **kwargs):
            spawned.append(cmd)
            raise AssertionError("must not spawn on a non-subscription login")

        with mock.patch("asyncio.create_subprocess_exec", tripwire):
            with self.assertRaises(CodexSeatError):
                asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(spawned, [])

    def test_auth_env_vars_are_stripped_from_the_child(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ,
                             {k: "sk-leak" for k in BLOCKED_AUTH_ENVS}):
            env = child_env()
        for key in BLOCKED_AUTH_ENVS:
            self.assertNotIn(key, env)
        self.assertIn("PATH", env)

    def test_spawned_env_carries_no_api_key(self):
        import os
        from unittest import mock

        box = patch_subprocess(self, message="```\nDOC\n```\n")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-leak"}):
            asyncio.run(run_codex_seat("prompt"))
        self.assertNotIn("OPENAI_API_KEY", box["kwargs"]["env"])


class TestBibliographyBackend(unittest.TestCase):
    """codex r7 #5: a supplied bibliography we cannot process is a gate
    failure — pdflatex exits 0 and the document would ship with an empty
    bibliography and unresolved citations."""

    BIBLATEX_DOC = ("\\documentclass{article}\n"
                    "\\usepackage[backend=biber]{biblatex}\n"
                    "\\addbibresource{refs.bib}\n"
                    "\\begin{document}\nSee \\cite{key1}.\n"
                    "\\printbibliography\n\\end{document}\n")
    BIB = "@article{key1, author={A}, title={T}, journal={J}, year={2020}}\n"

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_missing_backend_fails_the_gate_with_a_note(self):
        from unittest import mock

        real_which = shutil.which  # captured before the patch: the lambda
        with mock.patch(          # below must not call the patched name
                "proofstack.agents.writeup_loop.shutil.which",
                lambda name: None if name == "biber" else real_which(name)):
            compiled, pages, note = _compile_raw(self.BIBLATEX_DOC, self.BIB)
        self.assertFalse(compiled)
        self.assertEqual(pages, 0)
        self.assertIn("biber", note)
        self.assertIn("not installed", note)

    def test_the_note_reaches_the_step_record(self):
        loop = make_loop()
        loop._gate = lambda tex, inp, *a: GateResult(
            ok=False, compiled=True, pages=1,
            note="bibliography backend 'biber' not installed")
        script = ["```\n" + DOC + "\n```"] * 2

        async def fake_call_seat(seat_cls, name, prompt):
            return script.pop(0)

        loop._call_seat = fake_call_seat
        out = asyncio.run(loop.run(WriteupLoopInputs(
            document_text="orig", bib_text=self.BIB)))
        self.assertIn("not installed", out.steps[0].detail)

    def test_no_bibliography_needed_leaves_the_note_empty(self):
        if not HAS_PDFLATEX:
            self.skipTest("pdflatex not installed")
        compiled, _pages, note = _compile_raw(DOC, None)
        self.assertTrue(compiled)
        self.assertEqual(note, "")


class TestParagraphCounters(unittest.TestCase):
    """codex r7 #6: the run-in replacement dropped \\paragraph's counter, so
    under \\setcounter{secnumdepth}{4} two labelled proof steps both
    resolved to the enclosing section."""

    LABELLED = ("\\begin{proof}\n"
                "\\paragraph{Step one}\\label{step:a}\ntext\n"
                "\\paragraph{Step two}\\label{step:b}\ntext\n"
                "\\end{proof}\n")

    def test_unstarred_paragraph_keeps_its_counter(self):
        fixed = _defuse_proof_sectioning(self.LABELLED)
        self.assertNotIn("\\paragraph{", fixed)
        self.assertEqual(fixed.count("\\refstepcounter{paragraph}"), 2)
        # LaTeX's own test, so the default (unnumbered) rendering is
        # unchanged: the plain bold heading is still the \else branch.
        self.assertIn("\\ifnum\\value{secnumdepth}>3", fixed)
        self.assertIn("\\else\\textbf{Step one}\\fi", fixed)

    def test_starred_paragraph_steps_nothing(self):
        fixed = _defuse_proof_sectioning(
            "\\begin{proof}\\paragraph*{Step}x\\end{proof}")
        self.assertNotIn("\\refstepcounter", fixed)
        self.assertIn("\\textbf{Step}", fixed)

    def test_subparagraph_uses_its_own_counter_and_depth(self):
        fixed = _defuse_proof_sectioning(
            "\\begin{proof}\\subparagraph{S}x\\end{proof}")
        self.assertIn("\\ifnum\\value{secnumdepth}>4", fixed)
        self.assertIn("\\refstepcounter{subparagraph}", fixed)

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_labels_resolve_to_the_step_under_secnumdepth_4(self):
        import subprocess
        import tempfile

        tex = ("\\documentclass{article}\n\\setcounter{secnumdepth}{4}\n"
               "\\usepackage{amsthm}\n\\begin{document}\n\\section{S}\n"
               + _defuse_proof_sectioning(self.LABELLED)
               + "Refs: \\ref{step:a} and \\ref{step:b}.\n\\end{document}\n")
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "main.tex").write_text(tex, encoding="utf-8")
            for _ in range(2):
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                     "main.tex"], cwd=work, capture_output=True, timeout=120)
            self.assertEqual(proc.returncode, 0)
            aux = (Path(work) / "main.aux").read_text(encoding="utf-8")
        labels = [ln for ln in aux.splitlines() if "newlabel" in ln]
        self.assertEqual(len(labels), 2)
        # Distinct, and pointing at the paragraph rather than the section.
        self.assertNotEqual(labels[0].split("{", 2)[2],
                            labels[1].split("{", 2)[2])
        self.assertIn("paragraph.", labels[0])


class TestProviderPinning(unittest.TestCase):
    """codex r8 #1: with cli_ignore_user_config=false, config.toml's
    top-level model_provider could route a '$0' seat to the key-billed
    provider while auth.json still held a valid ChatGPT login."""

    BASE = dict(codex_bin="/fake/codex", model="m", reasoning_effort="low",
                workdir="/tmp/w", last_message_path="/tmp/w/last.md")

    def test_provider_pinned_with_user_config_ignored(self):
        self.assertIn('model_provider="openai"', build_codex_cmd(**self.BASE))

    def test_provider_pinned_with_user_config_honoured(self):
        # The flag keeps its meaning (personality / project docs) but must
        # not also hand billing to the file.
        cmd = build_codex_cmd(**self.BASE, ignore_user_config=False)
        self.assertNotIn("--ignore-user-config", cmd)
        self.assertIn('model_provider="openai"', cmd)

    def test_the_pin_is_a_config_override(self):
        cmd = build_codex_cmd(**self.BASE)
        self.assertEqual(cmd[cmd.index('model_provider="openai"') - 1], "-c")

    def test_the_spawned_command_carries_the_pin(self):
        box = patch_subprocess(self, message="```\nDOC\n```\n")
        asyncio.run(run_codex_seat("prompt", ignore_user_config=False))
        self.assertIn('model_provider="openai"', box["cmd"])


class TestBibliographyDetection(unittest.TestCase):
    """codex r8 #2 and #5: the resource regex missed an optional argument,
    and the backend was chosen from the resource command rather than from
    biblatex's own configuration."""

    def test_addbibresource_with_optional_argument_is_found(self):
        tex = ("\\documentclass{article}\n"
               "\\addbibresource[location=local]{refs.bib}\n")
        self.assertEqual([m.group(1) for m in _BIBLATEX_RE.finditer(tex)],
                         ["refs.bib"])

    def test_plain_addbibresource_still_found(self):
        self.assertEqual(
            [m.group(1) for m in
             _BIBLATEX_RE.finditer("\\addbibresource{refs.bib}")],
            ["refs.bib"])

    def test_backend_defaults_to_biber(self):
        self.assertEqual(_biblatex_backend("\\usepackage{biblatex}"), "biber")
        self.assertEqual(
            _biblatex_backend("\\usepackage[style=alphabetic]{biblatex}"),
            "biber")

    def test_backend_bibtex_is_respected(self):
        self.assertEqual(
            _biblatex_backend("\\usepackage[backend=bibtex]{biblatex}"),
            "bibtex")
        self.assertEqual(
            _biblatex_backend(
                "\\usepackage[style=plain,backend=bibtex8]{biblatex}"),
            "bibtex")

    def test_another_package_backend_option_is_not_read(self):
        # 'backend=' belongs to whichever package declared it.
        self.assertEqual(
            _biblatex_backend("\\usepackage[backend=bibtex]{minted}\n"
                              "\\usepackage{biblatex}"),
            "biber")

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_optional_argument_document_reaches_a_backend(self):
        # Before the fix this compiled clean with an empty bibliography;
        # now the backend is required, so a host without it fails the gate
        # rather than shipping unresolved citations.
        tex = ("\\documentclass{article}\n"
               "\\usepackage[backend=biber]{biblatex}\n"
               "\\addbibresource[location=local]{refs.bib}\n"
               "\\begin{document}\nSee \\cite{key1}.\n"
               "\\printbibliography\n\\end{document}\n")
        bib = ("@article{key1, author={A}, title={T}, journal={J},"
               " year={2020}}\n")
        compiled, _pages, note = _compile_raw(tex, bib)
        if shutil.which("biber") is None:
            self.assertFalse(compiled)
            self.assertIn("biber", note)
        else:
            self.assertTrue(compiled)

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_bibtex_backed_biblatex_uses_bibtex(self):
        from unittest import mock

        picked = {}
        real_which = shutil.which

        def spy(name):
            picked.setdefault("first", name)
            return real_which(name)

        tex = ("\\documentclass{article}\n"
               "\\usepackage[backend=bibtex]{biblatex}\n"
               "\\addbibresource{refs.bib}\n"
               "\\begin{document}\nSee \\cite{key1}.\n"
               "\\printbibliography\n\\end{document}\n")
        bib = ("@article{key1, author={A}, title={T}, journal={J},"
               " year={2020}}\n")
        with mock.patch("proofstack.agents.writeup_loop.shutil.which", spy):
            _compile_raw(tex, bib)
        self.assertEqual(picked["first"], "bibtex")


class TestTruncatedFenceDeclaration(unittest.TestCase):
    """codex r8 #3: a fence holding \\documentclass but no \\end{document}
    hid an UNABLE: line from the declaration scan."""

    TRUNCATED = ("```latex\n\\documentclass{article}\n\\begin{document}\n"
                 "cut off here\nUNABLE: the lemma is false\n```\n")

    def test_declaration_inside_a_truncated_fence_is_seen(self):
        _tex, decl = _extract_tex(self.TRUNCATED)
        self.assertTrue(_declares(decl, "UNABLE:"))

    def test_complete_fence_body_is_not_scanned(self):
        # A COMPLETE fence's body must still not count: only what follows
        # \end{document} is declaration text.
        raw = ("```latex\n\\documentclass{article}\n\\begin{document}\n"
               "UNABLE: this is body text, not a declaration\n"
               "\\end{document}\n```\n")
        _tex, decl = _extract_tex(raw)
        self.assertFalse(_declares(decl, "UNABLE:"))

    def test_truncated_repair_still_ships_the_original(self):
        out, calls = run_loop(
            make_loop(),
            ["```\nREWRITTEN\n```",
             "bad\nERRORS FOUND\n",
             self.TRUNCATED,
             "still bad\nERRORS FOUND\n",
             self.TRUNCATED],
            gates=[GOOD_GATE, BAD_GATE, BAD_GATE])
        self.assertEqual(out.document_text, "original text")
        self.assertIn("UNABLE", out.shipped)


class TestCLIToolCallBudget(unittest.TestCase):
    """codex r8 #4: CLI spawns never reached the tool-call counter, so a
    shared max_tool_calls did not constrain this seat (CLIAgent charges
    its own spawns, kinds/cli.py)."""

    REPLIES = ["```\nREWRITTEN\n```", "fine\nNO ERRORS\n",
               "```\nPOLISHED\n```\n"]

    def test_max_tool_calls_stops_the_cli_loop(self):
        out, loop, seen = run_cli_loop(
            self, list(self.REPLIES),
            loop=make_loop(spec=BudgetSpec(max_tool_calls=1)))
        self.assertEqual(len(seen), 1)
        self.assertEqual(loop.tracker.counters.tool_calls, 1)
        self.assertEqual(out.document_text, "original text")
        self.assertIn("budget exhausted", out.shipped)

    def test_every_spawn_is_charged(self):
        out, loop, seen = run_cli_loop(self, list(self.REPLIES))
        self.assertEqual(loop.tracker.counters.tool_calls, 3)

    def test_a_failed_spawn_is_charged_too(self):
        out, loop, seen = run_cli_loop(
            self, [CodexSeatError("codex exec exited 1"), *self.REPLIES])
        self.assertEqual(loop.tracker.counters.tool_calls, 4)


class TestCLIFailureDiagnostics(unittest.TestCase):
    """codex r8 #6: a failed CLI seat left only its exception TYPE in the
    step record; nothing in the run directory said why."""

    def _failing_loop(self, tmp, exc):
        from unittest import mock

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE
        events = []

        class Events:
            async def emit(self, name, payload, **kw):
                events.append((name, payload))

        loop.events = Events()

        async def fake(prompt, **kwargs):
            raise exc

        with mock.patch.object(WriteupLoop, "workdir",
                               property(lambda self: Path(tmp))), \
             mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))
        return out, events

    def test_failure_reason_and_prompt_land_in_the_run_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out, _events = self._failing_loop(
                tmp, CodexSeatError("codex exec exited 1; log tail: boom"))
            seats = Path(tmp) / "cli-seats"
            err = (seats / "rewrite-a1.error.txt").read_text(encoding="utf-8")
            self.assertTrue((seats / "rewrite-a1.prompt.txt").exists())
        self.assertIn("CodexSeatError", err)
        self.assertIn("log tail: boom", err)
        self.assertEqual(out.document_text, "orig")

    def test_an_agent_error_event_is_emitted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _out, events = self._failing_loop(
                tmp, CodexSeatError("codex binary not found"))
        errors = [p for n, p in events if n == "agent.error"]
        self.assertTrue(errors)
        self.assertEqual(errors[0]["type"], "CodexSeatError")
        self.assertIn("not found", errors[0]["msg"])
        self.assertEqual(errors[0]["seat"], "rewrite-a1")

    def test_the_record_is_redacted(self):
        import json
        import tempfile

        fake_codex_home(self)
        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        self.assertEqual(redact_seat_text(f"leaked {secret} here"),
                         "leaked [redacted-codex-credential] here")
        with tempfile.TemporaryDirectory() as tmp:
            self._failing_loop(tmp,
                               CodexSeatError(f"log tail: Bearer {secret}"))
            err = (Path(tmp) / "cli-seats" / "rewrite-a1.error.txt").read_text(
                encoding="utf-8")
        self.assertNotIn(secret, err)
        self.assertIn("[redacted-codex-credential]", err)

    def test_a_broken_workdir_still_degrades(self):
        out, events = self._failing_loop("/proc/nonexistent/nope",
                                         CodexSeatError("exited 1"))
        self.assertEqual(out.document_text, "orig")
        self.assertTrue([p for n, p in events if n == "agent.error"])


class TestSuccessfulTranscriptRedaction(unittest.TestCase):
    """codex r9 #1: only FAILED seats were redacted, so a credential echoed
    by a successful turn went verbatim into the run artifacts."""

    def _run_seat(self, tmp, text, log_tail):
        from unittest import mock

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        async def fake(prompt, **kwargs):
            return CodexSeatResult(text=text, log_tail=log_tail)

        with mock.patch.object(WriteupLoop, "workdir",
                               property(lambda self: Path(tmp))), \
             mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            return asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))

    def test_reply_and_log_tail_are_redacted(self):
        import json
        import tempfile

        fake_codex_home(self)
        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        with tempfile.TemporaryDirectory() as tmp:
            self._run_seat(tmp, f"```\nX {secret}\n```", f"Bearer {secret}")
            seats = Path(tmp) / "cli-seats"
            reply = (seats / "rewrite-a1.reply.txt").read_text(
                encoding="utf-8")
            tail = (seats / "rewrite-a1.log-tail.txt").read_text(
                encoding="utf-8")
        for got in (reply, tail):
            self.assertNotIn(secret, got)
            self.assertIn("[redacted-codex-credential]", got)

    def test_ordinary_transcripts_are_untouched(self):
        import tempfile

        fake_codex_home(self)
        with tempfile.TemporaryDirectory() as tmp:
            self._run_seat(tmp, "```\nPLAIN\n```", "ordinary log tail")
            seats = Path(tmp) / "cli-seats"
            self.assertIn("PLAIN", (seats / "rewrite-a1.reply.txt").read_text(
                encoding="utf-8"))
            self.assertEqual(
                (seats / "rewrite-a1.log-tail.txt").read_text(
                    encoding="utf-8"),
                "ordinary log tail")


class TestUsageOnFailedCLITurn(unittest.TestCase):
    """codex r9 #2: usage was parsed only after the failure checks, so a
    turn that reported tokens and then failed was recorded as free."""

    def test_nonzero_exit_still_carries_its_usage(self):
        patch_subprocess(self, message=None, stdout=TURN_COMPLETED,
                         returncode=1)
        with self.assertRaises(CodexSeatError) as cm:
            asyncio.run(run_codex_seat("prompt"))
        self.assertIsNotNone(cm.exception.partial)
        self.assertEqual(cm.exception.partial.metered_tokens, 1500)

    def test_empty_final_message_still_carries_its_usage(self):
        patch_subprocess(self, message="   \n", stdout=TURN_COMPLETED)
        with self.assertRaises(CodexSeatError) as cm:
            asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(cm.exception.partial.metered_tokens, 1500)

    def test_a_failure_with_no_usage_reports_zero(self):
        patch_subprocess(self, message=None, stdout="no json here\n",
                         returncode=1)
        with self.assertRaises(CodexSeatError) as cm:
            asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(cm.exception.partial.metered_tokens, 0)

    def test_the_loop_charges_a_failed_turn(self):
        from unittest import mock

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE
        err = CodexSeatError("codex exec exited 1")
        err.partial = CodexSeatResult(
            text="", usage=CodexUsage(input_tokens=1000, output_tokens=200,
                                      n_turns=1))

        async def fake(prompt, **kwargs):
            raise err

        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))
        # Two rewrite attempts, both charged.
        self.assertEqual(loop.tracker.counters.tokens, 2400)
        self.assertEqual(out.document_text, "orig")

    def test_a_charged_failure_can_exhaust_the_token_budget(self):
        from unittest import mock

        loop = make_loop(spec=BudgetSpec(max_tokens=1500))
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE
        err = CodexSeatError("codex exec exited 1")
        err.partial = CodexSeatResult(
            text="", usage=CodexUsage(input_tokens=2000, output_tokens=0,
                                      n_turns=1))
        seen = []

        async def fake(prompt, **kwargs):
            seen.append(1)
            raise err

        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=0)))
        # The second attempt is refused rather than retried for free.
        self.assertEqual(len(seen), 1)
        self.assertIn("budget exhausted", out.shipped)


class TestGateCancellation(unittest.TestCase):
    """codex r9 #3: cancelling asyncio.to_thread abandons the future but
    not the worker, so the compiler passes ran on after run() had shipped."""

    def test_a_stopped_canceller_launches_no_passes(self):
        c = _GateCanceller()
        c.cancel()
        started = time.monotonic()
        compiled, pages, _note = _compile_raw(DOC, None, canceller=c)
        self.assertFalse(compiled)
        self.assertEqual(pages, 0)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_cancel_kills_the_pass_in_flight(self):
        c = _GateCanceller()
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.addCleanup(_reap, proc)
        c.track(proc)
        c.cancel()
        self.assertEqual(proc.wait(timeout=10), -9)

    def test_tracking_after_cancel_kills_immediately(self):
        # The race: cancelled between the flag check and the spawn.
        c = _GateCanceller()
        c.cancel()
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.addCleanup(_reap, proc)
        c.track(proc)
        self.assertEqual(proc.wait(timeout=10), -9)

    def test_gate_async_cancels_its_canceller(self):
        seen = {}
        entered = threading.Event()
        loop = make_loop()
        loop._deadline = time.monotonic() + 3600

        def blocking_gate(tex, inp, pass_timeout, deadline, canceller):
            seen["canceller"] = canceller
            entered.set()
            # Stands in for a pdflatex pass: returns once stopped, or after
            # a short bound, so an UNpropagated cancellation fails the
            # assertion below instead of hanging the suite.
            bound = time.monotonic() + 5.0
            while not canceller.stopped() and time.monotonic() < bound:
                time.sleep(0.01)
            return BAD_GATE

        loop._gate = blocking_gate

        async def drive():
            task = asyncio.create_task(
                loop._gate_async(DOC, WriteupLoopInputs(document_text="x")))
            await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(drive())
        self.assertTrue(seen["canceller"].stopped())


class TestKillGroupAfterLeaderExit(unittest.TestCase):
    """codex r9 #4: communicate() can time out with returncode already set
    when a descendant still holds the pipe; the old early return then left
    that descendant alive."""

    class Reaped:
        pid = 424242

        def __init__(self, returncode=0):
            self.returncode = returncode
            self.killed = False

        def kill(self):
            self.killed = True

    def test_group_is_signalled_even_when_the_leader_is_reaped(self):
        from unittest import mock

        calls = []
        with mock.patch("proofstack.agents.writeup_codex_seat.os.killpg",
                        lambda pgid, sig: calls.append((pgid, sig))):
            _kill_process_group(self.Reaped(), 4242)
        self.assertEqual(calls, [(4242, signal.SIGKILL)])

    def test_the_saved_pgid_is_used_not_a_lookup(self):
        from unittest import mock

        def boom(pid):
            raise ProcessLookupError("leader already reaped")

        calls = []
        with mock.patch("proofstack.agents.writeup_codex_seat.os.killpg",
                        lambda pgid, sig: calls.append(pgid)), \
             mock.patch("proofstack.agents.writeup_codex_seat.os.getpgid",
                        boom):
            _kill_process_group(self.Reaped(), 4242)
        self.assertEqual(calls, [4242])

    def test_falls_back_to_kill_when_the_group_is_gone(self):
        proc = self.Reaped(returncode=None)
        _kill_process_group(proc, 4242)   # no such group on this host
        self.assertTrue(proc.killed)

    def test_the_timeout_path_passes_the_saved_pgid(self):
        from unittest import mock

        box = patch_subprocess(self, hang=True)
        seen = []
        with mock.patch("proofstack.agents.writeup_codex_seat.os.killpg",
                        lambda pgid, sig: seen.append(pgid)):
            with self.assertRaises(CodexSeatError):
                asyncio.run(run_codex_seat("prompt", timeout_s=0.1))
        self.assertEqual(seen, [box["proc"].pid])


class TestBibliographyNotDuplicated(unittest.TestCase):
    """codex r9 #5: \\bibliography{first,second} got the merged bib written
    into BOTH files, so bibtex failed on repeated entries."""

    TEX = ("\\documentclass{article}\\begin{document}"
           "cites~\\cite{onlyentry}"
           "\\bibliographystyle{plain}\\bibliography{first,second}"
           "\\end{document}")
    BIB = ("@article{onlyentry, author={A. Author}, title={T},"
           " journal={J}, year={2020}}")

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_two_resources_do_not_duplicate_entries(self):
        compiled, pages, note = _compile_raw(self.TEX, self.BIB)
        self.assertTrue(compiled, note)
        self.assertEqual(note, "")
        self.assertEqual(pages, 1)

    @unittest.skipUnless(HAS_PDFLATEX, "pdflatex not installed")
    def test_a_single_resource_still_resolves(self):
        tex = self.TEX.replace("{first,second}", "{mybib}")
        compiled, _pages, note = _compile_raw(tex, self.BIB)
        self.assertTrue(compiled, note)


class TestCLICredentialInReply(unittest.TestCase):
    """codex r10 #1 / r11 #5: only the on-disk transcripts were redacted, so
    a credential echoed by the turn became the document and the next seat's
    prompt. Returning it SCRUBBED is not the fix either — that silently
    edits the document's mathematics — so such a reply fails the seat."""

    def _drive(self, replies, rounds=1):
        """Run the CLI loop with scripted replies, capturing every prompt
        the seat was actually handed."""
        from unittest import mock

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE
        script, prompts = list(replies), []

        async def fake(prompt, **kwargs):
            prompts.append(prompt)
            return CodexSeatResult(text=script.pop(0))

        with mock.patch("proofstack.agents.writeup_loop.run_codex_seat", fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=rounds)))
        return out, prompts

    def test_a_credential_in_a_rewrite_never_reaches_the_next_prompt(self):
        import json

        fake_codex_home(self)
        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        out, prompts = self._drive([
            f"```\n% leaked {secret}\nX\n```",
            "```\nCLEAN\n```",
            "NO ERRORS\n",
            "```\nY\n```\n",
        ])
        for prompt in prompts:
            self.assertNotIn(secret, prompt)
        self.assertNotIn(secret, out.document_text)
        # The contaminated candidate was discarded, not scrubbed and kept.
        self.assertNotIn("[redacted-codex-credential]", out.document_text)
        self.assertEqual(out.steps[0].detail, "seat error: CodexSeatError")
        # The re-roll ran: a fresh sample, not the contaminated one.
        self.assertEqual(out.steps[1].step, "rewrite-a2")
        self.assertEqual(out.document_text, "Y\n")

    def test_a_credential_in_a_repair_leaves_the_earlier_document(self):
        import json

        fake_codex_home(self)
        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["refresh_token"]
        out, _prompts = self._drive([
            "```\nX\n```",
            "NO ERRORS\n",
            f"```\n% {secret}\nY\n```\n",
        ])
        self.assertEqual(out.document_text, "X\n")
        self.assertNotIn(secret, out.document_text)

    def test_a_credential_never_on_disk_is_refused_on_its_shape(self):
        """codex r11 #1: watching auth.json can always miss a value that
        lived only between two polls, so the shape decides too."""
        fake_codex_home(self)
        token = "sk-proj-Ab3dEf9hIjKlMn0pQrStUv"
        patch_subprocess(self, message=f"```\nDOC {token}\n```\n")
        with self.assertRaises(CodexSeatError) as cm:
            asyncio.run(run_codex_seat("prompt"))
        self.assertNotIn(token, str(cm.exception))

    def test_a_jwt_in_a_reply_is_refused_too(self):
        fake_codex_home(self)
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J"
        patch_subprocess(self, message=f"```\nDOC {jwt}\n```\n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_ordinary_mathematics_is_not_credential_shaped(self):
        # codex r12 #1: `sk-polynomial-estimates` lives inside the
        # perfectly ordinary label `task-polynomial-estimates`, and a
        # document whose every draft trips the screen never improves.
        self.assertFalse(looks_like_credential(DOC))
        self.assertFalse(looks_like_credential(
            "\\label{task-polynomial-estimates}"))
        self.assertFalse(looks_like_credential(
            "\\cite{sk-polynomial-estimates-2024}"))
        self.assertFalse(looks_like_credential(
            "see Sec. 3.1, Thm. 4.2 and the bound in eq. (17)"))
        self.assertFalse(looks_like_credential(
            "\\usepackage{amsmath}\n\\newcommand{\\sk-notation}{x}"))

    def test_a_document_full_of_hyphenated_labels_still_ships(self):
        fake_codex_home(self)
        doc = ("```\n\\documentclass{article}\\begin{document}\n"
               "\\label{task-polynomial-estimates-for-sums}\n"
               "\\end{document}\n```")
        out, _prompts = self._drive([doc, "NO ERRORS\n", doc])
        self.assertTrue(out.improved)
        self.assertIn("task-polynomial-estimates", out.document_text)

    def test_a_credential_wrapped_in_markdown_emphasis_is_refused(self):
        """codex r13 #1: `_sk-..._` and `_<jwt>_` slipped past the word
        boundary the previous round added."""
        import base64

        fake_codex_home(self)
        key = "sk-proj-Ab3dEf9hIjKlMn0pQrStUv"
        head = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        jwt = f"{head}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        self.assertTrue(looks_like_credential(f"Diagnostics: _{key}_"))
        self.assertTrue(looks_like_credential(f"Diagnostics: _{jwt}_"))
        self.assertNotIn(jwt, scrub_credentials(f"x _{jwt}_ y", ()))
        patch_subprocess(self, message=f"```\nDOC _{key}_\n```\n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_a_project_key_with_separators_in_its_body_is_refused(self):
        """codex r14 #1: `sk-proj-Ab3dEf9_...` has no unbroken 16-character
        run, so the run rule alone never saw it."""
        fake_codex_home(self)
        key = "sk-proj-" + "Ab3dEf9_hIjKlMn0-pQrStUvWx1Yz2" * 5
        self.assertTrue(looks_like_credential(f"DOC {key}"))
        self.assertTrue(looks_like_credential(
            "sk-ant-api03-Ab3dEf9_hIjKlMn0pQrStUvWx1Yz2"))
        self.assertNotIn(key, scrub_credentials(f"x {key} y", ()))
        patch_subprocess(self, message=f"```\nDOC {key}\n```\n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_an_unknown_provider_prefix_is_caught_by_length_and_randomness(self):
        import random
        import string

        alphabet = string.ascii_letters + string.digits + "-_"
        rng = random.Random(11)
        for _ in range(50):
            body = "".join(rng.choice(alphabet) for _ in range(48))
            self.assertTrue(looks_like_credential("sk-newprov-" + body), body)

    def test_screening_a_long_run_is_not_quadratic(self):
        # codex r14 #3 / r15 #2, #3: every position the screen admits as a
        # start is a position it scans the rest of the run from, and the
        # scan is synchronous — seconds of it block cancellation too.
        started = time.monotonic()
        for pathological in ("1234567890" * 6400, "a" * 64000,
                             "_" * 64000, "-" * 64000,
                             "\\cite" + " " * 32000 + "x",
                             "\\label" + "[a]" * 8000 + "{x}",
                             "sk-" * 8000,              # codex r16 #2
                             "\\cite[" * 32000,
                             "\\ref{sk-SobolevEstimates2024}" * 8000):
            looks_like_credential(pathological)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_a_long_citation_keeps_its_exemption(self):
        """codex r17 #2: caps meant for pathological input were low enough
        that a long postnote or a two-dozen-key citation list lost the
        exemption, and the citation KEY was then flagged."""
        postnote = "See the proof of the main theorem and its corollaries. " * 4
        self.assertFalse(looks_like_credential(
            "\\cite[" + postnote + "]{sk-SobolevEstimates2024}"))
        keys = ",".join(f"sk-SobolevEstimates20{i:02d}" for i in range(24))
        self.assertFalse(looks_like_credential("\\cite{" + keys + "}"))

    def test_an_oversized_identifier_is_judged_on_all_of_itself(self):
        # codex r17 #3: judging the capped prefix and redacting the whole
        # run let an over-long identifier be classified on a fragment.
        body = "DeligneMumfordCompactification2024" * 8 + "-for-elliptic-equations"
        self.assertFalse(looks_like_credential("\\label{sk-" + body + "}"))

    def test_screening_repeated_key_text_stays_linear(self):
        # codex r17 #1: extending each capped match walked the same run
        # again — 125kB took 2.8s with the event loop blocked.
        # codex r18 P3: and building each rejected match's argument was
        # unbounded work of its own — 4MB took 4s.
        chunk = "sk-proj-Ab3dEf9_hIjKlMn0-pQrStUvWx1Yz2-"
        started = time.monotonic()
        looks_like_credential(chunk * 102400)     # ~4MB
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_credential_inside_a_cross_reference_is_still_refused(self):
        """codex r15 #1: exempting cross-reference arguments outright let a
        real key ride into the shipped document inside \\label{...}."""
        import base64

        fake_codex_home(self)
        key = "sk-proj-" + "Ab3dEf9_hIjKlMn0-pQrStUvWx1Yz2" * 5
        head = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        jwt = f"{head}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        for hidden in (key, jwt):
            self.assertTrue(looks_like_credential("\\label{" + hidden + "}"))
            self.assertNotIn(hidden, scrub_credentials(
                "\\cite{" + hidden + "}", ()))
        patch_subprocess(self, message="```\nDOC \\label{" + key + "}\n```\n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_any_amount_of_leading_emphasis_is_trimmed(self):
        # codex r14 P3: a fixed four-character trim missed `____<jwt>____`.
        import base64

        head = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        jwt = f"{head}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        for lead in ("_", "____", "_" * 1000, "--", "_-_-"):
            self.assertTrue(looks_like_credential(f"x {lead}{jwt}"), lead)

    def test_a_word_prefixed_token_is_still_found(self):
        """codex r16 #1: `\\label{jwt_<JWT>}` put a whole word in front, and
        trimming only emphasis characters decoded the wrong header."""
        import base64

        head = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        jwt = f"{head}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        for wrapper in ("jwt_", "bearer-", "token_value_", "x-auth_"):
            self.assertTrue(
                looks_like_credential("\\label{" + wrapper + jwt + "}"),
                wrapper)
            self.assertNotIn(jwt, scrub_credentials(wrapper + jwt, ()))

    def test_a_key_longer_than_the_pattern_bound_is_redacted_whole(self):
        # The quantifiers are capped for cost; a longer key must still be
        # scrubbed to the end of its run, not up to the cap.
        key = "sk-proj-" + "Ab3dEf9hIjKlMn0pQrStUv" * 20
        self.assertTrue(looks_like_credential(key))
        self.assertNotIn(key[-12:], scrub_credentials(f"x {key} y", ()))

    def test_a_hyphenated_identifier_with_mixed_case_is_not_a_credential(self):
        # codex r13 #4: upper + lower + digit is satisfied by ordinary
        # LaTeX identifiers; a key's randomness is in ONE unbroken run.
        self.assertFalse(looks_like_credential(
            "\\label{sk-Sobolev-estimates-2024}"))
        self.assertFalse(looks_like_credential(
            "\\cite{sk-Sobolev-estimates-2024}"))
        self.assertFalse(looks_like_credential(
            "\\ref{sk-Lemma3-bound-2024b} and \\ref{sk-Thm7-case-1999}"))
        # An author's identifier inside a cross-reference is never a
        # credential, however random it looks (codex r14 P3).
        self.assertFalse(looks_like_credential(
            "\\cite{sk-SobolevEstimates2024}"))
        self.assertFalse(looks_like_credential(
            "\\label{sk-Sobolev-estimates-for-elliptic-equations-2024b}"))
        # ... and a long hyphenated identifier is not one in prose either.
        self.assertFalse(looks_like_credential(
            "see sk-Hardy-Littlewood-maximal-function-bounds-1930a below"))
        self.assertTrue(looks_like_credential("sk-ant-api03-Ab3dEf9hIjKlMn0p"))

    def test_a_jwt_with_a_spaced_header_is_refused(self):
        # codex r12 #2: RFC 7519 allows whitespace in the header JSON, so
        # the encoding need not begin "eyJ".
        import base64

        fake_codex_home(self)
        head = base64.urlsafe_b64encode(
            b'{ "alg": "HS256", "typ": "JWT" }').decode().rstrip("=")
        jwt = f"{head}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        self.assertTrue(looks_like_credential(f"token {jwt}"))
        patch_subprocess(self, message=f"```\nDOC {jwt}\n```\n")
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt"))

    def test_the_seat_itself_refuses_a_contaminated_reply(self):
        import json

        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        patch_subprocess(self, message=f"```\nDOC {secret}\n```\n")
        with self.assertRaises(CodexSeatError) as cm:
            asyncio.run(run_codex_seat("prompt"))
        # The complaint never names the value it found.
        self.assertNotIn(secret, str(cm.exception))
        # ... and the turn is still charged for what it spent.
        self.assertEqual(cm.exception.partial.metered_tokens, 1500)

    def test_a_credential_in_the_log_only_is_redacted_not_refused(self):
        import json

        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        patch_subprocess(self, message="```\nDOC\n```\n",
                         stdout=TURN_COMPLETED + f"\nnoise {secret}\n")
        result = asyncio.run(run_codex_seat("prompt"))
        self.assertIn("DOC", result.text)
        self.assertNotIn(secret, result.log_tail)
        self.assertIn("[redacted-codex-credential]", result.log_tail)
        # Usage is still parsed off the RAW log, not the redacted tail.
        self.assertEqual(result.metered_tokens, 1500)

    def test_a_contaminated_repair_keeps_its_UNABLE_declaration(self):
        """codex r11 #2: the candidate is discarded, but the catastrophe
        signal inside it is a completed declaration and must still bite."""
        fake_codex_home(self)
        token = "sk-proj-Zy8xWv7uTs6rQp5oNm4l"
        out, _prompts = self._drive([
            "```\n" + DOC + "```",               # rewrite, gated ok
            "ERRORS FOUND\nLemma 2 is false\n",  # cold referee
            f"```\n{DOC}```\nUNABLE: {token} the gap is real\n",
        ])
        self.assertEqual(out.document_text, "orig")
        self.assertIn("UNABLE", out.shipped)
        repair = [st for st in out.steps if st.step == "repair-r1"][0]
        self.assertIn("completed UNABLE", repair.detail)

    def test_a_declaration_after_the_fence_survives_an_open_listing(self):
        """codex r13 #2: stripping everything after an unterminated
        listing swallowed a real declaration written after the fence."""
        raw = ("```latex\n\\documentclass{article}\n\\begin{document}\n"
               "\\begin{verbatim}\nexample output\n```\n"
               "UNABLE: the lemma is false\n")
        _tex, declaration = _extract_tex(raw)
        self.assertTrue(_declares(declaration, "UNABLE:"))

    def test_a_declaration_after_end_document_inside_the_fence_survives(self):
        """codex r14 #2: text after \\end{document} is the model talking,
        even when the fence has not closed yet."""
        raw = ("```latex\n" + DOC.strip() + "\n"
               "\\begin{verbatim}\nUNABLE: the lemma is false\n```\n")
        _tex, declaration = _extract_tex(raw)
        self.assertTrue(_declares(declaration, "UNABLE:"))

    def test_a_listing_inside_a_truncated_reply_is_not_a_declaration(self):
        """codex r12 #3: a reply cut off inside a verbatim block had the
        block's own contents read as the model's declarations."""
        fake_codex_home(self)
        truncated = ("```\n\\documentclass{article}\n\\begin{document}\n"
                     "\\begin{verbatim}\n"
                     "UNABLE: output of the example algorithm\n")
        err = CodexSeatError("codex exec exited 1")
        err.partial = CodexSeatResult(text=truncated)
        script = ["```\n" + DOC + "```",
                  "ERRORS FOUND\nLemma 2 is false\n",
                  err]

        async def fake(prompt, **kwargs):
            item = script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return CodexSeatResult(text=item)

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE
        with mock_run_codex_seat(fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=1)))
        self.assertNotEqual(out.document_text, "orig")
        self.assertNotIn("UNABLE", out.shipped)

    def _cli_loop_run(self, loop, script, rounds, gates=None):
        async def fake(prompt, **kwargs):
            item = script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return CodexSeatResult(text=item)

        if gates is not None:
            queue = list(gates)
            loop._gate = lambda tex, inp, *a: (queue.pop(0) if queue
                                               else GOOD_GATE)
        with mock_run_codex_seat(fake):
            return asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=rounds)))

    def test_a_declaration_from_one_invocation_cannot_taint_the_next(self):
        """codex r13 #3: the pending reply lived on the instance and step
        names repeat, so a declaration already dealt with in one run could
        discard the next run's rewrite."""
        fake_codex_home(self)
        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        # Run one: repair-r1 arrives declaring UNABLE and then fails its
        # gate, so it is discarded with the declaration standing. The
        # original ships — correctly — and the reply stays on the node.
        first = self._cli_loop_run(loop, [
            "```\n" + DOC + "```",
            "ERRORS FOUND\nLemma 2 is false\n",
            "```\n" + DOC + "```\nUNABLE: the gap is real\n",
        ], rounds=1, gates=[GOOD_GATE, BAD_GATE])
        self.assertEqual(first.document_text, "orig")
        self.assertIn("UNABLE", first.shipped)

        # Run two: a repair-r1 that fails with nothing in hand. Nothing
        # from run one may reach it.
        second = self._cli_loop_run(loop, [
            "```\nFRESH\n```",
            "ERRORS FOUND\nLemma 2 is false\n",
            CodexSeatError("codex exec exited 1"),   # no partial at all
        ], rounds=1, gates=[GOOD_GATE])
        self.assertEqual(second.document_text, "FRESH\n")
        self.assertNotIn("UNABLE", second.shipped)

    def test_the_pending_reply_does_not_outlive_its_invocation(self):
        reply = ("repair-r1", "UNABLE: from this invocation")
        for error in (None, asyncio.CancelledError(), RuntimeError("interrupted")):
            with self.subTest(error=type(error).__name__):
                loop = make_loop()
                seen = []

                async def inner(inp, state):
                    loop._note_discarded_reply(*reply)
                    seen.append(loop._discarded_reply)
                    if error is not None:
                        raise error
                    return loop._ship(inp, state, "original")

                loop._run_inner = inner

                async def drive():
                    await loop.run(WriteupLoopInputs(document_text=DOC))
                    # Check before asyncio.run discards this task's context.
                    return loop._discarded_reply

                pending_after_run = asyncio.run(drive())
                self.assertEqual(seen, [reply])
                self.assertIsNone(pending_after_run)

    def test_updating_the_deadline_does_not_discard_a_completed_reply(self):
        fake_codex_home(self)
        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        async def update_then_cancel(name, result):
            if name == "repair-r1":
                loop._deadline += 60
                raise asyncio.CancelledError()

        loop._record_cli_usage = update_then_cancel
        out = self._cli_loop_run(loop, [
            "```\n" + DOC + "```",
            "ERRORS FOUND\nLemma 2 is false\n",
            "```\n" + DOC + "```\nUNABLE: the lemma is false\n",
        ], rounds=1)
        self.assertEqual(out.document_text, "orig")
        self.assertIn("UNABLE", out.shipped)

    def test_a_cancellation_while_accounting_a_GOOD_reply_keeps_it(self):
        """codex r13 #5: the reply arrived intact; the cancellation lands
        while its usage event is being emitted, before the loop sees it."""
        fake_codex_home(self)
        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        async def cancel_on_repair(name, result):
            if name.startswith("repair"):
                raise asyncio.CancelledError()

        loop._record_cli_usage = cancel_on_repair
        out = self._cli_loop_run(loop, [
            "```\n" + DOC + "```",
            "ERRORS FOUND\nLemma 2 is false\n",
            "```\n" + DOC + "```\nUNABLE: the lemma is false\n",
        ], rounds=1)
        self.assertEqual(out.document_text, "orig")
        self.assertIn("UNABLE", out.shipped)

    def test_a_cancellation_while_accounting_keeps_the_declaration(self):
        """codex r12 #4: the failure is accounted for over several awaits;
        a cancellation in one of them replaced the seat's exception, and
        the repair handler that reads the declaration never ran."""
        fake_codex_home(self)
        err = CodexSeatError("codex exec reply contained a credential")
        err.partial = CodexSeatResult(
            text="```\n" + DOC + "```\nUNABLE: the gap is real\n")
        script = ["```\n" + DOC + "```",
                  "ERRORS FOUND\nLemma 2 is false\n",
                  err]

        async def fake(prompt, **kwargs):
            item = script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return CodexSeatResult(text=item)

        loop = make_loop()
        loop.seat = "cli"
        loop._gate = lambda tex, inp, *a: GOOD_GATE

        async def cancel_on_repair(name, result):
            if name.startswith("repair"):
                raise asyncio.CancelledError()

        loop._record_cli_usage = cancel_on_repair
        with mock_run_codex_seat(fake):
            out = asyncio.run(loop.run(
                WriteupLoopInputs(document_text="orig", rounds=1)))
        self.assertEqual(out.document_text, "orig")
        self.assertIn("UNABLE", out.shipped)

    def test_a_failed_repair_without_a_declaration_taints_nothing(self):
        fake_codex_home(self)
        token = "sk-proj-Zy8xWv7uTs6rQp5oNm4l"
        out, _prompts = self._drive([
            "```\n" + DOC + "```",
            "ERRORS FOUND\nLemma 2 is false\n",
            f"```\n{DOC}```\nFIXED: {token} rewrote the step\n",
        ])
        self.assertNotEqual(out.document_text, "orig")
        self.assertNotIn("UNABLE", out.shipped)

    def test_a_refused_reply_still_leaves_a_redacted_transcript(self):
        import json
        import tempfile
        from unittest import mock

        fake_codex_home(self)
        secret = json.loads(SUBSCRIPTION_AUTH)["tokens"]["access_token"]
        err = CodexSeatError("reply contained a local codex credential")
        err.partial = CodexSeatResult(
            text=f"```\nX {secret}\n```", log_tail="tail")

        async def fake(prompt, **kwargs):
            raise err

        with tempfile.TemporaryDirectory() as tmp:
            loop = make_loop()
            loop.seat = "cli"
            loop._gate = lambda tex, inp, *a: GOOD_GATE
            with mock.patch.object(WriteupLoop, "workdir",
                                   property(lambda self: Path(tmp))), \
                 mock.patch("proofstack.agents.writeup_loop.run_codex_seat",
                            fake):
                out = asyncio.run(loop.run(
                    WriteupLoopInputs(document_text="orig", rounds=0)))
            reply = (Path(tmp) / "cli-seats" / "rewrite-a1.reply.txt").read_text(
                encoding="utf-8")
        self.assertNotIn(secret, reply)
        self.assertIn("[redacted-codex-credential]", reply)
        self.assertEqual(out.document_text, "orig")

    def test_ordinary_replies_are_returned_unchanged(self):
        fake_codex_home(self)
        out, _prompts = self._drive([
            "```\nPLAIN\n```", "NO ERRORS\n", "```\nPLAINER\n```\n"])
        self.assertEqual(out.document_text, "PLAINER\n")


class TestRedactionSurvivesATokenRefresh(unittest.TestCase):
    """codex r10 #2 / r11 #1: the redactor read only the CURRENT auth.json,
    so a token codex rotated during the turn was invisible — including one
    rotated IN and OUT again while the turn ran."""

    def test_both_the_old_and_the_new_credential_are_scrubbed(self):
        home = fake_codex_home(self)
        rotated = SUBSCRIPTION_AUTH.replace("tok-aaaaaaaa", "tok-dddddddd")
        (home / "auth.json").write_text(rotated, encoding="utf-8")
        got = redact_seat_text("old tok-aaaaaaaa new tok-dddddddd",
                               extra_secrets=("tok-aaaaaaaa",))
        self.assertNotIn("tok-aaaaaaaa", got)
        self.assertNotIn("tok-dddddddd", got)

    def test_an_unreadable_auth_file_still_scrubs_the_snapshot(self):
        fake_codex_home(self, auth_json=None)   # no auth.json at all
        got = redact_seat_text("leaked tok-aaaaaaaa",
                               extra_secrets=("tok-aaaaaaaa",))
        self.assertEqual(got, "leaked [redacted-codex-credential]")

    def test_auth_secrets_never_raises(self):
        fake_codex_home(self, auth_json="{not json")
        self.assertEqual(auth_secrets(), ())

    def test_a_pathological_auth_file_does_not_raise_either(self):
        # codex r11 #8: extraction, not just the read, must be guarded.
        fake_codex_home(self, auth_json="[" * 20000 + "]" * 20000)
        self.assertEqual(auth_secrets(), ())
        self.assertEqual(auth_secrets(additional=("keep-this-secret",)),
                         ("keep-this-secret",))

    def _rotating_run(self, home, *, final_auth, echoed):
        """A turn that rewrites auth.json twice while it runs and echoes the
        INTERMEDIATE credential — the value a before/after pair of reads
        never sees."""
        from unittest import mock

        second = SUBSCRIPTION_AUTH.replace("tok-aaaaaaaa", "tok-bbbbbbbbbb")

        class RotatingProc(FakeProc):
            async def communicate(self, payload=None):
                # The watcher's FIRST read sees only the original token, so
                # nothing but a repeated poll can catch what follows
                # (codex r11: the old test did not pin the polling).
                await asyncio.sleep(0.08)
                (home / "auth.json").write_text(second, encoding="utf-8")
                await asyncio.sleep(0.15)      # long enough to be observed
                if final_auth is None:
                    (home / "auth.json").unlink()
                else:
                    (home / "auth.json").write_text(final_auth,
                                                    encoding="utf-8")
                return await super().communicate(payload)

        async def fake_exec(*cmd, **kwargs):
            return RotatingProc(cmd, message=echoed, stdout=TURN_COMPLETED,
                                returncode=0)

        with mock.patch("asyncio.create_subprocess_exec", fake_exec), \
             mock.patch(
                 "proofstack.agents.writeup_codex_seat._AUTH_POLL_SECONDS",
                 0.01), \
             mock.patch(
                 "proofstack.agents.writeup_codex_seat.resolve_codex_bin",
                 lambda raw=None: "/fake/codex"), \
             mock.patch(
                 "proofstack.agents.writeup_codex_seat.kill_seat_survivors",
                 lambda pgid, marker: 0), \
             mock.patch("proofstack.agents.writeup_codex_seat.os.killpg",
                        lambda pgid, sig: None):
            return asyncio.run(run_codex_seat("prompt"))

    def test_a_credential_rotated_in_and_out_mid_turn_is_still_caught(self):
        home = fake_codex_home(self)
        third = SUBSCRIPTION_AUTH.replace("tok-aaaaaaaa", "tok-cccccccccc")
        with self.assertRaises(CodexSeatError) as cm:
            self._rotating_run(home, final_auth=third,
                               echoed="```\nDOC tok-bbbbbbbbbb\n```\n")
        self.assertIn("credential", str(cm.exception))
        self.assertNotIn("tok-bbbbbbbbbb", str(cm.exception))

    def test_it_is_caught_even_if_the_auth_file_ends_up_unreadable(self):
        home = fake_codex_home(self)
        with self.assertRaises(CodexSeatError):
            self._rotating_run(home, final_auth=None,
                               echoed="```\nDOC tok-bbbbbbbbbb\n```\n")

    def test_a_clean_reply_across_a_rotation_still_succeeds(self):
        home = fake_codex_home(self)
        third = SUBSCRIPTION_AUTH.replace("tok-aaaaaaaa", "tok-cccccccccc")
        result = self._rotating_run(home, final_auth=third,
                                    echoed="```\nDOC\n```\n")
        self.assertIn("DOC", result.text)


class TestStageIdentity(unittest.TestCase):
    """codex r10 #3 / r11 #2 / r11 #6: every chain stage ran seats named
    rewrite-a1, rewrite-a2, ..., so the resume cache — keyed on (class,
    name, config, inputs) — replayed an earlier stage's failed candidates
    whenever a stage passed its input through unchanged. The stage goes
    into the seat's cache KEY, not its name: names are what
    ``model_for``/``component_config_for`` resolve overrides by."""

    def setUp(self):
        _STAGE_CLAIMED.clear()
        self.addCleanup(_STAGE_CLAIMED.clear)

    def _loop(self, name, run_id="run-1"):
        import types

        loop = make_loop()
        loop.name = name
        loop.ctx = types.SimpleNamespace(run_id=run_id)
        return loop

    def test_distinct_stages_get_distinct_tags(self):
        tags = {self._loop(f"writeup_loop_{i}")._stage_tag()
                for i in (1, 2, 3, 4)}
        self.assertEqual(len(tags), 4)

    def test_a_first_claim_depends_only_on_the_name(self):
        # So an ordinary run and its resume produce the same cache keys.
        _STAGE_CLAIMED.clear()
        first = self._loop("writeup_loop_1")._stage_tag()
        _STAGE_CLAIMED.clear()
        self.assertEqual(self._loop("writeup_loop_1")._stage_tag(), first)

    def test_stages_sharing_one_name_never_share_a_tag(self):
        # A chain whose nodes were left unnamed: every stage is
        # "WriteupLoop" to the framework (codex r11 #2).
        tags = [self._loop("WriteupLoop")._stage_tag() for _ in range(3)]
        self.assertEqual(len(set(tags)), 3)

    def test_a_repeat_claim_cannot_collide_with_a_real_name(self):
        # codex r11 #4: "polish" claimed twice must not encode to the tag
        # of a stage genuinely named "polish#1".
        tags = [self._loop("polish")._stage_tag(),
                self._loop("polish")._stage_tag(),
                self._loop("polish#1")._stage_tag()]
        self.assertEqual(len(set(tags)), 3)

    def test_the_tag_is_claimed_once_per_instance(self):
        loop = self._loop("WriteupLoop")
        self.assertEqual(loop._stage_tag(), loop._stage_tag())

    def test_a_full_register_stops_issuing_bare_tags_rather_than_forget(self):
        """codex r12 #6: pruning another run's claims because its id
        differs assumes it has finished; a run still in flight would then
        reissue a bare tag it has already used."""
        from proofstack.agents import writeup_loop as module

        live_other = self._loop("polish", run_id="run-A")._stage_tag()
        for i in range(module._STAGE_CLAIMED_MAX + 2):
            _STAGE_CLAIMED.add((f"other-run-{i}", "WriteupLoop"))
        # run-A is still going: its next stage must not be handed run-A's
        # first stage's identity.
        again = self._loop("polish", run_id="run-A")._stage_tag()
        self.assertNotEqual(again, live_other)
        tags = [self._loop("WriteupLoop", run_id="live")._stage_tag()
                for _ in range(3)]
        self.assertEqual(len(set(tags)), 3)

    def test_a_different_run_starts_its_claims_again(self):
        a = self._loop("WriteupLoop", run_id="run-1")._stage_tag()
        b = self._loop("WriteupLoop", run_id="run-2")._stage_tag()
        self.assertEqual(a, b)

    def test_the_seat_keeps_its_plain_name_so_overrides_still_resolve(self):
        seen = {}

        class FakeSeat:
            def __init__(self, ctx, name=None, parent_budget_scope=None):
                seen["name"] = name
                self.wallclock_cap_s = None
                self.stage = ""

            async def __call__(self, prompt):
                seen["stage"] = self.stage
                return type("O", (), {"text": "ok"})()

        loop = self._loop("writeup_loop_3")
        loop._deadline = time.monotonic() + 3600
        asyncio.run(loop._call_seat(FakeSeat, "rewrite-a1", "p"))
        self.assertEqual(seen["name"], "rewrite-a1")
        self.assertEqual(seen["stage"], loop._stage_tag())

    def _seat(self, stage):
        seat = RewriteSeat.__new__(RewriteSeat)
        seat.name = "rewrite-a1"
        seat.component_config = {}
        seat.stage = stage
        return seat

    def test_the_same_prompt_from_two_stages_has_two_cache_keys(self):
        inp = _SeatInputs(prompt="identical prompt")
        keys = {self._seat(self._loop(n)._stage_tag())._cache_key(inp)
                for n in ("writeup_loop_1", "writeup_loop_2")}
        self.assertEqual(len(keys), 2)

    def test_one_stage_keeps_one_cache_key_so_resume_still_hits(self):
        inp = _SeatInputs(prompt="identical prompt")
        tag = self._loop("writeup_loop_1")._stage_tag()
        self.assertEqual(self._seat(tag)._cache_key(inp),
                         self._seat(tag)._cache_key(inp))

    def test_an_unstaged_seat_keeps_the_frameworks_own_key(self):
        inp = _SeatInputs(prompt="p")
        seat = self._seat("")
        self.assertEqual(seat._cache_key(inp),
                         APICallAgent._cache_key(seat, inp))

    def test_the_real_resume_cache_does_not_replay_across_stages(self):
        """The reproduction, against the framework's own cache: the same
        seat name and the same prompt, from two stages."""
        import tempfile

        from proofstack.context import RunContext

        runs = {"n": 0}

        class CountingSeat(_OneShotSeat):
            Inputs = _SeatInputs
            Outputs = _SeatOutputs

            async def run(self, inp):
                runs["n"] += 1
                return _SeatOutputs(text=f"sample {runs['n']}")

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(run_id="stage-test",
                                    root_workdir=Path(tmp), flat=True)
            for stage in (self._loop("stage_1")._stage_tag(),
                          self._loop("stage_2")._stage_tag()):
                seat = CountingSeat(ctx, name="rewrite-a1")
                seat.stage = stage
                asyncio.run(seat(prompt="identical prompt"))
            self.assertEqual(runs["n"], 2)
            # Same stage twice: the resume cache is still allowed to work.
            stage_1 = self._loop("stage_1", run_id="again")._stage_tag()
            for _ in range(2):
                seat = CountingSeat(ctx, name="rewrite-a1")
                seat.stage = stage_1
                asyncio.run(seat(prompt="identical prompt"))
            self.assertEqual(runs["n"], 2)

    def test_step_records_keep_the_plain_step_name(self):
        # The stage tag is seat identity, not the decision trail: the
        # steps list is part of the node's output contract.
        loop = self._loop("writeup_loop_2")
        out, _calls = run_loop(loop, ["```\n" + DOC + "```", "NO ERRORS\n",
                                      "```\n" + DOC + "```"], rounds=1)
        self.assertEqual(out.steps[0].step, "rewrite-a1")


class TestSeatCleanupOnEveryExitPath(unittest.TestCase):
    """codex r10 #4 / r11 #3 / r11 #4: cleanup ran only on the timeout and
    cancellation paths, so a descendant that did not hold the output pipe
    survived a clean exit. After the leader is reaped its pid number is the
    kernel's to reissue, so survivors are identified per process rather
    than signalled by that number."""

    def _dummy_cli(self, exit_code, *, detach=False):
        """A stand-in codex that leaves a child behind. The child redirects
        its own output, so communicate() returns as soon as the leader
        exits; with ``detach`` it also calls setsid and so leaves the
        process group entirely."""
        import tempfile
        from unittest import mock

        fake_codex_home(self)
        tmp = Path(tempfile.mkdtemp(prefix="writeup_dummy_cli_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pidfile = tmp / "child.pid"
        def fake_cmd(*, last_message_path, **kwargs):
            script = (
                "import pathlib, subprocess, sys; "
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(120)'], stdout=subprocess.DEVNULL, "
                f"stderr=subprocess.DEVNULL, start_new_session={detach}); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
                "pathlib.Path(sys.argv[2]).write_text('ok\\n'); "
                f"sys.exit({exit_code})"
            )
            return [sys.executable, "-c", script, str(pidfile), str(last_message_path)]

        patcher = mock.patch(
            "proofstack.agents.writeup_codex_seat.build_codex_cmd", fake_cmd)
        patcher.start()
        self.addCleanup(patcher.stop)
        bin_patcher = mock.patch(
            "proofstack.agents.writeup_codex_seat.resolve_codex_bin",
            lambda raw=None: "/bin/sh")
        bin_patcher.start()
        self.addCleanup(bin_patcher.stop)
        return pidfile

    def _assert_child_is_gone(self, pidfile):
        import os

        pid = int(pidfile.read_text().strip())
        self.addCleanup(_kill_pid, pid)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:  # recycled onto another user's pid
                return
            time.sleep(0.05)
        self.fail(f"descendant {pid} outlived the seat call")

    def test_a_clean_exit_does_not_leave_a_child_running(self):
        pidfile = self._dummy_cli(0)
        result = asyncio.run(run_codex_seat("prompt", timeout_s=30))
        self.assertIn("ok", result.text)
        self._assert_child_is_gone(pidfile)

    def test_a_failed_exit_does_not_leave_a_child_running_either(self):
        pidfile = self._dummy_cli(3)
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt", timeout_s=30))
        self._assert_child_is_gone(pidfile)

    def test_a_child_that_left_the_process_group_is_caught_too(self):
        # codex r11 #4: a group kill alone never reaches this one.
        pidfile = self._dummy_cli(0, detach=True)
        asyncio.run(run_codex_seat("prompt", timeout_s=30))
        self._assert_child_is_gone(pidfile)

    def test_no_group_is_signalled_by_number_after_a_clean_exit(self):
        """r11 #3: the saved pgid may belong to somebody else by now."""
        from unittest import mock

        box = patch_subprocess(self, message="```\nDOC\n```\n")
        asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(box["killpg_calls"], [])
        self.assertEqual([pgid for pgid, _marker in box["survivor_calls"]],
                         [box["proc"].pid])

    def test_a_reaped_leaders_group_number_is_not_signalled_on_timeout(self):
        """codex r12 #7: communicate() can also time out after the leader
        exited, and by then the number may be somebody else's — for as
        long as the rest of the timeout."""
        box = patch_subprocess(self, hang=True, exit_then_hang=True)
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt", timeout_s=0.1))
        self.assertEqual(box["killpg_calls"], [])
        self.assertTrue(box["survivor_calls"])

    def test_the_timeout_path_still_kills_the_group_outright(self):
        # There the leader is alive and unreaped, so its number is still
        # its own and the group kill is both safe and the fastest stop.
        box = patch_subprocess(self, hang=True)
        with self.assertRaises(CodexSeatError):
            asyncio.run(run_codex_seat("prompt", timeout_s=0.1))
        self.assertEqual(box["killpg_calls"], [box["proc"].pid])
        self.assertTrue(box["proc"].killed)   # killpg refused; fell back

    def test_the_marker_is_passed_to_the_child_and_to_the_sweep(self):
        import os

        box = patch_subprocess(self, message="```\nDOC\n```\n")
        asyncio.run(run_codex_seat("prompt"))
        marker = box["kwargs"]["env"][SEAT_MARKER_ENV]
        self.assertTrue(marker)
        self.assertEqual(box["survivor_calls"][0][1], marker)
        # Never leaked into this process's own environment.
        self.assertNotIn(SEAT_MARKER_ENV, os.environ)

    def test_survivors_are_found_by_group_and_by_marker(self):
        import os

        marker = "writeup-test-marker-abcdef"
        env = dict(os.environ, **{SEAT_MARKER_ENV: marker})
        leader = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys, time; "
             "child = subprocess.Popen([sys.executable, '-c', "
             "'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, "
             "stderr=subprocess.DEVNULL, start_new_session=True); "
             "print(child.pid, flush=True); time.sleep(60)"],
            env=env, stdout=subprocess.PIPE, start_new_session=True)
        self.addCleanup(_reap, leader)
        detached = int(leader.stdout.readline().decode().strip())
        self.addCleanup(_kill_pid, detached)
        found = seat_survivors(leader.pid, marker)
        self.assertIn(leader.pid, found)        # same group and session
        self.assertIn(detached, found)          # left the group; marked
        self.assertNotIn(os.getpid(), found)

    def test_a_wrong_marker_and_a_foreign_group_match_nothing(self):
        self.assertEqual(seat_survivors(-12345, "no-such-marker-xyz"), [])

    def test_process_enumeration_errors_do_not_escape_cleanup(self):
        from unittest import mock

        import psutil
        from proofstack.agents import writeup_codex_seat as seat_mod

        for error in (OSError("unavailable"), psutil.Error("unavailable"),
                      RuntimeError("unavailable")):
            with self.subTest(error=type(error).__name__), \
                 mock.patch.object(seat_mod, "_proc_fs_available", return_value=False), \
                 mock.patch.object(psutil, "pids", side_effect=error):
                self.assertEqual(seat_survivors(4242, "ours"), [])
                self.assertEqual(kill_seat_survivors(4242, "ours"), 0)

    def test_successful_seat_survives_a_psutil_cleanup_error(self):
        from unittest import mock

        import psutil
        from proofstack.agents import writeup_codex_seat as seat_mod

        box = patch_subprocess(self, message="OK")
        with mock.patch.object(seat_mod, "kill_seat_survivors", kill_seat_survivors), \
             mock.patch.object(seat_mod, "_proc_fs_available", return_value=False), \
             mock.patch.object(psutil, "pids", side_effect=psutil.Error("unavailable")):
            result = asyncio.run(run_codex_seat("prompt"))
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.metered_tokens, 1500)
        self.assertEqual(box["proc"].returncode, 0)

    def test_without_proc_survivors_need_a_matching_session_or_marker(self):
        import os
        from unittest import mock

        import psutil
        from proofstack.agents import writeup_codex_seat as seat_mod

        def group(pid):
            if pid == 14:
                raise PermissionError("unreadable process")
            return 4242 if pid in (11, 13) else 999

        def process(pid):
            if pid == 14:
                raise psutil.AccessDenied(pid)
            return mock.Mock(environ=lambda: {
                SEAT_MARKER_ENV: "ours" if pid == 12 else "foreign"})

        with mock.patch.object(seat_mod, "_proc_fs_available", return_value=False) as probe, \
             mock.patch.object(psutil, "pids", return_value=[os.getpid(), 11, 12, 13, 14]), \
             mock.patch.object(psutil, "Process", side_effect=process), \
             mock.patch.object(seat_mod.os, "getpgid", side_effect=group), \
             mock.patch.object(seat_mod.os, "getsid", side_effect=lambda pid: 4242 if pid == 11 else 999):
            self.assertEqual(seat_survivors(4242, "ours"), [11, 12])
        probe.assert_called_once_with()

    def test_proc_scan_still_uses_linux_process_identity(self):
        import os
        from unittest import mock
        from proofstack.agents import writeup_codex_seat as seat_mod

        with mock.patch.object(seat_mod, "_proc_fs_available", return_value=True) as probe, \
             mock.patch.object(seat_mod.os, "listdir", return_value=[str(os.getpid()), "self", "11", "12", "13"]), \
             mock.patch.object(seat_mod, "_in_seat_session", side_effect=lambda pid, pgid, **kw: pid == 11), \
             mock.patch.object(seat_mod, "_carries_marker", side_effect=lambda pid, marker, **kw: pid == 12):
            self.assertEqual(seat_survivors(4242, "ours"), [11, 12])
        probe.assert_called_once_with()

    def test_the_sweep_runs_again_for_children_forked_mid_cleanup(self):
        """codex r11 #7: a survivor can fork between the scan and its own
        death, and that child is in no snapshot taken so far."""
        from unittest import mock

        from proofstack.agents import writeup_codex_seat as seat_mod

        # Four generations: a three-pass sweep would leave the last.
        scans, killed = [[11], [12], [13], [14], []], []
        with mock.patch.object(seat_mod, "seat_survivors",
                               lambda pgid, marker: scans.pop(0)), \
             mock.patch.object(seat_mod, "_is_seat_survivor",
                               lambda pid, pgid, marker: True), \
             mock.patch.object(seat_mod.os, "kill",
                               lambda pid, sig: killed.append(pid)):
            signalled = kill_seat_survivors(4242, "marker")
        self.assertEqual(killed, [11, 12, 13, 14])
        self.assertEqual(signalled, 4)

    def test_a_pid_that_stopped_being_ours_is_not_signalled(self):
        """The scan-to-kill window: re-identified immediately before."""
        from unittest import mock

        from proofstack.agents import writeup_codex_seat as seat_mod

        killed = []
        with mock.patch.object(seat_mod, "seat_survivors",
                               lambda pgid, marker: [11]), \
             mock.patch.object(seat_mod, "_is_seat_survivor",
                               lambda pid, pgid, marker: False), \
             mock.patch.object(seat_mod.os, "kill",
                               lambda pid, sig: killed.append(pid)):
            signalled = kill_seat_survivors(4242, "marker", passes=1)
        self.assertEqual(killed, [])
        self.assertEqual(signalled, 0)

    def test_a_failing_auth_watcher_cannot_skip_the_cleanup(self):
        """codex r11 #8: the watcher is awaited in the same finally, so an
        exception from it used to leave the sweep unrun."""
        from unittest import mock

        async def exploding_watcher(env, seen, interval=None):
            raise RecursionError("pathological auth.json")

        box = patch_subprocess(self, message="```\nDOC\n```\n")
        with mock.patch(
                "proofstack.agents.writeup_codex_seat._watch_auth",
                exploding_watcher):
            result = asyncio.run(run_codex_seat("prompt"))
        self.assertIn("DOC", result.text)
        self.assertEqual(result.metered_tokens, 1500)
        self.assertTrue(box["survivor_calls"])


def _kill_pid(pid):
    """Cleanup for a test-spawned process we do not own."""
    import os

    with contextlib.suppress(Exception):
        os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()


# --- presets -------------------------------------------------------------

def test_presets_validate_and_chain4_feeds_each_stage_the_previous_output():
    from proofstack.registry import load_preset
    from app.dev_data import validate_preset_yaml

    for name in ("writeup_loop", "writeup_loop_chain4"):
        preset = load_preset(name)
        report = validate_preset_yaml(preset.source_path.read_text())
        assert report["ok"], (name, report["errors"])
        assert preset.component_configs["WriteupLoop"]["seat"] == "api"

    chain = load_preset("writeup_loop_chain4")
    nodes = chain.raw["dag"]["nodes"]
    assert [n["id"] for n in nodes] == ["polish1", "polish2", "polish3", "polish4"]
    assert nodes[0]["inputs"]["document_text"] == "$input.document_text"
    for prev, node in zip(nodes, nodes[1:]):
        assert node["needs"] == [prev["id"]]
        assert node["inputs"]["document_text"] == f"$node.{prev['id']}.document_text"
    # Distinct node names are what keeps one stage's seats (and so its
    # resume-cache entries) from standing in for another's (codex r10 #3).
    assert len({n["name"] for n in nodes}) == len(nodes)
    assert chain.raw["dag"]["outputs"]["document_text"] == "$node.polish4.document_text"
    # Four stages at the node's own default budget (20 USD, 7200 s).
    assert chain.budget.max_usd == 80.0
    assert chain.budget.max_wallclock_s == 28800
