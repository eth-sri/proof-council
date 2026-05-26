from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.ac_workflow import (  # noqa: E402
    ACWorkflow,
    _CompileResult,
    _simple_compile_latex,
)
from proofstack.budget import BudgetExhausted  # noqa: E402
from proofstack.latex_contract import normalize_firstproof_latex  # noqa: E402


class ACLatexContractTests(unittest.TestCase):
    def test_author_compile_feedback_normalizes_answer_and_reports_overflow(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)
                (workspace / "answer.tex").write_text(
                    "\\documentclass[11pt]{article}\n"
                    "\\usepackage{geometry}\n"
                    "\\begin{document}X\\end{document}",
                    encoding="utf-8",
                )
                workflow = object.__new__(ACWorkflow)

                def fake_compile(
                    tex_body: str,
                    *,
                    bib_path: Path | None,
                    page_limit: int,
                    is_full_document: bool,
                ) -> _CompileResult:
                    removals: list[str] = []
                    normalized = normalize_firstproof_latex(tex_body, removals=removals)
                    return _CompileResult(
                        tex=normalized,
                        tex_path=None,
                        pdf_path=None,
                        compiled=True,
                        pages=13,
                        compile_log="fake compile ok",
                        normalization_removals=removals,
                    )

                with patch("proofstack.agents.ac.ac_workflow._simple_compile_latex", fake_compile):
                    feedback = await workflow._compile_feedback_after_author(
                        workspace,
                        page_limit=12,
                        round=2,
                    )

                answer_tex = (workspace / "answer.tex").read_text(encoding="utf-8")
                self.assertIn("\\documentclass[12pt]{article}", answer_tex)
                self.assertNotIn("geometry", answer_tex)
                self.assertIn("normalized answer.tex", feedback)
                self.assertIn("above the First Proof limit of 12", feedback)

                log_text = (workspace / ".ac" / "compile-round-2.log").read_text(
                    encoding="utf-8"
                )
                self.assertIn("compiled: True", log_text)
                self.assertIn("pages: 13", log_text)
                self.assertIn("page_limit: 12", log_text)
                self.assertIn("fake compile ok", log_text)

        asyncio.run(run_check())

    def test_last_gasp_budget_exhaustion_returns_firstproof_latex_contract(self) -> None:
        async def run_check() -> None:
            workflow = object.__new__(ACWorkflow)
            out = await workflow._last_gasp_finalize(
                "P",
                "Bare emergency draft.",
                error=BudgetExhausted("run", "usd", 1.0, 1.0),
            )

            self.assertIn("\\documentclass[12pt]{article}", out)
            self.assertNotIn("\\documentclass[11pt]{article}", out)
            self.assertIn("Bare emergency draft.", out)

        asyncio.run(run_check())

    @unittest.skipIf(
        shutil.which("pdflatex") is None or shutil.which("pdfinfo") is None,
        "pdflatex/pdfinfo not installed",
    )
    def test_compile_helper_counts_pages_without_fitz(self) -> None:
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "Page one.\\newpage Page two.\n"
            "\\end{document}\n"
        )

        with patch.dict(sys.modules, {"fitz": None}):
            out = _simple_compile_latex(
                tex,
                bib_path=None,
                page_limit=12,
                is_full_document=True,
            )

        self.assertTrue(out.compiled)
        self.assertEqual(out.pages, 2)


if __name__ == "__main__":
    unittest.main()
