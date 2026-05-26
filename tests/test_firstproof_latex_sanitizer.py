"""Lock the behaviour of the LaTeX sanitizer + pending-solution fallback.

The First Proof spec requires each solution to be a standard 12pt
article with no margin/spacing changes. ``_strip_forbidden_packages``
+ ``_normalize_documentclass`` enforce that on the workflow's final
output. The trickiest case is shared-option ``\\usepackage`` lines:
``\\usepackage[margin=1in]{geometry,amsmath}`` -- if we drop geometry
but keep amsmath, the ``margin=1in`` would silently re-attach to
amsmath and produce invalid LaTeX. These tests pin the safer behaviour
(drop the whole line in that case) and a handful of the other paths.
"""
from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "firstproof_entrypoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_firstproof_entrypoint_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ep = _load_module()


class StripForbiddenPackagesTests(unittest.TestCase):
    def test_drops_solitary_forbidden_package(self) -> None:
        tex = "\\documentclass{article}\n\\usepackage{geometry}\n\\begin{document}X\\end{document}"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertNotIn("geometry", cleaned)
        self.assertTrue(any("geometry" in r for r in removals))

    def test_keeps_safe_packages_on_unforbidden_line(self) -> None:
        tex = "\\usepackage{amsmath,amssymb,amsthm}\n"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertEqual(cleaned, tex)
        self.assertEqual(removals, [])

    def test_drops_only_forbidden_names_when_no_options(self) -> None:
        tex = "\\usepackage{amsmath,geometry,amssymb}\n"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertIn("amsmath", cleaned)
        self.assertIn("amssymb", cleaned)
        self.assertNotIn("geometry", cleaned)
        self.assertTrue(any("geometry" in r for r in removals))

    def test_keeps_survivors_without_options_when_forbidden_package_has_options(self) -> None:
        """``[margin=1in]`` is geometry-specific. We drop the forbidden
        package and its options but re-emit surviving math packages
        WITHOUT options (math packages accept no options, so this is
        a no-op for them; for any other survivor the worst case is
        losing a tuning knob, not compilation).
        """
        tex = "\\documentclass{article}\n\\usepackage[margin=1in]{geometry,amsmath}\n\\begin{document}X\\end{document}"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        # Forbidden bits are gone.
        self.assertNotIn("geometry", cleaned)
        self.assertNotIn("margin=1in", cleaned)
        # But amsmath survives, without the options.
        self.assertIn("\\usepackage{amsmath}", cleaned)
        # Removal log explains what happened.
        self.assertTrue(
            any("re-emitted survivors without options" in r for r in removals)
        )

    def test_drops_whole_line_when_only_forbidden_package_has_options(self) -> None:
        tex = "\\usepackage[showframe]{geometry}\n"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertEqual(cleaned.strip(), "")
        self.assertTrue(any("geometry" in r for r in removals))

    def test_fullpage_is_allowed_but_setspace_is_dropped(self) -> None:
        tex = "\\usepackage{fullpage}\n\\usepackage{setspace}\n"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertIn("fullpage", cleaned)
        self.assertNotIn("setspace", cleaned)
        self.assertEqual(len(removals), 1)

    def test_fullpage_options_are_stripped(self) -> None:
        tex = "\\usepackage[cm]{fullpage}\n"
        cleaned, removals = ep._strip_forbidden_packages(tex)
        self.assertEqual(cleaned.strip(), "\\usepackage{fullpage}")
        self.assertTrue(any("fullpage" in r for r in removals))


class NormalizeDocumentClassTests(unittest.TestCase):
    def test_strips_twocolumn_option(self) -> None:
        tex = "\\documentclass[11pt,twocolumn]{article}\n\\begin{document}X\\end{document}"
        removals: list[str] = []
        out = ep._normalize_documentclass(tex, removals=removals)
        self.assertIn("\\documentclass[12pt]{article}", out)
        self.assertNotIn("twocolumn", out)
        self.assertTrue(any("twocolumn" in r for r in removals))

    def test_strips_landscape_option(self) -> None:
        tex = "\\documentclass[landscape]{article}\n\\begin{document}X\\end{document}"
        out = ep._normalize_documentclass(tex, removals=[])
        self.assertNotIn("landscape", out)

    def test_rewrites_amsart_to_article(self) -> None:
        tex = "\\documentclass{amsart}\n\\begin{document}X\\end{document}"
        removals: list[str] = []
        out = ep._normalize_documentclass(tex, removals=removals)
        self.assertIn("\\documentclass[12pt]{article}", out)
        self.assertNotIn("amsart", out)
        self.assertTrue(any("amsart" in r for r in removals))

    def test_strips_all_non_12pt_class_options(self) -> None:
        tex = "\\documentclass[reqno,a4paper,11pt]{amsart}\n\\begin{document}X\\end{document}"
        out = ep._normalize_documentclass(tex, removals=[])
        self.assertTrue(out.startswith("\\documentclass[12pt]{article}"))
        self.assertNotIn("reqno", out)
        self.assertNotIn("a4paper", out)
        self.assertNotIn("11pt", out)


class EnsureCompleteLatexTests(unittest.TestCase):
    def test_wraps_bare_text(self) -> None:
        out = ep._ensure_complete_latex("Just some text.")
        self.assertIn("\\documentclass[12pt]{article}", out)
        self.assertIn("\\begin{document}", out)
        self.assertIn("\\end{document}", out)
        self.assertIn("Just some text.", out)

    def test_threads_removals_list(self) -> None:
        tex = (
            "\\documentclass[twocolumn]{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\begin{document}X\\end{document}"
        )
        removals: list[str] = []
        out = ep._ensure_complete_latex(tex, removals=removals)
        self.assertNotIn("twocolumn", out)
        self.assertNotIn("geometry", out)
        self.assertTrue(removals, "expected normalization to surface removals")

    def test_fallback_strips_forbidden_usepackage_without_docclass(self) -> None:
        """Regression: a partial document opening with
        ``\\usepackage[margin=1in]{geometry}\\begin{document}...`` had
        no ``\\documentclass`` so the wrapper used to embed the body
        verbatim, smuggling geometry into the final .tex."""
        tex = "\\usepackage[margin=1in]{geometry}\\begin{document}Some proof body.\\end{document}"
        removals: list[str] = []
        out = ep._ensure_complete_latex(tex, removals=removals)
        # Forbidden package must be gone.
        self.assertNotIn("geometry", out)
        self.assertNotIn("margin=1in", out)
        # Standard preamble must be applied.
        self.assertIn("\\documentclass[12pt]{article}", out)
        # The body content survives.
        self.assertIn("Some proof body.", out)
        # Exactly one begin/end pair (no doubled-up markers).
        self.assertEqual(out.count("\\begin{document}"), 1)
        self.assertEqual(out.count("\\end{document}"), 1)
        self.assertTrue(removals, "expected fallback path to surface removals")

    def test_mixed_line_preserves_amsmath_without_options(self) -> None:
        """Pro #1b regression: \\usepackage[margin=1in]{geometry,amsmath,amssymb}
        previously lost amsmath/amssymb too. They should survive
        without options."""
        tex = (
            "\\documentclass{article}\n"
            "\\usepackage[margin=1in]{geometry,amsmath,amssymb}\n"
            "\\begin{document}X\\end{document}"
        )
        out = ep._ensure_complete_latex(tex)
        self.assertNotIn("geometry", out)
        self.assertNotIn("margin=1in", out)
        self.assertIn("\\usepackage{amsmath,amssymb}", out)

    def test_fallback_strips_solitary_geometry_without_docclass(self) -> None:
        tex = "\\usepackage{geometry}\nSome content."
        out = ep._ensure_complete_latex(tex)
        self.assertNotIn("geometry", out)
        self.assertIn("Some content.", out)

    def test_strips_orphan_geometry_command_after_package_removal(self) -> None:
        tex = (
            "\\documentclass{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\geometry{margin=1in}\n"
            "\\begin{document}X\\end{document}"
        )
        removals: list[str] = []
        out = ep._ensure_complete_latex(tex, removals=removals)

        self.assertNotIn("\\usepackage", out)
        self.assertNotIn("\\geometry", out)
        self.assertTrue(any("geometry command" in item for item in removals))

    def test_inserts_common_package_for_missing_cref_support(self) -> None:
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}See \\cref{main}.\\end{document}"
        )
        out = ep._ensure_complete_latex(tex)

        self.assertIn("\\usepackage{cleveref}", out)
        self.assertLess(out.index("\\usepackage{cleveref}"), out.index("\\begin{document}"))

    def test_fallback_handles_unmatched_begin_document(self) -> None:
        tex = "\\begin{document}\nProof body without preamble.\n"
        out = ep._ensure_complete_latex(tex)
        # Exactly one \begin{document} after wrapping.
        self.assertEqual(out.count("\\begin{document}"), 1)
        # Exactly one \end{document}, supplied by the wrapper.
        self.assertEqual(out.count("\\end{document}"), 1)
        self.assertIn("Proof body without preamble.", out)

    def test_strips_manual_margin_and_spacing_commands(self) -> None:
        tex = (
            "\\documentclass{article}\n"
            "\\linespread{1.2}\n"
            "\\setlength{\\textwidth}{7in}\n"
            "\\begin{document}X\\end{document}"
        )
        out = ep._ensure_complete_latex(tex)
        self.assertNotIn("\\linespread", out)
        self.assertNotIn("\\setlength{\\textwidth}", out)

    def test_strips_in_document_font_size_commands(self) -> None:
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\\small X \\fontsize{9}{10}\\selectfont Y\\end{document}"
        )
        out = ep._ensure_complete_latex(tex)
        self.assertNotIn("\\small", out)
        self.assertNotIn("\\fontsize", out)


class FinalSubmissionCompileTests(unittest.TestCase):
    def _settings(self, output_dir: Path) -> object:
        return ep.Settings(
            input_path=output_dir / "input.json",
            output_dir=output_dir,
            workflow="author_critic_long",
            max_parallel=1,
            page_limit=12,
            budget_usd_per_question=1.0,
            n_rounds=1,
            round_batch_size=1,
            compute_codex_sandbox="docker-bypass",
            runner_script="scripts/run_workflow.py",
            warnings=[],
            deadline_seconds=None,
        )

    def _problem(self, output_dir: Path) -> object:
        return ep.Problem(
            ordinal=1,
            original_id="prob-001",
            safe_id="prob-001",
            text="Problem",
            input_error=None,
            problem_path=output_dir / "prob-001.input.tex",
            log_path=output_dir / "logs" / "prob-001.log",
            output_tex_path=output_dir / "prob-001.tex",
            run_id="firstproof-prob-001",
        )

    @unittest.skipIf(shutil.which("pdflatex") is None, "pdflatex not installed")
    def test_adapter_recompiles_normalized_final_tex_before_ok_status(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                problem = self._problem(output_dir)
                settings = self._settings(output_dir)
                solution_path = (
                    output_dir
                    / "workflow_runs"
                    / problem.run_id
                    / "solutions"
                    / f"{problem.safe_id}.tex"
                )
                solution_path.parent.mkdir(parents=True)
                solution_path.write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{setspace}\n"
                    "\\begin{document}\n"
                    "\\begin{spacing}{2}This compiles before normalization.\\end{spacing}\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )

                async def fake_run_subprocess(_problem, _settings, **_kwargs) -> int:
                    return 0

                with patch.object(ep, "_run_subprocess", fake_run_subprocess):
                    result = await ep._run_problem(
                        problem,
                        settings,
                        asyncio.Semaphore(1),
                    )

                self.assertEqual(result.status, "solution_contract_error")
                self.assertIn("did not compile", result.error or "")
                shipped = problem.output_tex_path.read_text(encoding="utf-8")
                self.assertIn("First Proof fallback solution", shipped)
                self.assertTrue(result.rejected_solution_path)

        asyncio.run(run_check())

    @unittest.skipIf(
        shutil.which("pdflatex") is None or shutil.which("pdfinfo") is None,
        "pdflatex/pdfinfo not installed",
    )
    def test_adapter_checks_final_pdf_page_limit_without_fitz(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                problem = self._problem(output_dir)
                settings = self._settings(output_dir)
                settings = replace(settings, page_limit=1)
                solution_path = (
                    output_dir
                    / "workflow_runs"
                    / problem.run_id
                    / "solutions"
                    / f"{problem.safe_id}.tex"
                )
                solution_path.parent.mkdir(parents=True)
                solution_path.write_text(
                    "\\documentclass{article}\n"
                    "\\begin{document}\n"
                    "Page one.\\newpage Page two.\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )

                async def fake_run_subprocess(_problem, _settings, **_kwargs) -> int:
                    return 0

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.dict(
                    sys.modules, {"fitz": None}
                ):
                    result = await ep._run_problem(
                        problem,
                        settings,
                        asyncio.Semaphore(1),
                    )

                self.assertEqual(result.status, "solution_contract_error")
                self.assertIn("above page_limit=1", result.error or "")

        asyncio.run(run_check())


class RoundBatchTests(unittest.TestCase):
    def _settings(self, output_dir: Path):
        return ep.Settings(
            input_path=output_dir / "input.json",
            output_dir=output_dir,
            workflow="author_critic_long",
            max_parallel=1,
            page_limit=12,
            budget_usd_per_question=1.0,
            n_rounds=10,
            round_batch_size=5,
            compute_codex_sandbox="docker-bypass",
            runner_script="scripts/run_workflow.py",
            warnings=[],
            deadline_seconds=None,
        )

    def _problem(self, output_dir: Path, problem_id: str = "prob-001"):
        return ep.Problem(
            ordinal=int(problem_id.rsplit("-", 1)[-1]),
            original_id=problem_id,
            safe_id=problem_id,
            text="Problem",
            input_error=None,
            problem_path=output_dir / f"{problem_id}.input.tex",
            log_path=output_dir / "logs" / f"{problem_id}.log",
            output_tex_path=output_dir / f"{problem_id}.tex",
            run_id=f"firstproof-{problem_id}",
        )

    def test_round_schedule_batches_to_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings(Path(temp_dir))
            self.assertEqual(ep._round_schedule(settings), [5, 10])
            self.assertEqual(ep._round_schedule(replace(settings, n_rounds=12)), [5, 10, 12])
            self.assertEqual(ep._round_schedule(replace(settings, round_batch_size=20)), [10])

    def test_adaptive_round_schedule_extends_past_initial_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(
                self._settings(Path(temp_dir)),
                adaptive_continuation=True,
                adaptive_max_rounds=20,
            )

            self.assertEqual(ep._round_schedule(settings), [5, 10, 15, 20])
            self.assertTrue(ep._stage_has_followup_rounds(settings, 10))
            self.assertFalse(ep._stage_has_followup_rounds(settings, 20))

    def test_problem_run_id_uses_run_namespace_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(
                self._settings(Path(temp_dir)),
                run_namespace="trial-20260524",
            )
            problems = ep._parse_problems(
                [{"id": "prob-001", "text": "Problem"}],
                settings,
            )

            self.assertEqual(problems[0].run_id, "firstproof-prob-001-trial-20260524")

    def test_unsolved_problem_is_resumed_for_next_round_batch(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)
                calls: list[tuple[int, str | None, int]] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append((n_rounds, restart_from, stage_index))
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"Draft after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": n_rounds == 10,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [(5, None, 1), (10, problem.run_id, 2)])
                self.assertEqual([stage.status for stage in result.stages], ["needs_more_rounds", "solved"])
                self.assertTrue(result.solved)
                self.assertIn("Draft after 10 rounds", result.latex)
                self.assertTrue(
                    (output_dir / "staged_solutions" / "rounds-005" / "prob-001.tex").exists()
                )
                self.assertTrue(
                    (output_dir / "staged_solutions" / "rounds-010" / "prob-001.tex").exists()
                )

        asyncio.run(run_check())

    def test_early_stopped_stage_below_batch_target_is_solved(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)
                calls: list[tuple[int, str | None, int]] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append((n_rounds, restart_from, stage_index))
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        "Early stopped after 1 round.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": 1,
                                    "early_stopped": True,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [(5, None, 1)])
                self.assertEqual([stage.status for stage in result.stages], ["solved"])
                self.assertTrue(result.solved)
                self.assertEqual(result.status, "ok")
                self.assertIn("Early stopped after 1 round.", result.latex)
                self.assertTrue(
                    (output_dir / "staged_solutions" / "rounds-005" / "prob-001.tex").exists()
                )
                self.assertFalse(
                    (output_dir / "staged_solutions" / "rounds-010" / "prob-001.tex").exists()
                )

        asyncio.run(run_check())

    def test_adaptive_unsolved_problem_continues_past_initial_total(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = replace(
                    self._settings(output_dir),
                    adaptive_continuation=True,
                    adaptive_max_rounds=15,
                )
                problem = self._problem(output_dir)
                calls: list[tuple[int, str | None, int]] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append((n_rounds, restart_from, stage_index))
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"Draft after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": n_rounds == 15,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(
                    calls,
                    [(5, None, 1), (10, problem.run_id, 2), (15, problem.run_id, 3)],
                )
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["needs_more_rounds", "needs_more_rounds", "solved"],
                )
                self.assertTrue(result.solved)
                self.assertIn("Draft after 15 rounds", result.latex)

        asyncio.run(run_check())

    def test_adaptive_budget_exhaustion_stops_without_extra_stage(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = replace(
                    self._settings(output_dir),
                    adaptive_continuation=True,
                    adaptive_max_rounds=20,
                )
                problem = self._problem(output_dir)
                calls: list[int] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append(n_rounds)
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"Draft after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    error = (
                        "BudgetExhausted: run budget exhausted"
                        if n_rounds == 10
                        else None
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "error" if error else "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": error is None,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": False,
                                    "error": error,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [5, 10])
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["needs_more_rounds", "budget_exhausted_with_solution"],
                )
                self.assertEqual(result.status, "best_stage_solution")
                self.assertIn("Draft after 5 rounds", result.latex)

        asyncio.run(run_check())

    def test_budget_exhaustion_first_stage_ships_valid_last_gasp_solution(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = replace(self._settings(output_dir), n_rounds=5)
                problem = self._problem(output_dir)
                calls: list[int] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append(n_rounds)
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        "Usable last-gasp draft after budget exhaustion.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "error",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": 1,
                                    "early_stopped": False,
                                    "error": "BudgetExhausted: run budget exhausted",
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertIsNone(restart_from)
                    self.assertEqual(stage_index, 1)
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [5])
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["budget_exhausted_with_solution"],
                )
                self.assertEqual(result.status, "budget_exhausted_with_solution")
                self.assertIn("Usable last-gasp draft", result.latex)
                self.assertNotIn("could not be solved", result.latex)

        asyncio.run(run_check())

    def test_adaptive_finalization_rejects_stale_lower_round_output(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = replace(
                    self._settings(output_dir),
                    adaptive_continuation=True,
                    adaptive_max_rounds=15,
                )
                problem = self._problem(output_dir)
                calls: list[int] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append(n_rounds)
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    text = (
                        f"Good draft after {n_rounds} rounds."
                        if n_rounds < 15
                        else "Stale draft that only completed 10 rounds."
                    )
                    completed = n_rounds if n_rounds < 15 else 10
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"{text}\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": completed,
                                    "early_stopped": False,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [5, 10, 15])
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["needs_more_rounds", "needs_more_rounds", "solution_contract_error"],
                )
                self.assertEqual(result.status, "best_stage_solution")
                self.assertIn("Good draft after 10 rounds", result.latex)
                self.assertNotIn("Stale draft", result.latex)

        asyncio.run(run_check())

    def test_stage_snapshot_promotes_current_answer_before_resume(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)
                snapshots: list[ep.ProblemResult] = []
                top_level_after_first_stage: list[str] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"Snapshot after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": n_rounds == 10,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                async def on_stage(snapshot: ep.ProblemResult) -> None:
                    snapshots.append(snapshot)
                    if len(snapshots) == 1:
                        top_level_after_first_stage.append(
                            problem.output_tex_path.read_text(encoding="utf-8")
                        )

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    await ep._run_problem(
                        problem,
                        settings,
                        asyncio.Semaphore(1),
                        on_stage_complete=on_stage,
                    )

                self.assertEqual([snap.status for snap in snapshots], ["needs_more_rounds", "solved"])
                self.assertTrue(snapshots[0].in_progress)
                self.assertIn("Snapshot after 5 rounds", snapshots[0].latex)
                self.assertIn(
                    "Snapshot after 5 rounds",
                    top_level_after_first_stage[0],
                )

        asyncio.run(run_check())

    def test_final_submission_keeps_best_stage_if_later_stage_regresses(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)
                snapshots: list[ep.ProblemResult] = []
                top_level_after_stage: list[str] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    if n_rounds == 5:
                        body = "Compiling stage-one draft."
                    else:
                        body = "\\begin{spacing}{2}Broken after normalization.\\end{spacing}"
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"{body}\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": n_rounds == 10,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 0

                async def fake_verify(_problem, _settings, tex):
                    if "Broken after normalization" in tex:
                        return False, "pdflatex failed"
                    return True, "compiled with 1 pages"

                async def on_stage(snapshot: ep.ProblemResult) -> None:
                    snapshots.append(snapshot)
                    top_level_after_stage.append(
                        problem.output_tex_path.read_text(encoding="utf-8")
                    )

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(
                        problem,
                        settings,
                        asyncio.Semaphore(1),
                        on_stage_complete=on_stage,
                    )

                shipped = problem.output_tex_path.read_text(encoding="utf-8")
                self.assertEqual(result.status, "best_stage_solution")
                self.assertEqual(
                    [snapshot.status for snapshot in snapshots],
                    ["needs_more_rounds", "solution_contract_error"],
                )
                self.assertIn("Compiling stage-one draft.", snapshots[1].latex)
                self.assertIn("Compiling stage-one draft.", top_level_after_stage[1])
                self.assertIn("Compiling stage-one draft.", result.latex)
                self.assertIn("Compiling stage-one draft.", shipped)
                self.assertNotIn("First Proof fallback solution", shipped)
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["needs_more_rounds", "solution_contract_error"],
                )

        asyncio.run(run_check())

    def test_final_submission_keeps_best_stage_if_later_stage_exits_nonzero(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    body = (
                        "Compiling stage-one draft."
                        if n_rounds == 5
                        else "Late partial from failed stage."
                    )
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"{body}\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": False,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 1 if n_rounds == 10 else 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                shipped = problem.output_tex_path.read_text(encoding="utf-8")
                self.assertEqual(result.status, "best_stage_solution")
                self.assertIn("Compiling stage-one draft.", shipped)
                self.assertNotIn("Late partial from failed stage.", shipped)
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["needs_more_rounds", "workflow_error"],
                )

        asyncio.run(run_check())

    def test_slow_first_stage_does_not_block_other_resume_rounds(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = replace(self._settings(output_dir), max_parallel=2)
                problems = [
                    self._problem(output_dir, "prob-001"),
                    self._problem(output_dir, "prob-002"),
                    self._problem(output_dir, "prob-003"),
                ]
                calls: list[tuple[str, int]] = []
                slow_started = asyncio.Event()
                release_slow = asyncio.Event()
                other_resume_started = asyncio.Event()

                async def fake_run_subprocess(
                    problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append((problem.safe_id, n_rounds))
                    if problem.safe_id == "prob-001" and n_rounds == 5:
                        slow_started.set()
                        await release_slow.wait()
                    else:
                        await asyncio.sleep(0)
                        if problem.safe_id != "prob-001" and n_rounds == 10:
                            other_resume_started.set()
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"{problem.safe_id} after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": True,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": n_rounds == 10,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    semaphore = asyncio.Semaphore(2)
                    tasks = [
                        asyncio.create_task(ep._run_problem(problem, settings, semaphore))
                        for problem in problems
                    ]
                    await slow_started.wait()
                    await asyncio.wait_for(other_resume_started.wait(), timeout=1.0)
                    self.assertFalse(release_slow.is_set())
                    release_slow.set()
                    await asyncio.gather(*tasks)

                self.assertIn(("prob-001", 5), calls)
                self.assertTrue(
                    any(
                        problem_id != "prob-001" and n_rounds == 10
                        for problem_id, n_rounds in calls
                    )
                )

        asyncio.run(run_check())

    def test_contract_error_does_not_count_as_solved_for_batching(self) -> None:
        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                settings = self._settings(output_dir)
                problem = self._problem(output_dir)
                calls: list[int] = []

                async def fake_run_subprocess(
                    _problem,
                    _settings,
                    *,
                    n_rounds,
                    restart_from=None,
                    stage_index=1,
                ) -> int:
                    calls.append(n_rounds)
                    run_dir = output_dir / "workflow_runs" / problem.run_id
                    solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
                    solution.parent.mkdir(parents=True, exist_ok=True)
                    solution.write_text(
                        "\\documentclass{article}\n"
                        "\\begin{document}\n"
                        f"Draft after {n_rounds} rounds.\n"
                        "\\end{document}\n",
                        encoding="utf-8",
                    )
                    (run_dir / "run-metadata.json").write_text(
                        ep.json.dumps(
                            {
                                "status": "ok",
                                "outputs": {
                                    "answer_tex": str(solution),
                                    "compiled": n_rounds == 10,
                                    "pages": 1,
                                    "rounds_completed": n_rounds,
                                    "early_stopped": True,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.assertEqual(restart_from, problem.run_id if stage_index > 1 else None)
                    return 0

                async def fake_verify(_problem, _settings, _tex):
                    return True, "compiled with 1 pages"

                with patch.object(ep, "_run_subprocess", fake_run_subprocess), patch.object(
                    ep, "_verify_exact_latex_for_submission", fake_verify
                ):
                    result = await ep._run_problem(problem, settings, asyncio.Semaphore(1))

                self.assertEqual(calls, [5, 10])
                self.assertEqual(
                    [stage.status for stage in result.stages],
                    ["solution_contract_error", "solved"],
                )
                self.assertTrue(result.solved)
                self.assertEqual(result.status, "ok")

        asyncio.run(run_check())

    def test_workflow_metadata_error_rejects_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = self._settings(output_dir)
            problem = self._problem(output_dir)
            run_dir = output_dir / "workflow_runs" / problem.run_id
            run_dir.mkdir(parents=True)
            (run_dir / "run-metadata.json").write_text(
                ep.json.dumps(
                    {
                        "status": "error",
                        "outputs": {
                            "compiled": True,
                            "pages": 1,
                            "error": "BudgetExhausted: stopped",
                        },
                    }
                ),
                encoding="utf-8",
            )

            reason = ep._workflow_output_rejection(problem, settings)

            self.assertEqual(reason, "workflow metadata reported status=error")

    def test_stage_snapshot_rejects_stale_lower_round_solution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = self._settings(output_dir)
            problem = self._problem(output_dir)
            run_dir = output_dir / "workflow_runs" / problem.run_id
            solution = run_dir / "solutions" / f"{problem.safe_id}.tex"
            solution.parent.mkdir(parents=True)
            solution.write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "Old stage answer.\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (run_dir / "run-metadata.json").write_text(
                ep.json.dumps(
                    {
                        "status": "ok",
                        "outputs": {
                            "answer_tex": str(solution),
                            "compiled": True,
                            "pages": 1,
                            "rounds_completed": 5,
                            "early_stopped": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            reason = ep._workflow_output_rejection(
                problem,
                settings,
                min_rounds_completed=10,
            )
            self.assertIn("below required 10", reason or "")
            self.assertIsNone(
                ep._stage_solution_candidate(
                    problem,
                    settings,
                    n_rounds=10,
                    returncode=0,
                )
            )


class FailOpenFinalizationTests(unittest.TestCase):
    def _settings(self, output_dir: Path):
        return ep.Settings(
            input_path=output_dir / "input.json",
            output_dir=output_dir,
            workflow="author_critic_long",
            max_parallel=1,
            page_limit=50,
            budget_usd_per_question=1.0,
            n_rounds=1,
            round_batch_size=1,
            compute_codex_sandbox="subprocess",
            runner_script="scripts/run_workflow.py",
            warnings=[],
            deadline_seconds=None,
        )

    def _problem(self, output_dir: Path):
        return ep.Problem(
            ordinal=1,
            original_id="prob-001",
            safe_id="prob-001",
            text="Problem",
            input_error=None,
            problem_path=output_dir / "prob-001.txt",
            log_path=output_dir / "logs" / "prob-001.log",
            output_tex_path=output_dir / "prob-001.tex",
            run_id="firstproof-prob-001",
        )

    def test_contract_error_preserves_real_solution_but_ships_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run_dir = output_dir / "workflow_runs" / "firstproof-prob-001"
            solution = run_dir / "solutions" / "prob-001.tex"
            solution.parent.mkdir(parents=True)
            solution.write_text(
                "\\documentclass{article}\\begin{document}Real proof.\\end{document}",
                encoding="utf-8",
            )
            problem = self._problem(output_dir)
            problem.log_path.parent.mkdir(parents=True)

            async def run_check():
                solution_path = ep._find_solution_tex(problem, self._settings(output_dir))
                status = "solution_contract_error"
                latex = ep._fallback_tex(
                    problem,
                    "workflow reported that the final LaTeX did not compile",
                )
                rejected = ep._preserve_rejected_solution(
                    problem,
                    self._settings(output_dir),
                    reason="workflow reported that the final LaTeX did not compile",
                    solution_path=solution_path,
                )
                return latex, status, rejected

            latex, status, rejected = asyncio.run(run_check())

            self.assertEqual(status, "solution_contract_error")
            self.assertIn("First Proof fallback solution", latex)
            self.assertNotIn("Real proof.", latex)
            self.assertIsNotNone(rejected)
            self.assertIn("Real proof.", rejected.read_text(encoding="utf-8"))

    @unittest.skipIf(shutil.which("pdflatex") is None, "pdflatex not installed")
    def test_fail_open_solution_must_still_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run_dir = output_dir / "workflow_runs" / "firstproof-prob-001"
            solution = run_dir / "solutions" / "prob-001.tex"
            solution.parent.mkdir(parents=True)
            solution.write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\begin{spacing}{2}Broken.\\end{spacing}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            problem = self._problem(output_dir)
            problem.log_path.parent.mkdir(parents=True)

            latex, status, error, rejected = asyncio.run(
                ep._ship_solution_or_fallback(
                    problem,
                    self._settings(output_dir),
                    reason="workflow subprocess exited with return code 1",
                    fallback_status="workflow_error",
                    solution_status="workflow_error_with_solution",
                )
            )

            self.assertEqual(status, "workflow_error")
            self.assertIn("First Proof fallback solution", latex)
            self.assertIn("did not compile", error)
            self.assertIsNotNone(rejected)

    def test_salvages_workspace_answer_when_solution_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            answer = (
                output_dir
                / "workflow_runs"
                / "firstproof-prob-001"
                / "ac_workspaces"
                / "prob-001"
                / "answer.tex"
            )
            answer.parent.mkdir(parents=True)
            answer.write_text(
                "\\documentclass{article}\\begin{document}Workspace draft.\\end{document}",
                encoding="utf-8",
            )

            found = ep._find_solution_tex(self._problem(output_dir), self._settings(output_dir))

            self.assertEqual(found, answer.resolve())


class OutputRetrievalFinalizationTests(unittest.TestCase):
    def test_removes_secret_scratch_and_makes_remaining_tree_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run_dir = output_dir / "workflow_runs" / "firstproof-prob-001"
            compute_dir = run_dir / "ac_workspaces" / "prob-001" / "compute"
            cache_file = compute_dir / ".cache" / "pip" / "selfcheck" / "state"
            sage_dir = compute_dir / ".sage"
            problem_file = run_dir / "ac_workspaces" / "prob-001" / "problem.txt"
            auth_file = run_dir / ".compute_codex_home" / "prob-001-r1" / "auth.json"
            env_file = output_dir / "logs" / "requests" / "secret.env"
            top_level_tex = output_dir / "prob-001.tex"

            for path in (cache_file, problem_file, auth_file, env_file, top_level_tex):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
                path.chmod(0o600)
            sage_dir.mkdir(parents=True, exist_ok=True)
            sage_dir.chmod(0o700)

            warnings = ep._finalize_output_permissions(output_dir)

            self.assertEqual(warnings, [])
            self.assertFalse((run_dir / ".compute_codex_home").exists())
            self.assertFalse(env_file.exists())
            self.assertTrue(problem_file.exists())
            self.assertTrue(cache_file.exists())
            self.assertEqual(problem_file.stat().st_mode & 0o777, 0o644)
            self.assertEqual(cache_file.stat().st_mode & 0o777, 0o644)
            self.assertEqual(sage_dir.stat().st_mode & 0o777, 0o755)
            self.assertEqual(top_level_tex.stat().st_mode & 0o777, 0o644)

    def test_unlinks_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            outside = output_dir.parent / "outside-secret-target"
            outside.write_text("keep", encoding="utf-8")
            link = output_dir / "workflow_runs" / "run" / "notes-link"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(outside)

            warnings = ep._finalize_output_permissions(output_dir)

            self.assertEqual(warnings, [])
            self.assertFalse(link.exists())
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")

    def test_failed_secret_removal_is_not_chmodded_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            secret_dir = output_dir / "workflow_runs" / "run" / ".compute_codex_home"
            secret_file = secret_dir / "auth.json"
            public_file = output_dir / "prob-001.tex"
            secret_file.parent.mkdir(parents=True)
            secret_file.write_text("secret", encoding="utf-8")
            public_file.write_text("solution", encoding="utf-8")
            secret_dir.chmod(0o700)
            secret_file.chmod(0o600)
            public_file.chmod(0o600)

            with patch.object(ep.shutil, "rmtree", side_effect=OSError("still mounted")):
                warnings = ep._finalize_output_permissions(output_dir)

            self.assertTrue(any("RETRIEVAL_UNSAFE" in warning for warning in warnings))
            self.assertTrue(secret_dir.exists())
            self.assertTrue(secret_file.exists())
            self.assertEqual(secret_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(secret_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(public_file.stat().st_mode & 0o777, 0o644)

    def test_late_discovered_secret_file_is_removed_before_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            locked_dir = output_dir / "workflow_runs" / "run" / "locked"
            secret_file = locked_dir / "secret.env"
            public_file = output_dir / "prob-001.tex"
            secret_file.parent.mkdir(parents=True)
            secret_file.write_text("secret", encoding="utf-8")
            public_file.write_text("solution", encoding="utf-8")
            locked_dir.chmod(0o000)
            public_file.chmod(0o600)

            real_walk = ep.os.walk
            walk_calls = 0

            def fake_walk(path, *args, **kwargs):
                nonlocal walk_calls
                if Path(path) != output_dir:
                    yield from real_walk(path, *args, **kwargs)
                    return
                walk_calls += 1
                if walk_calls == 1:
                    yield (str(output_dir), ["workflow_runs"], ["prob-001.tex"])
                    yield (str(output_dir / "workflow_runs"), ["run"], [])
                    yield (str(output_dir / "workflow_runs" / "run"), ["locked"], [])
                    return
                yield (str(output_dir), ["workflow_runs"], ["prob-001.tex"])
                yield (str(output_dir / "workflow_runs"), ["run"], [])
                yield (str(output_dir / "workflow_runs" / "run"), ["locked"], [])
                yield (str(locked_dir), [], ["secret.env"])

            try:
                with patch.object(ep.os, "walk", side_effect=fake_walk):
                    warnings = ep._finalize_output_permissions(output_dir)
                locked_mode = locked_dir.stat().st_mode & 0o777
                public_mode = public_file.stat().st_mode & 0o777
            finally:
                locked_dir.chmod(0o700)

            self.assertEqual(warnings, [])
            self.assertFalse(secret_file.exists())
            self.assertEqual(locked_mode, 0o755)
            self.assertEqual(public_mode, 0o644)


if __name__ == "__main__":
    unittest.main()
