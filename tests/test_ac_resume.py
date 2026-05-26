from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.agents.ac.ac_workflow import (  # noqa: E402
    ACWorkflow,
    _CompileResult,
    _problem_hash,
)
from proofstack.agents.ac.author import Author  # noqa: E402
from proofstack.agents.ac.council import CouncilReply  # noqa: E402
from proofstack.agents.ac.critic import ACCritic  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from scripts import run_workflow  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _workspace(root: Path, problem_id: str = "p", problem: str = "P") -> Path:
    return root / "ac_workspaces" / f"{problem_id}-{_problem_hash(problem)}"


class ACResumeTests(unittest.TestCase):
    def test_default_terminal_author_round_gets_in_loop_critic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            calls: list[tuple] = []

            async def fake_author_run(self, inp):
                calls.append(("author", inp.round))
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                calls.append(("critic", inp.round, inp.mode))
                return self.Outputs(
                    review_md=f"review r{inp.round}",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        enable_final_critic=False,
                    )
                )

            self.assertIn(("critic", 1, "fresh"), calls)
            self.assertFalse(out.last_critic_accepted)
            self.assertEqual(out.final_critic_mode_run, "not_run")

    def test_terminal_auxiliary_requests_are_not_hidden_ship_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            calls = {"council": 0, "compute": 0}

            async def fake_author_run(self, inp):
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=inp.round == 1,
                    council_question="check this" if inp.round == 1 else None,
                    compute_instructions="search for counterexamples" if inp.round == 1 else None,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md=f"review r{inp.round}",
                    answer_ready=True,
                    mode=inp.mode,
                    messages_after=[{"role": "assistant", "content": "ok"}],
                )

            async def fake_council_run(self, inp):
                calls["council"] += 1
                return self.Outputs(replies=[])

            async def fake_compute_run(self, inp):
                calls["compute"] += 1
                return self.Outputs(status="done", summary="counterexample")

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)
            workflow.council.run = types.MethodType(fake_council_run, workflow.council)
            workflow.compute.run = types.MethodType(fake_compute_run, workflow.compute)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=True,
                        enable_compute=True,
                        enable_final_critic=False,
                    )
                )

            self.assertEqual(calls, {"council": 0, "compute": 0})
            self.assertFalse(out.early_stopped)
            self.assertTrue(out.last_critic_accepted)
            workspace = _workspace(ctx.root_workdir)
            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertIn("terminal Author turn requested", state["pending_critique"])

    def test_staged_boundary_auxiliary_requests_are_not_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            calls = {"council": 0, "compute": 0}

            async def fake_author_run(self, inp):
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=inp.round == 1,
                    council_question="check this" if inp.round == 1 else None,
                    compute_instructions="search for counterexamples" if inp.round == 1 else None,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md=f"review r{inp.round}",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[{"role": "assistant", "content": "ok"}],
                )

            async def fake_council_run(self, inp):
                calls["council"] += 1
                return self.Outputs(
                    replies=[
                        CouncilReply(
                            member="Member",
                            model_ref="models/test",
                            text="council advice",
                        )
                    ]
                )

            async def fake_compute_run(self, inp):
                calls["compute"] += 1
                return self.Outputs(status="done", response_md="compute advice")

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)
            workflow.council.run = types.MethodType(fake_council_run, workflow.council)
            workflow.compute.run = types.MethodType(fake_compute_run, workflow.compute)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        stop_after_review_round=True,
                        enable_council=True,
                        enable_compute=True,
                        enable_final_critic=False,
                    )
                )

            self.assertEqual(calls, {"council": 1, "compute": 1})
            state = json.loads(
                (_workspace(ctx.root_workdir) / ".ac" / "resume-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("council advice", state["pending_council_text"])
            self.assertIn("compute advice", state["pending_compute_text"])
            self.assertNotIn("terminal Author turn requested", state["pending_critique"])

    def test_resume_terminal_auxiliary_requests_are_not_hidden_ship_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            awaiting = Author.Outputs(
                answer_tex="draft",
                research_notes_tex="notes",
                references_bib="",
                ready=True,
                council_question="check this",
                compute_instructions="search for counterexamples",
                thinking_summary="author just finished",
            )
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(
                    problem="P",
                    problem_id="p",
                    n_rounds=1,
                    enable_council=True,
                    enable_compute=True,
                ),
                last_round_run=0,
                next_round=1,
                review_history=[],
                critic_conversation=[{"role": "user", "content": "old"}],
                critic_instance_turn=1,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=False,
                awaiting_review_round=1,
                awaiting_review_kind="round_review",
                awaiting_author=awaiting,
            )
            calls = {"council": 0, "compute": 0}

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md="same-round review",
                    answer_ready=True,
                    mode=inp.mode,
                    messages_after=[
                        *inp.prior_messages,
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            async def fake_council_run(self, inp):
                calls["council"] += 1
                return self.Outputs(replies=[])

            async def fake_compute_run(self, inp):
                calls["compute"] += 1
                return self.Outputs(status="done", response_md="hidden")

            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)
            workflow.council.run = types.MethodType(fake_council_run, workflow.council)
            workflow.compute.run = types.MethodType(fake_compute_run, workflow.compute)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="draft",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=True,
                        enable_compute=True,
                        enable_final_critic=False,
                        resume_run=True,
                    )
                )

            self.assertEqual(calls, {"council": 0, "compute": 0})
            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertIn("terminal Author turn requested", state["pending_critique"])
            self.assertEqual(state["pending_council_text"], "")
            self.assertEqual(state["pending_compute_text"], "")

    def test_workspace_path_includes_problem_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            workflow = ACWorkflow(ctx)

            first = workflow._workspace_path("p", "Problem A")
            second = workflow._workspace_path("p", "Problem B")

            self.assertNotEqual(first, second)
            self.assertEqual(first.name, f"p-{_problem_hash('Problem A')}")

    def test_stop_after_review_round_reviews_boundary_author(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            calls: list[tuple] = []

            async def fake_author_run(self, inp):
                calls.append(("author", inp.round, inp.prev_critique))
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                calls.append(("critic", inp.round, inp.mode))
                return self.Outputs(
                    review_md=f"review r{inp.round}",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        stop_after_review_round=True,
                        enable_council=False,
                        enable_compute=False,
                    )
                )

            self.assertIn(("critic", 1, "stateful"), calls)
            self.assertEqual(out.rounds_completed, 1)
            state = json.loads(
                (_workspace(ctx.root_workdir) / ".ac" / "resume-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state.get("next_round"), 2)
            self.assertIsNone(state.get("awaiting_review_round"))
            self.assertEqual(
                state.get("review_history", [])[-1].get("review_md"),
                "review r1",
            )

    def test_legacy_final_author_resume_starts_with_missing_fresh_critic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="old-run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = ctx.root_workdir / "ac_workspaces" / "p"
            snap = workspace / ".ac" / "round-1"
            snap.mkdir(parents=True)
            for name, body in (
                ("answer.tex", "old answer"),
                ("research_notes.tex", "old notes"),
                ("references.bib", ""),
            ):
                (workspace / name).write_text(body, encoding="utf-8")
                (snap / name).write_text(body, encoding="utf-8")
            (workspace / "problem.txt").write_text("Original problem", encoding="utf-8")
            _write_json(
                snap / "author_outputs.json",
                Author.Outputs(
                    answer_tex="old answer",
                    research_notes_tex="old notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="final author notes",
                ).model_dump(mode="json"),
            )
            calls: list[tuple] = []

            async def fake_critic_run(self, inp):
                calls.append(("critic", inp.round, inp.mode, inp.omit_author_thinking))
                return self.Outputs(
                    review_md="missing final review",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            async def fake_author_run(self, inp):
                calls.append(("author", inp.round, inp.prev_critique))
                return self.Outputs(
                    answer_tex="new answer",
                    research_notes_tex="new notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="continued",
                )

            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)
            workflow.author.run = types.MethodType(fake_author_run, workflow.author)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="new answer",
                    tex_path=None,
                    pdf_path=None,
                    compiled=False,
                    pages=0,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="Original problem",
                        problem_id="p",
                        n_rounds=2,
                        enable_council=False,
                        enable_compute=False,
                        resume_run=True,
                    )
                )

            self.assertEqual(calls[0], ("critic", 1, "fresh", True))
            self.assertEqual(calls[1][0:2], ("author", 2))
            self.assertIn("missing final review", calls[1][2])
            self.assertEqual(out.rounds_completed, 2)

    def test_completed_default_run_checkpoint_noops_without_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)

            async def fake_author_run(self, inp):
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md="review",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        enable_final_critic=False,
                    )
                )
            self.assertEqual(out.rounds_completed, 1)

            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(state.get("awaiting_review_round"))
            self.assertIsNone(state.get("awaiting_review_kind"))
            self.assertEqual(state.get("next_round"), 2)
            self.assertEqual(state.get("terminal_outputs", {}).get("pages"), 1)

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run on restart of finished run")

            async def fail_critic_run(self, inp):
                raise AssertionError("Critic should not run on restart of finished run")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fail_critic_run, workflow.critic)
            with patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                side_effect=AssertionError("compile should not run on no-op restart"),
            ):
                out2 = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        enable_final_critic=False,
                        resume_run=True,
                    )
                )

            self.assertEqual(out2.rounds_completed, 1)
            self.assertTrue(out2.compiled)
            self.assertEqual(out2.pages, 1)
            self.assertTrue(out2.answer_tex.exists())

    def test_early_stop_cancelled_before_stash_resumes_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)

            async def fake_author_run(self, inp):
                return self.Outputs(
                    answer_tex=f"answer r{inp.round}",
                    research_notes_tex="notes",
                    references_bib="",
                    ready=inp.round == 1,
                    thinking_summary="t",
                )

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md="ready",
                    answer_ready=True,
                    mode=inp.mode,
                    messages_after=[
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            async def deterministic_ready(self, workspace, *, page_limit):
                return True, []

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)
            workflow._deterministic_ready = types.MethodType(
                deterministic_ready, workflow
            )

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ), patch.object(
                workflow, "_stash_answer", side_effect=asyncio.CancelledError()
            ):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(
                        workflow(
                            problem="P",
                            problem_id="p",
                            n_rounds=3,
                            enable_council=False,
                            enable_compute=False,
                            enable_final_critic=False,
                        )
                    )

            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertTrue(state.get("awaiting_finalization"))
            self.assertIsNone(state.get("awaiting_review_round"))

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run while resuming finalization")

            async def fail_critic_run(self, inp):
                raise AssertionError("Critic should not run while resuming finalization")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fail_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="answer r1",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=3,
                        enable_council=False,
                        enable_compute=False,
                        enable_final_critic=False,
                        resume_run=True,
                    )
                )

            self.assertTrue(out.answer_tex.exists())
            self.assertTrue(out.compiled)
            self.assertEqual(out.pages, 1)
            self.assertEqual(out.rounds_completed, 1)
            self.assertTrue(out.early_stopped)
            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertFalse(state.get("awaiting_finalization"))
            self.assertTrue(state.get("terminal_outputs", {}).get("compiled"))

    def test_checkpoint_after_author_resumes_at_same_round_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            awaiting = Author.Outputs(
                answer_tex="draft",
                research_notes_tex="notes",
                references_bib="",
                ready=False,
                thinking_summary="author just finished",
            )
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="P", problem_id="p", n_rounds=1),
                last_round_run=0,
                next_round=1,
                review_history=[],
                critic_conversation=[{"role": "user", "content": "old"}],
                critic_instance_turn=1,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=False,
                awaiting_review_round=1,
                awaiting_review_kind="round_review",
                awaiting_author=awaiting,
            )
            calls: list[tuple] = []

            async def fake_critic_run(self, inp):
                calls.append(("critic", inp.round, inp.mode, inp.omit_author_thinking))
                return self.Outputs(
                    review_md="same-round review",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[
                        *inp.prior_messages,
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                )

            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="draft",
                    tex_path=None,
                    pdf_path=None,
                    compiled=False,
                    pages=0,
                ),
            ):
                asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        resume_run=True,
                    )
                )

            self.assertEqual(calls, [("critic", 1, "fresh", False)])

    def test_resume_noops_when_round_bound_already_reached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            (workspace / "answer.tex").write_text("done", encoding="utf-8")
            (workspace / "research_notes.tex").write_text("notes", encoding="utf-8")
            (workspace / "references.bib").write_text("", encoding="utf-8")
            answer_path = ctx.root_workdir / "solutions" / "p.tex"
            answer_path.parent.mkdir()
            answer_path.write_text("done", encoding="utf-8")
            _write_json(
                ctx.root_workdir / "run-metadata.json",
                {
                    "outputs": {
                        "compiled": True,
                        "pages": 1,
                        "rounds_completed": 1,
                        "early_stopped": False,
                    }
                },
            )
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="P", problem_id="p", n_rounds=1),
                last_round_run=1,
                next_round=2,
                review_history=[],
                critic_conversation=[],
                critic_instance_turn=0,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=False,
            )

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run on no-op resume")

            async def fail_critic_run(self, inp):
                raise AssertionError("Critic should not run on no-op resume")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fail_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                side_effect=AssertionError("compile should not run on no-op resume"),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        resume_run=True,
                    )
                )

            self.assertEqual(out.answer_tex, answer_path)
            self.assertTrue(out.compiled)
            self.assertEqual(out.pages, 1)
            self.assertEqual(out.rounds_completed, 1)

    def test_plain_resume_of_early_stopped_run_noops_without_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            (workspace / "answer.tex").write_text("done", encoding="utf-8")
            (workspace / "research_notes.tex").write_text("notes", encoding="utf-8")
            (workspace / "references.bib").write_text("", encoding="utf-8")
            answer_path = ctx.root_workdir / "solutions" / "p.tex"
            answer_path.parent.mkdir()
            answer_path.write_text("done", encoding="utf-8")
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="P", problem_id="p", n_rounds=5),
                last_round_run=2,
                next_round=3,
                review_history=[],
                critic_conversation=[],
                critic_instance_turn=0,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=True,
                terminal_outputs={
                    "answer_tex": "solutions/p.tex",
                    "compiled": True,
                    "pages": 1,
                    "rounds_completed": 2,
                    "early_stopped": True,
                },
            )

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run after early-stop resume")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)

            out = asyncio.run(
                workflow(
                    problem="P",
                    problem_id="p",
                    n_rounds=5,
                    enable_council=False,
                    enable_compute=False,
                    resume_run=True,
                )
            )

            self.assertEqual(out.rounds_completed, 2)
            self.assertTrue(out.early_stopped)
            self.assertEqual(out.answer_tex, answer_path)
            self.assertTrue(out.compiled)

    def test_stale_early_stop_checkpoint_without_stash_finalizes_before_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            (workspace / "answer.tex").write_text("done", encoding="utf-8")
            (workspace / "research_notes.tex").write_text("notes", encoding="utf-8")
            (workspace / "references.bib").write_text("", encoding="utf-8")
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="P", problem_id="p", n_rounds=5),
                last_round_run=2,
                next_round=3,
                review_history=[],
                critic_conversation=[],
                critic_instance_turn=0,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=True,
            )

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run while finalizing stale early-stop")

            async def fail_critic_run(self, inp):
                raise AssertionError("Critic should not run while finalizing stale early-stop")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fail_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="done",
                    tex_path=None,
                    pdf_path=None,
                    compiled=True,
                    pages=1,
                ),
            ):
                out = asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=5,
                        enable_council=False,
                        enable_compute=False,
                        resume_run=True,
                    )
                )

            self.assertTrue(out.answer_tex.exists())
            self.assertTrue(out.compiled)
            self.assertEqual(out.pages, 1)
            self.assertEqual(out.rounds_completed, 2)
            self.assertTrue(out.early_stopped)
            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertFalse(state.get("awaiting_finalization"))
            self.assertTrue(state.get("terminal_outputs", {}).get("compiled"))

    def test_resume_extension_replays_pending_context_and_budget_offset_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = _workspace(ctx.root_workdir)
            workspace.mkdir(parents=True)
            (workspace / "answer.tex").write_text("draft", encoding="utf-8")
            (workspace / "research_notes.tex").write_text("notes", encoding="utf-8")
            (workspace / "references.bib").write_text("", encoding="utf-8")
            zip_path = workspace / ".ac" / "round-0" / "compute_workspace_round_0.zip"
            zip_path.parent.mkdir(parents=True)
            zip_path.write_bytes(b"zip")
            (ctx.root_workdir / "events.jsonl").write_text(
                json.dumps({"kind": "model.call", "payload": {"cost_usd": 2.75}})
                + "\n",
                encoding="utf-8",
            )
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="P", problem_id="p", n_rounds=2),
                last_round_run=0,
                next_round=1,
                review_history=[
                    ACCritic.Outputs(review_md="previous review", answer_ready=False)
                ],
                critic_conversation=[],
                critic_instance_turn=0,
                pending_council_text="council reply",
                pending_compute_text="compute reply",
                pending_compute_zip_path=zip_path,
                pending_critique="",
                early_stopped=False,
            )
            seen: dict[str, object] = {}

            async def fake_author_run(self, inp):
                seen["budget_used_usd"] = inp.budget_used_usd
                seen["prev_critique"] = inp.prev_critique
                seen["prev_council"] = inp.prev_council
                seen["prev_compute_response"] = inp.prev_compute_response
                seen["compute_zip_path"] = inp.compute_zip_path
                return self.Outputs(
                    answer_tex="continued",
                    research_notes_tex="continued notes",
                    references_bib="",
                    ready=False,
                    thinking_summary="continued",
                )

            async def fake_critic_run(self, inp):
                return self.Outputs(
                    review_md="review",
                    answer_ready=False,
                    mode=inp.mode,
                    messages_after=[],
                )

            workflow.author.run = types.MethodType(fake_author_run, workflow.author)
            workflow.critic.run = types.MethodType(fake_critic_run, workflow.critic)

            with patch(
                "proofstack.agents.ac.ac_workflow.asyncio.to_thread",
                side_effect=_immediate_to_thread,
            ), patch(
                "proofstack.agents.ac.ac_workflow._simple_compile_latex",
                return_value=_CompileResult(
                    tex="continued",
                    tex_path=None,
                    pdf_path=None,
                    compiled=False,
                    pages=0,
                ),
            ):
                asyncio.run(
                    workflow(
                        problem="P",
                        problem_id="p",
                        n_rounds=1,
                        enable_council=False,
                        enable_compute=False,
                        resume_run=True,
                    )
                )

            self.assertAlmostEqual(float(seen["budget_used_usd"]), 2.75)
            self.assertEqual(seen["prev_critique"], "previous review")
            self.assertEqual(seen["prev_council"], "council reply")
            self.assertEqual(seen["prev_compute_response"], "compute reply")
            self.assertEqual(seen["compute_zip_path"], zip_path)
            state = json.loads(
                (workspace / ".ac" / "resume-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["pending_council_text"], "")
            self.assertEqual(state["pending_compute_text"], "")
            self.assertNotIn(
                "messages_after",
                state["review_history"][0],
                "checkpoint review history should not duplicate Critic transcripts",
            )

    def test_run_workflow_restart_copy_does_not_enable_resume_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = root / "old-run"
            workspace = old / "ac_workspaces" / "p"
            workspace.mkdir(parents=True)
            (workspace / "problem.txt").write_text("P", encoding="utf-8")
            _write_json(
                old / "run-metadata.json",
                {"config_snapshot": {"problem_id": "p"}},
            )
            seen: dict[str, object] = {}

            async def fake_ac_run(self, inp):
                seen["resume_from"] = self.ctx.resume_from
                seen["root_workdir"] = self.ctx.root_workdir
                seen["resume_run"] = inp.resume_run
                out_path = self.ctx.root_workdir / "solutions" / "p.tex"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("answer", encoding="utf-8")
                return self.Outputs(
                    problem_id=inp.problem_id,
                    answer_tex=out_path,
                    research_notes_tex=(
                        self.ctx.root_workdir
                        / "ac_workspaces"
                        / "p"
                        / "research_notes.tex"
                    ),
                    references_bib=(
                        self.ctx.root_workdir
                        / "ac_workspaces"
                        / "p"
                        / "references.bib"
                    ),
                    rounds_completed=0,
                )

            argv = [
                "run_workflow.py",
                "--workflow",
                "author_critic",
                "--output",
                str(root),
                "--restart-from",
                "old-run",
                "--restart-copy",
                "--run-id",
                "new-run",
                "--input",
                "n_rounds=1",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                ACWorkflow, "run", fake_ac_run
            ), patch(
                "proofstack.events.JSONLSink.write",
                side_effect=_noop_event_write,
            ), patch(
                "sys.stdout",
                new=io.StringIO(),
            ):
                rc = asyncio.run(run_workflow.amain())

            self.assertEqual(rc, 0)
            self.assertIsNone(seen["resume_from"])
            self.assertEqual(seen["root_workdir"], root / "new-run")
            self.assertTrue(seen["resume_run"])
            self.assertTrue((root / "old-run" / "ac_workspaces" / "p").exists())
            self.assertTrue((root / "new-run" / "ac_workspaces" / "p").exists())

    def test_resume_preserves_original_problem_and_records_effective_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx = RunContext.create(run_id="run", root_workdir=root, flat=True)
            _disable_event_writes(ctx)
            workflow = ACWorkflow(ctx)
            workspace = ctx.root_workdir / "ac_workspaces" / "p"
            workspace.mkdir(parents=True)
            (workspace / "problem.txt").write_text(
                "Original problem statement", encoding="utf-8"
            )
            (workspace / "answer.tex").write_text("done", encoding="utf-8")
            answer_path = ctx.root_workdir / "solutions" / "p.tex"
            answer_path.parent.mkdir()
            answer_path.write_text("done", encoding="utf-8")
            _write_json(
                ctx.root_workdir / "run-metadata.json",
                {"outputs": {"compiled": True, "pages": 1, "rounds_completed": 1}},
            )
            # Bound already reached -> no-op resume (no model calls), but
            # the effective-problem artifact must still be written.
            workflow._save_resume_state(
                workspace,
                inp=ACWorkflow.Inputs(problem="x", problem_id="p", n_rounds=1),
                last_round_run=1,
                next_round=2,
                review_history=[],
                critic_conversation=[],
                critic_instance_turn=0,
                pending_council_text="",
                pending_compute_text="",
                pending_compute_zip_path=None,
                pending_critique="",
                early_stopped=False,
            )

            async def fail_author_run(self, inp):
                raise AssertionError("Author should not run on no-op resume")

            workflow.author.run = types.MethodType(fail_author_run, workflow.author)

            asyncio.run(
                workflow(
                    problem="Original problem statement",
                    problem_id="p",
                    n_rounds=1,
                    enable_council=False,
                    enable_compute=False,
                    resume_run=True,
                    additional_instructions="focus on the prime case",
                )
            )

            self.assertEqual(
                (workspace / "problem.txt").read_text(encoding="utf-8"),
                "Original problem statement",
            )
            effective = (workspace / "problem-effective.txt").read_text(encoding="utf-8")
            self.assertIn("Original problem statement", effective)
            self.assertIn("resumed from an earlier workflow", effective)
            self.assertIn("focus on the prime case", effective)


def _disable_event_writes(ctx: RunContext) -> None:
    async def no_write(record: dict) -> None:
        return None

    ctx.events.sink.write = no_write  # type: ignore[method-assign]


async def _immediate_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


async def _noop_event_write(*args, **kwargs) -> None:
    return None


if __name__ == "__main__":
    unittest.main()
