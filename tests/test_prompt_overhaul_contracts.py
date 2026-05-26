from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.author import (  # noqa: E402
    AUTHOR_LOOP_SYSTEM,
    AUTHOR_LOOP_USER,
    AUTHOR_LOOP_USER_CONTAINER,
    AUTHOR_LOOP_SYSTEM_CONTAINER,
    AUTHOR_ROUND0_SYSTEM,
    AUTHOR_ROUND0_USER,
    AUTHOR_ROUND0_USER_CONTAINER,
    AUTHOR_ROUND0_SYSTEM_CONTAINER,
    Author,
)
from proofstack.agents.ac.critic import (  # noqa: E402
    CRITIC_PROMPT_HEAD,
    CRITIC_STATEFUL_USER,
    ACCritic,
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


class PromptOverhaulContractTests(unittest.TestCase):
    def test_ac_author_prompts_have_research_ambition_and_interpretation_rule(self) -> None:
        prompts = [
            AUTHOR_ROUND0_SYSTEM,
            AUTHOR_LOOP_SYSTEM,
            AUTHOR_ROUND0_SYSTEM_CONTAINER,
            AUTHOR_LOOP_SYSTEM_CONTAINER,
        ]

        for prompt in prompts:
            flat = _squash(prompt)
            self.assertIn("novel, creative, and non-trivial elements", flat)
            self.assertIn("attempt to prove the lemma, run the computation", flat)
            self.assertIn("Problem statement and interpretation", flat)
            self.assertIn("do not silently solve a different problem", flat)

    def test_ac_author_prompts_use_functional_role_wording(self) -> None:
        prompts = [
            AUTHOR_ROUND0_SYSTEM,
            AUTHOR_LOOP_SYSTEM,
            AUTHOR_ROUND0_SYSTEM_CONTAINER,
            AUTHOR_LOOP_SYSTEM_CONTAINER,
        ]

        for prompt in prompts:
            self.assertIn("Act as a research-level mathematical proof author", prompt)
            self.assertNotIn("You are a research mathematician", prompt)

        for prompt in (AUTHOR_ROUND0_SYSTEM, AUTHOR_ROUND0_SYSTEM_CONTAINER):
            self.assertIn("\n\nResearch ambition and problem interpretation.", prompt)
            self.assertNotIn(".\nResearch ambition and problem interpretation.", prompt)

    def test_ac_author_readiness_requires_complete_solution(self) -> None:
        for prompt in (AUTHOR_LOOP_SYSTEM, AUTHOR_LOOP_SYSTEM_CONTAINER):
            flat = _squash(prompt)
            self.assertIn("complete rigorous solution", flat)
            self.assertIn("no remaining open gaps", flat)
            self.assertIn("no unproved essential lemmas", flat)
            self.assertIn("Do not declare ``<ready>true</ready>`` merely because", flat)
            self.assertNotIn("or — if no more turns remain", prompt)
            self.assertNotIn("or - if no more turns remain", prompt)

    def test_ac_critic_rejects_partial_or_open_issue_answers(self) -> None:
        flat = _squash(CRITIC_PROMPT_HEAD + "\n" + CRITIC_STATEFUL_USER)

        self.assertIn("fully solves the stated problem as a complete rigorous solution", flat)
        self.assertIn("Act as a strict mathematical referee", flat)
        self.assertIn("unproved essential lemmas", flat)
        self.assertIn("Remaining open issues", flat)
        self.assertIn("Problem statement and interpretation", flat)
        self.assertIn("partial final answer that merely lists open issues is not answer-ready", flat)
        self.assertIn("`<answer_ready>false</answer_ready>`", flat)
        self.assertNotIn("You are a research mathematician", flat)

    def test_ac_prompts_surface_firstproof_latex_contract(self) -> None:
        author_inputs = Author.Inputs(
            problem="Prove X.",
            round=1,
            n_rounds=2,
            page_limit=12,
            prev_critique="Fix the proof.",
            workflow_feedback="No LaTeX compile or formatting issues detected.",
        )
        author = object.__new__(Author)
        author_user = author.render_messages(author_inputs)[1]["content"]
        self.assertIn("First Proof LaTeX contract", author_user)
        self.assertIn("\\documentclass[12pt]{article}", author_user)
        self.assertIn("at most 12 pages", author_user)
        self.assertIn("fullpage", author_user)
        self.assertIn("font size", author_user)
        self.assertIn("Workflow compile/format feedback", author_user)

        critic_inputs = ACCritic.Inputs(
            problem="Prove X.",
            round=1,
            n_rounds=2,
            page_limit=12,
            answer_tex="\\documentclass[12pt]{article}\\begin{document}X\\end{document}",
        )
        critic = object.__new__(ACCritic)
        critic_user = critic.render_messages(critic_inputs)[0]["content"]
        self.assertIn("First Proof LaTeX contract", critic_user)
        self.assertIn("wrong document class", critic_user)
        self.assertIn("line-spacing changes", critic_user)
        self.assertIn("font-size", critic_user)

    def test_pwc_prompts_have_research_ambition_and_interpretation_rule(self) -> None:
        round_cfg = yaml.safe_load((ROOT / "configs/workflows/pwc_round.yaml").read_text())
        workflow_cfg = yaml.safe_load((ROOT / "configs/workflows/pwc_workflow.yaml").read_text())

        planner = round_cfg["components"]["cfg_pwc_planner"]["system_prompt"]
        worker = round_cfg["components"]["cfg_pwc_worker"]["prompt"]
        round_critic = round_cfg["components"]["cfg_pwc_critic"]["system_prompt"]
        final_critic = workflow_cfg["components"]["cfg_pwc_critic"]["system_prompt"]

        self.assertIn("Act as the Planner", planner)
        self.assertIn("Act as a research-level mathematical proof worker", worker)
        self.assertIn("Act as a strict mathematical referee", round_critic)
        self.assertIn("Act as a strict final mathematical referee", final_critic)

        for prompt in (planner, worker):
            flat = _squash(prompt)
            self.assertIn("novel, creative, and non-trivial elements", flat)
            self.assertIn("non-trivial lemma, computation, or reduction", flat)
            self.assertIn("If rounds remain", flat)

        for prompt in (planner, worker, round_critic, final_critic):
            flat = _squash(prompt)
            self.assertIn("Problem statement and interpretation", flat)
            self.assertIn("complete rigorous solution", flat)

        for prompt in (planner, round_critic, final_critic):
            flat = _squash(prompt)
            self.assertIn("answer_ready=true only if", flat)
            self.assertIn("no unproved essential lemmas", flat)
            self.assertIn("Remaining open issues", flat)

        combined = "\n".join((planner, worker, round_critic, final_critic))
        self.assertNotIn("You are the Planner", combined)
        self.assertNotIn("You are the Worker", combined)
        self.assertNotIn("You are the Critic", combined)
        self.assertNotIn("You are the final Critic", combined)


if __name__ == "__main__":
    unittest.main()
