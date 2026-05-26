"""Parallel solve / verify / improve orchestration node."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict

from proofstack.agent import Agent
from proofstack.agents.configurable_prompt import ConfigurablePromptAgent
from proofstack.context import ModelSpec


DEFAULT_MODEL: ModelSpec = "models/openai/gpt-54-mini"
DEFAULT_TOOL_REFS = ["web_search_preview"]
DEFAULT_BRANCH_COUNT = 3
DEFAULT_PASS_COUNT = 3

SOLVER_SYSTEM = "You are an expert research mathematician. Return a complete proof inside <solution>...</solution>."
SOLVER_USER = """Problem:
{problem}

Literature search:
{literature_search}

This is branch {branch_index} of {n}. Try a proof strategy that is meaningfully independent from the other branches.

Return the proof inside <solution>...</solution>."""

VERIFIER_SYSTEM = """You are a strict mathematical verifier.
Return detailed feedback inside <verification>...</verification>.
Return the verdict inside <verdict>...</verdict>, using exactly correct or incorrect."""
VERIFIER_USER = """Problem:
{problem}

Literature search:
{literature_search}

Candidate proof:
{solution}"""

IMPROVER_SYSTEM = "You improve mathematical proofs according to verifier feedback. Return only the improved proof inside <solution>...</solution>."
IMPROVER_USER = """Problem:
{problem}

Literature search:
{literature_search}

Current proof:
{solution}

Verification feedback:
{verification}

Return the improved proof inside <solution>...</solution>."""

MERGER_SYSTEM = "You merge candidate mathematical proofs into one rigorous standalone proof. Return only the final proof inside <solution>...</solution>."
MERGER_USER = """Problem:
{problem}

Literature search:
{literature_search}

Candidate proofs:
{candidate_proofs}

Merge the strongest correct material into one complete proof inside <solution>...</solution>."""


class ParallelSolveVerifyImprove(Agent):
    """Run n independent solve/verify/improve branches, each for m passes, then merge."""

    description = "Run n parallel solve/verify/improve branches and merge them into one proof."
    PALETTE = {
        "id": "parallel_svi",
        "label": "Parallel Solve / Verify / Improve",
        "group": "Proof Work",
        "description": "Runs n proof branches, improves each for m passes, then merges one proof.",
        "keywords": "parallel solve verify improve merge n m custom python",
    }
    HIDDEN_GRAPH_INPUTS = {"n", "m"}

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        problem: str = ""
        literature_search: str = ""

    class Outputs(BaseModel):
        solution: str = ""

    @classmethod
    def default_component_config(cls) -> dict[str, Any]:
        return {
            "model": DEFAULT_MODEL,
            "tool_refs": list(DEFAULT_TOOL_REFS),
            "n": DEFAULT_BRANCH_COUNT,
            "m": DEFAULT_PASS_COUNT,
            "solver_system_prompt": SOLVER_SYSTEM,
            "solver_user_prompt": SOLVER_USER,
            "verifier_system_prompt": VERIFIER_SYSTEM,
            "verifier_user_prompt": VERIFIER_USER,
            "improver_system_prompt": IMPROVER_SYSTEM,
            "improver_user_prompt": IMPROVER_USER,
            "merger_system_prompt": MERGER_SYSTEM,
            "merger_user_prompt": MERGER_USER,
        }

    @classmethod
    def component_config_editor(cls) -> dict[str, Any]:
        prompt_fields = []
        for role, label in (
            ("solver", "Solver prompt"),
            ("verifier", "Verifier prompt"),
            ("improver", "Improver prompt"),
            ("merger", "Merger prompt"),
        ):
            prompt_fields.extend(
                [
                    {
                        "key": f"{role}_system_prompt",
                        "label": "System prompt",
                        "type": "textarea",
                        "rows": 5,
                        "section": label,
                    },
                    {
                        "key": f"{role}_user_prompt",
                        "label": "User prompt",
                        "type": "textarea",
                        "rows": 7,
                        "section": label,
                    },
                ]
            )
        return {
            "title": "Parallel solve / verify / improve",
            "fields": [
                {"key": "model", "label": "Model", "type": "model"},
                {"key": "n", "label": "Parallel branches", "type": "integer", "min": 1},
                {"key": "m", "label": "Verify / improve passes", "type": "integer", "min": 0},
                {"key": "tool_refs", "label": "Tools", "type": "tools"},
                *prompt_fields,
            ],
        }

    async def run(self, inp: Inputs) -> Outputs:  # type: ignore[override]
        agents = self._agents()
        branch_count = self._int_config("n", DEFAULT_BRANCH_COUNT, minimum=1)
        pass_count = self._int_config("m", DEFAULT_PASS_COUNT, minimum=0)
        branches = await asyncio.gather(
            *[
                self._run_branch(
                    index=i + 1,
                    n=branch_count,
                    m=pass_count,
                    problem=inp.problem,
                    literature_search=inp.literature_search,
                    agents=agents,
                )
                for i in range(branch_count)
            ]
        )
        self._write_branch_log(branches)
        solutions = [b["solution"] for b in branches if str(b.get("solution") or "").strip()]
        if not solutions:
            return self.Outputs(solution="")
        if len(solutions) == 1:
            return self.Outputs(solution=solutions[0])
        merged = await agents["merger"](
            problem=inp.problem,
            literature_search=inp.literature_search,
            candidate_proofs=_candidate_proofs_text(solutions),
        )
        return self.Outputs(solution=str(getattr(merged, "solution", "") or ""))

    async def _run_branch(
        self,
        *,
        index: int,
        n: int,
        m: int,
        problem: str,
        literature_search: str,
        agents: dict[str, ConfigurablePromptAgent],
    ) -> dict[str, Any]:
        draft = await agents["solver"](
            problem=problem,
            literature_search=literature_search,
            branch_index=index,
            n=n,
        )
        solution = str(getattr(draft, "solution", "") or "")
        records: list[dict[str, str]] = []
        verdict = "unverified"
        for iteration in range(m):
            checked = await agents["verifier"](
                problem=problem,
                literature_search=literature_search,
                solution=solution,
                branch_index=index,
                iteration=iteration + 1,
                m=m,
            )
            verification = str(getattr(checked, "verification", "") or "")
            verdict = _normalize_verdict(str(getattr(checked, "verdict", "") or ""))
            records.append(
                {
                    "iteration": str(iteration + 1),
                    "verdict": verdict,
                    "verification": verification,
                }
            )
            if verdict == "correct":
                break
            improved = await agents["improver"](
                problem=problem,
                literature_search=literature_search,
                solution=solution,
                verification=verification,
                branch_index=index,
                iteration=iteration + 1,
                m=m,
            )
            replacement = str(getattr(improved, "solution", "") or "")
            if replacement.strip():
                solution = replacement
        return {
            "branch": index,
            "draft_solution": str(getattr(draft, "solution", "") or ""),
            "solution": solution,
            "verdict": verdict,
            "passes": records,
        }

    def _agents(self) -> dict[str, ConfigurablePromptAgent]:
        names = {
            "solver": f"{self.name}.solver",
            "verifier": f"{self.name}.verifier",
            "improver": f"{self.name}.improver",
            "merger": f"{self.name}.merger",
        }
        subconfigs = {names[role]: self._role_config(role) for role in names}
        ctx = replace(self.ctx, component_configs={**self.ctx.component_configs, **subconfigs})
        parent_scope = f"agent:{self.name}"
        return {
            role: ConfigurablePromptAgent(ctx, name=name, parent_budget_scope=parent_scope)
            for role, name in names.items()
        }

    def _role_config(self, role: str) -> dict[str, Any]:
        defaults = {
            "solver": (SOLVER_SYSTEM, SOLVER_USER, ["solution"], "solution"),
            "verifier": (VERIFIER_SYSTEM, VERIFIER_USER, ["verification", "verdict"], "verification"),
            "improver": (IMPROVER_SYSTEM, IMPROVER_USER, ["solution"], "solution"),
            "merger": (MERGER_SYSTEM, MERGER_USER, ["solution"], "solution"),
        }
        system_prompt, user_prompt, xml_tags, default_field = defaults[role]
        cfg = self.component_config
        out: dict[str, Any] = {
            "model": cfg.get(f"{role}_model", cfg.get("model", DEFAULT_MODEL)),
            "system_prompt": cfg.get(f"{role}_system_prompt", system_prompt),
            "user_prompt": cfg.get(f"{role}_user_prompt", user_prompt),
            "input_schema": _role_input_schema(role),
            "output": {"xml_tags": xml_tags, "default_field": default_field},
            "tool_refs": cfg.get(f"{role}_tool_refs", cfg.get("tool_refs", DEFAULT_TOOL_REFS)),
        }
        if f"{role}_max_tool_calls" in cfg or "max_tool_calls" in cfg:
            out["max_tool_calls"] = cfg.get(f"{role}_max_tool_calls", cfg.get("max_tool_calls"))
        return out

    def _write_branch_log(self, branches: list[dict[str, Any]]) -> None:
        try:
            (self.workdir / "branches.json").write_text(
                json.dumps(branches, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _int_config(self, key: str, default: int, *, minimum: int) -> int:
        try:
            value = int(self.component_config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)


def _role_input_schema(role: str) -> dict[str, str]:
    common = {
        "problem": "string",
        "literature_search": "string",
        "branch_index": "integer",
        "n": "integer",
    }
    if role == "solver":
        return common
    if role == "merger":
        return {
            "problem": "string",
            "literature_search": "string",
            "candidate_proofs": "string",
        }
    return {
        **common,
        "solution": "string",
        "verification": "string",
        "iteration": "integer",
        "m": "integer",
    }


def _normalize_verdict(raw: str) -> str:
    tokens = re.findall(r"[a-z]+", raw.lower())
    if "incorrect" in tokens:
        return "incorrect"
    if "correct" in tokens:
        return "correct"
    return "unknown"


def _candidate_proofs_text(solutions: list[str]) -> str:
    return "\n\n---\n\n".join(
        f"Candidate proof {i}:\n{solution}"
        for i, solution in enumerate(solutions, start=1)
    )


__all__ = ["ParallelSolveVerifyImprove"]
