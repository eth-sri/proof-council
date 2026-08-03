"""Self-reported failures (status error/timeout/partial/blocked, or a
non-empty error field) must not enter the resume cache: replaying a failed
round on resume silently turns one bad round into a permanent one. The
salvage output still flows to the current run's consumers."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agent import Agent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class Toy(Agent):
    class Inputs(BaseModel):
        x: int

    class Outputs(BaseModel):
        status: str = ""
        error: str | None = None
        value: int = 0

    result: Outputs

    async def run(self, inp: BaseModel) -> BaseModel:
        return self.result


class UncacheableOutputTests(unittest.TestCase):
    def _run(self, out: Toy.Outputs) -> tuple[Toy, object]:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="t", root_workdir=Path(tmp) / "run", flat=True
            )
            agent = Toy(ctx)
            agent.result = out
            result = asyncio.run(agent(x=1))
            key = agent._cache_key(Toy.Inputs(x=1))
            return result, ctx.resume_cache.get(key)

    def test_ok_output_is_cached(self) -> None:
        result, cached = self._run(Toy.Outputs(status="done", value=7))
        self.assertEqual(result.value, 7)
        self.assertIsNotNone(cached)

    def test_failure_statuses_are_not_cached(self) -> None:
        for status in ("error", "timeout", "partial", "blocked", "ERROR"):
            result, cached = self._run(Toy.Outputs(status=status, value=1))
            self.assertEqual(result.status, status)  # caller still sees it
            self.assertIsNone(cached, status)

    def test_error_field_is_not_cached(self) -> None:
        _, cached = self._run(Toy.Outputs(status="done", error="boom"))
        self.assertIsNone(cached)


class ExcludedKeyFieldTests(unittest.TestCase):
    def test_excluded_input_fields_do_not_change_the_key(self) -> None:
        class Metered(Toy):
            CACHE_KEY_EXCLUDE_INPUTS = frozenset({"budget"})

            class Inputs(BaseModel):
                x: int
                budget: float = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="t", root_workdir=Path(tmp) / "run", flat=True
            )
            agent = Metered(ctx)
            key_a = agent._cache_key(Metered.Inputs(x=1, budget=99.51))
            key_b = agent._cache_key(Metered.Inputs(x=1, budget=131.90))
            key_c = agent._cache_key(Metered.Inputs(x=2, budget=99.51))
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_ac_author_excludes_budget_meter(self) -> None:
        from proofstack.agents.ac.author import Author

        self.assertEqual(
            Author.CACHE_KEY_EXCLUDE_INPUTS,
            frozenset({"budget_used_usd", "budget_max_usd"}),
        )


if __name__ == "__main__":
    unittest.main()
