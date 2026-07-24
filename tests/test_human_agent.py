from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.human_agent import HumanAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class HumanAgentTests(unittest.TestCase):
    def test_response_without_status_defaults_to_done(self) -> None:
        async def run_agent_and_answer(ctx: RunContext) -> object:
            task = asyncio.create_task(HumanAgent(ctx, name="human_solver")(problem="P"))
            inbox = ctx.root_workdir / "human_inbox"
            response_path: Path | None = None
            for _ in range(200):
                task_files = list(inbox.glob("*.task.json")) if inbox.exists() else []
                if task_files:
                    payload = json.loads(task_files[0].read_text(encoding="utf-8"))
                    response_path = Path(payload["response_path"])
                    break
                await asyncio.sleep(0.005)
            self.assertIsNotNone(response_path)
            assert response_path is not None
            response_path.write_text(json.dumps({"answer_tex": "A"}), encoding="utf-8")
            return await asyncio.wait_for(task, timeout=2.0)

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=tmp,
                flat=True,
                component_configs={
                    "human_solver": {
                        "prompt": "Problem: {problem}",
                        "output_schema": {"answer_tex": "string", "status": "string"},
                        "human_timeout_s": 2.0,
                    }
                },
            )

            with mock.patch.object(HumanAgent, "POLL_INTERVAL_S", 0.01):
                out = asyncio.run(run_agent_and_answer(ctx))

        self.assertEqual(out.answer_tex, "A")
        self.assertEqual(out.status, "done")

    def test_cache_disabled_uses_a_fresh_response_file_each_call(self) -> None:
        async def answer_next(
            agent: HumanAgent,
            inbox: Path,
            seen_tasks: set[Path],
            answer: str,
        ) -> tuple[Path, object]:
            call = asyncio.create_task(agent(problem="P"))
            task_path: Path | None = None
            for _ in range(400):
                current = set(inbox.glob("*.task.json")) if inbox.exists() else set()
                fresh = current - seen_tasks
                if fresh:
                    task_path = fresh.pop()
                    seen_tasks.add(task_path)
                    break
                await asyncio.sleep(0.005)
            self.assertIsNotNone(task_path)
            assert task_path is not None
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            response_path = Path(payload["response_path"])
            response_path.write_text(json.dumps({"answer_tex": answer}), encoding="utf-8")
            return response_path, await asyncio.wait_for(call, timeout=2.0)

        async def run_twice(ctx: RunContext) -> tuple[Path, Path, object, object]:
            agent = HumanAgent(ctx, name="human_solver")
            inbox = ctx.root_workdir / "human_inbox"
            seen_tasks: set[Path] = set()
            first_path, first = await answer_next(agent, inbox, seen_tasks, "first")
            second_path, second = await answer_next(agent, inbox, seen_tasks, "second")
            return first_path, second_path, first, second

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=tmp,
                flat=True,
                component_configs={
                    "human_solver": {
                        "prompt": "Problem: {problem}",
                        "output_schema": {"answer_tex": "string", "status": "string"},
                        "cache_enabled": False,
                        "human_timeout_s": 2.0,
                    }
                },
            )
            with mock.patch.object(HumanAgent, "POLL_INTERVAL_S", 0.01):
                first_path, second_path, first, second = asyncio.run(run_twice(ctx))

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first.answer_tex, "first")
        self.assertEqual(second.answer_tex, "second")


if __name__ == "__main__":
    unittest.main()
