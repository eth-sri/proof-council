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


if __name__ == "__main__":
    unittest.main()
