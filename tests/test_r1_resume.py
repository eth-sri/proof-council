"""Regression tests for the resume state-loss cluster (R-A6) and the resume
env allow-list (B1).

R-A6: a human answer submitted while a run is stopped must survive resume.
The inbox filename is keyed on the resume-stable cache key (not the random
per-call workdir), and a pre-existing response is consumed, not unlinked.

B1: the dashboard resume route re-injects only allow-listed env from
resume.json, so a hand-edited resume.json cannot inject arbitrary environment.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.agents.human_agent import HumanAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HUMAN_CFG = {
    "human_solver": {
        "prompt": "Problem: {problem}",
        "output_schema": {"answer_tex": "string", "status": "string"},
        "human_timeout_s": 2.0,
    }
}


def _response_filename_from_inbox(inbox: Path) -> str:
    task_files = list(inbox.glob("*.task.json"))
    assert task_files, "human node never surfaced a task"
    payload = json.loads(task_files[-1].read_text(encoding="utf-8"))
    return Path(payload["response_path"]).name


class HumanResumeConsumesAnswerTests(unittest.TestCase):
    def test_stem_stable_across_resume_and_answer_is_consumed(self) -> None:
        # run1 reaches the human node and BLOCKS, then is STOPPED (cancelled,
        # like a SIGTERM) so nothing is cached. The human submits an answer to
        # the durable inbox file. On resume the node must poll the SAME filename
        # and replay that answer instead of re-asking.
        with tempfile.TemporaryDirectory() as tmp:
            outputs_root = Path(tmp)

            async def block_then_stop() -> str:
                ctx = RunContext.create(
                    run_id="run1",
                    root_workdir=outputs_root,
                    component_configs=_HUMAN_CFG,
                )
                task = asyncio.ensure_future(
                    HumanAgent(ctx, name="human_solver")(problem="P")
                )
                inbox = ctx.root_workdir / "human_inbox"
                for _ in range(400):
                    if inbox.exists() and list(inbox.glob("*.task.json")):
                        break
                    await asyncio.sleep(0.005)
                resp = _response_filename_from_inbox(inbox)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                # A stopped (not timed-out) node must leave nothing cached.
                cache_dir = ctx.root_workdir / "resume_cache"
                self.assertFalse(
                    cache_dir.exists() and list(cache_dir.glob("*.json"))
                )
                return resp

            with mock.patch.object(HumanAgent, "POLL_INTERVAL_S", 0.01):
                resp1 = asyncio.run(block_then_stop())

            inbox = outputs_root / "run1" / "human_inbox"
            # The human submits while the run is stopped.
            (inbox / resp1).write_text(
                json.dumps({"answer_tex": "42", "status": "done"}),
                encoding="utf-8",
            )

            async def resume_and_pick_up() -> object:
                ctx = RunContext.create(
                    run_id="run1",
                    root_workdir=outputs_root,
                    resume_from="run1",
                    component_configs=_HUMAN_CFG,
                )
                return await asyncio.wait_for(
                    HumanAgent(ctx, name="human_solver")(problem="P"), timeout=2.0
                )

            with mock.patch.object(HumanAgent, "POLL_INTERVAL_S", 0.01):
                out = asyncio.run(resume_and_pick_up())

            resp2 = _response_filename_from_inbox(inbox)

        # Stem is resume-stable: the second run polls the same file it was
        # answered on, so the durable answer is consumed rather than orphaned.
        self.assertEqual(resp1, resp2)
        self.assertEqual(out.answer_tex, "42")
        self.assertEqual(out.status, "done")

    def test_preexisting_response_is_not_unlinked(self) -> None:
        # An answer already sitting in the inbox before run() starts must be
        # read on the first poll, not deleted.
        with tempfile.TemporaryDirectory() as tmp:
            outputs_root = Path(tmp)

            async def probe_stem() -> str:
                ctx = RunContext.create(
                    run_id="r", root_workdir=outputs_root, component_configs=_HUMAN_CFG
                )
                agent = HumanAgent(ctx, name="human_solver")
                # cache_key is deterministic for the same node + inputs, so the
                # inbox stem can be computed ahead of the run.
                inp = agent.Inputs(problem="P")
                return f"{agent.name}__{agent._cache_key(inp)[:16]}.response.json"

            stem = asyncio.run(probe_stem())
            inbox = outputs_root / "r" / "human_inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / stem).write_text(
                json.dumps({"answer_tex": "pre", "status": "done"}), encoding="utf-8"
            )

            async def run_agent() -> object:
                ctx = RunContext.create(
                    run_id="r", root_workdir=outputs_root, component_configs=_HUMAN_CFG
                )
                return await asyncio.wait_for(
                    HumanAgent(ctx, name="human_solver")(problem="P"), timeout=2.0
                )

            with mock.patch.object(HumanAgent, "POLL_INTERVAL_S", 0.01):
                out = asyncio.run(run_agent())

            self.assertTrue((inbox / stem).exists())
        self.assertEqual(out.answer_tex, "pre")
        self.assertEqual(out.status, "done")


class DashboardResumeEnvAllowlistTests(unittest.TestCase):
    def test_resume_route_ignores_non_allowlisted_env(self) -> None:
        # B1: a hand-edited resume.json must not inject arbitrary environment on
        # resume — only keys on the writer's allowlist survive (now empty, so
        # none do).
        dev = _load_module("app_dev_b1", "app/dev.py")

        captured: dict = {}

        class FakePopen:
            def __init__(self, cmd, cwd=None, env=None, **kw):
                captured["env"] = env

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "myrun"
            run_dir.mkdir()
            (run_dir / "run-metadata.json").write_text(
                json.dumps({"status": "stopped", "preset": "author_critic"}),
                encoding="utf-8",
            )
            (run_dir / "resume.json").write_text(
                json.dumps(
                    {
                        "run_id": "myrun",
                        "argv": [
                            "scripts/run_workflow.py",
                            "--workflow",
                            "author_critic",
                            "--run-id",
                            "myrun",
                            "--output",
                            str(root),
                        ],
                        "env": {
                            "PROOFCOUNCIL_PACING": "off",
                            "EVIL_INJECTED": "1",
                            "LD_PRELOAD": "/tmp/evil.so",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(dev.subprocess, "Popen", FakePopen), mock.patch.dict(
                "os.environ", {}, clear=False
            ):
                import os

                os.environ.pop("PROOFCOUNCIL_PACING", None)
                os.environ.pop("LD_PRELOAD", None)
                app = dev.create_app(runs_roots=(root,))
                resp = app.test_client().post("/run/myrun/resume")

        self.assertEqual(resp.status_code, 302)
        # empty allowlist: even a formerly-allowlisted key no longer survives
        self.assertIsNone(captured["env"].get("PROOFCOUNCIL_PACING"))
        self.assertIsNone(captured["env"].get("EVIL_INJECTED"))
        self.assertIsNone(captured["env"].get("LD_PRELOAD"))


if __name__ == "__main__":
    unittest.main()
