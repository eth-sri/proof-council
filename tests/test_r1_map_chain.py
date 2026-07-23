"""R1-A1: map_chain reuses ONE agent instance across concurrent items, so a
ConfigurableCLIAgent's per-invocation state (CLI_CMD, workspace, copied-auth)
must not live on ``self`` — one item would clobber another mid-run.

DAGWorkflow._agent_for caches a single instance per ``node.step`` key, and that
key is identical for every map item, so all concurrent items share the object.
These tests pin that the per-call state is isolated regardless (it now rides
per-call ContextVars, like ``workdir``).
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.agents.dag_workflow import DAGWorkflow  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class _ProbeCLIAgent(ConfigurableCLIAgent):
    """Stub that mirrors the real run() lifecycle without a sandbox/CLI: set the
    per-call state from THIS item's input, yield at a barrier so every concurrent
    item is mid-run at once, then read the state back (as base run()/cli_input
    would at spawn time) and record what THIS item actually saw."""

    barrier: asyncio.Barrier | None = None
    seen: dict[int, dict] = {}

    async def run(self, inp):  # type: ignore[override]
        idx = int(getattr(inp, "index"))
        # write (as run() does: self.CLI_CMD = self._command_for(inp), etc.)
        self.CLI_CMD = ["codex", "--task", str(idx)]
        self._active_workspace_root = Path(f"/ws/item-{idx}")
        self._copied_codex_auth = (idx % 2 == 0)
        # ...await setup()+stream_command()... — every item parks here together,
        # so a shared plain attribute would be last-writer-wins by now.
        assert type(self).barrier is not None
        await type(self).barrier.wait()
        # read back, as the framework does after the await window
        type(self).seen[idx] = {
            "cmd": list(self.CLI_CMD),
            "ws": str(self._active_workspace_root),
            "auth": bool(self._copied_codex_auth),
        }
        return {"index": idx}


def _map_node() -> dict:
    return {
        "id": "mapnode",
        "kind": "map_chain",
        "foreach": ["a", "b", "c"],
        "max_parallel": 3,
        "steps": [
            {
                "id": "solve",
                "agent": "proofstack.agents.configurable_cli.ConfigurableCLIAgent",
                "inputs": {"index": "$index"},
            }
        ],
        "collect": {"index": "$step.solve.index"},
    }


class MapChainInstanceReuseTests(unittest.TestCase):
    def test_concurrent_items_do_not_share_per_call_cli_state(self) -> None:
        node = _map_node()
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="a1",
                root_workdir=Path(tmp),
                component_configs={"DAGWorkflow": {"dag": {"nodes": [node], "outputs": {}}}},
            )
            wf = DAGWorkflow(ctx, name="wf")
            # One shared instance for every item == the bug condition _agent_for
            # produces (identical node.step cache key). We assert it's still safe.
            _ProbeCLIAgent.barrier = asyncio.Barrier(3)
            _ProbeCLIAgent.seen = {}
            probe = _ProbeCLIAgent(ctx, name="mapnode.solve")

            with patch.object(DAGWorkflow, "_agent_for", return_value=probe):
                asyncio.run(wf._run_map_chain(node, {"node": {}}))

        seen = _ProbeCLIAgent.seen
        self.assertEqual(set(seen), {0, 1, 2})
        # Each item must have observed ITS OWN command / workspace / auth flag.
        for idx in (0, 1, 2):
            self.assertEqual(
                seen[idx]["cmd"], ["codex", "--task", str(idx)],
                f"item {idx} launched another item's command: {seen[idx]['cmd']}",
            )
            self.assertEqual(seen[idx]["ws"], f"/ws/item-{idx}")
            self.assertEqual(seen[idx]["auth"], idx % 2 == 0)

    def test_shared_instance_identity_is_the_reuse_condition(self) -> None:
        # Guard the premise: _agent_for really does hand every map item the same
        # object (so the isolation above matters).
        with tempfile.TemporaryDirectory() as tmp:
            node = _map_node()
            ctx = RunContext.create(
                run_id="a1id",
                root_workdir=Path(tmp),
                component_configs={"DAGWorkflow": {"dag": {"nodes": [node], "outputs": {}}}},
            )
            wf = DAGWorkflow(ctx, name="wf")
            step = node["steps"][0]
            a0 = wf._agent_for(step, default_name="mapnode.solve")
            a1 = wf._agent_for(step, default_name="mapnode.solve")
            self.assertIs(a0, a1)


if __name__ == "__main__":
    unittest.main()
