"""Prune must never delete outside the run: symlinked node dirs and
symlinked artifacts are refused, and deletion targets must resolve under
agents/. Sandboxes referenced by their node's output stay untouched."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import estimate_prunable_bytes, prune_run_artifacts  # noqa: E402


def _node(run_dir: Path, name: str, *, output: dict | None = None) -> Path:
    node = run_dir / "agents" / name
    (node / "sandbox").mkdir(parents=True)
    (node / "sandbox" / "junk.bin").write_bytes(b"x" * 64)
    (node / "output.json").write_text(
        json.dumps(output if output is not None else {"ok": True}), encoding="utf-8"
    )
    return node


class PruneSymlinkTests(unittest.TestCase):
    def test_symlinked_node_dir_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as ext:
            run_dir = Path(tmp) / "run"
            (run_dir / "agents").mkdir(parents=True)
            external = Path(ext) / "external-node"
            (external / "sandbox").mkdir(parents=True)
            (external / "sandbox" / "precious.txt").write_text("keep", encoding="utf-8")
            (external / "output.json").write_text("{}", encoding="utf-8")
            (run_dir / "agents" / "evil").symlink_to(external)

            result = prune_run_artifacts(run_dir)

            self.assertEqual(result["pruned_nodes"], 0)
            self.assertTrue((external / "sandbox" / "precious.txt").exists())
            self.assertEqual(estimate_prunable_bytes(run_dir), 0)

    def test_symlinked_agents_root_is_refused(self) -> None:
        # A symlinked top-level agents/ would bless its external target as
        # the containment root; refuse the whole prune instead.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as ext:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            external = Path(ext) / "agents"
            node = external / "node"
            (node / "sandbox").mkdir(parents=True)
            (node / "sandbox" / "precious.txt").write_text("keep", encoding="utf-8")
            (node / "output.json").write_text("{}", encoding="utf-8")
            (run_dir / "agents").symlink_to(external)

            result = prune_run_artifacts(run_dir)

            self.assertEqual(result["pruned_nodes"], 0)
            self.assertTrue((node / "sandbox" / "precious.txt").exists())
            self.assertEqual(estimate_prunable_bytes(run_dir), 0)

    def test_symlinked_sandbox_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as ext:
            run_dir = Path(tmp) / "run"
            node = run_dir / "agents" / "node"
            node.mkdir(parents=True)
            (node / "output.json").write_text("{}", encoding="utf-8")
            external = Path(ext) / "target"
            external.mkdir()
            (external / "precious.txt").write_text("keep", encoding="utf-8")
            (node / "sandbox").symlink_to(external)

            prune_run_artifacts(run_dir)

            self.assertTrue((external / "precious.txt").exists())
            self.assertTrue((node / "sandbox").is_symlink())

    def test_plain_unreferenced_sandbox_still_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            node = _node(run_dir, "node")

            result = prune_run_artifacts(run_dir)

            self.assertEqual(result["pruned_nodes"], 1)
            self.assertFalse((node / "sandbox").exists())

    def test_output_referenced_sandbox_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            node_path = run_dir / "agents" / "node"
            node = _node(
                run_dir, "node", output={"workspace": str(node_path / "sandbox")}
            )
            prune_run_artifacts(run_dir)
            self.assertTrue((node / "sandbox").exists())


if __name__ == "__main__":
    unittest.main()
