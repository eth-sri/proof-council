from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import (  # noqa: E402
    _rename_component_output_refs,
    mutate_preset_yaml,
    validate_preset_yaml,
)


def _component(raw_yaml: str, name: str) -> dict[str, Any]:
    return yaml.safe_load(raw_yaml)["components"][name]


# The UI output_schema-editing feature (schema-only projection/reconciliation)
# was dropped: no real editor affordance posts a bare output_schema, and the
# projection helper was a recurring source of silent corruption. These tests pin
# the replacement contract — a bare schema change is refused, the CLI file editor
# (which bundles output_files) still works — plus the two residual fixes that
# outlived the feature: B8 (output-path validation) and A5 (scoped rename).


API_STRUCTURED_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      judge:
        model: models/openai/gpt-54
        prompt: judge {problem}
        input_schema:
          problem: string
        output_schema:
          decision: object
          items: array
          note: string
        output:
          json_tags:
            decision: decision
          xml_lists:
            items: item
          xml_tags: [note]
          default_field: note
    dag:
      nodes:
        - id: j
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: judge
          inputs:
            problem: $input.problem
      outputs:
        decision: $node.j.decision
        items: $node.j.items
    """
)


CLI_FIXTURE_TEMPLATE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      solver:
        cmd: [claude, -p]
        prompt: solve {problem}
        input_schema:
          problem: string
        output_schema:
          draft: string
        output_files:
          draft:
            path: PATH_HERE
            type: text
    dag:
      nodes:
        - id: solve
          kind: agent
          agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
          name: solver
          inputs:
            problem: $input.problem
      outputs:
        draft: $node.solve.draft
    """
)

CLI_FIXTURE = CLI_FIXTURE_TEMPLATE.replace("PATH_HERE", "notes.txt")


class OutputSchemaEditRefusedTests(unittest.TestCase):
    def test_bare_business_schema_change_is_refused(self) -> None:
        result = mutate_preset_yaml(
            API_STRUCTURED_FIXTURE,
            {
                "op": "update_component",
                "name": "judge",
                "fields": {"output_schema": {"decision": "object", "items": "array"}},
            },
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("not supported in the editor" in e for e in result["errors"]),
            result["errors"],
        )
        # refusal is atomic: the document is left untouched
        self.assertEqual(
            _component(result["raw_yaml"], "judge")["output_schema"],
            {"decision": "object", "items": "array", "note": "string"},
        )

    def test_unchanged_schema_is_allowed(self) -> None:
        # re-submitting the same business schema is a no-op, not a refusal
        result = mutate_preset_yaml(
            API_STRUCTURED_FIXTURE,
            {
                "op": "update_component",
                "name": "judge",
                "fields": {
                    "output_schema": {
                        "decision": "object",
                        "items": "array",
                        "note": "string",
                    }
                },
            },
        )
        self.assertTrue(result["ok"], result["errors"])

    def test_cli_file_edit_bundling_output_files_is_allowed(self) -> None:
        # the CLI file editor always posts output_files alongside output_schema;
        # that legitimate flow must pass through, not hit the refusal.
        result = mutate_preset_yaml(
            CLI_FIXTURE,
            {
                "op": "update_component",
                "name": "solver",
                "fields": {
                    "output_schema": {"draft": "string", "notes": "string"},
                    "output_files": {
                        "draft": {"path": "notes.txt", "type": "text"},
                        "notes": "notes_2.txt",
                    },
                },
            },
        )
        self.assertTrue(result["ok"], result["errors"])
        cfg = _component(result["raw_yaml"], "solver")
        self.assertIn("notes", cfg["output_files"])
        self.assertIn("notes", cfg["output_schema"])


class OutputPathValidationB8Tests(unittest.TestCase):
    def test_absolute_and_escaping_paths_rejected(self) -> None:
        for bad in ("/tmp/notes.txt", "../notes.txt", "../../etc/passwd"):
            raw = CLI_FIXTURE_TEMPLATE.replace("PATH_HERE", bad)
            report = validate_preset_yaml(raw)
            self.assertFalse(report["ok"], bad)
            self.assertTrue(
                any("relative path inside the workspace" in e for e in report["errors"]),
                (bad, report["errors"]),
            )

    def test_relative_path_has_no_path_error(self) -> None:
        report = validate_preset_yaml(CLI_FIXTURE)
        self.assertEqual(
            [e for e in report["errors"] if "workspace" in e],
            [],
        )


class ScopedRenameA5Tests(unittest.TestCase):
    def test_rename_does_not_corrupt_sibling_repeat_scope(self) -> None:
        # Two repeat bodies each own a node id `worker`; renaming component a's
        # output must not touch component b's independent scope.
        raw = {
            "dag": {
                "nodes": [
                    {
                        "id": "loopA",
                        "body": {
                            "nodes": [
                                {"id": "worker", "name": "a"},
                                {"id": "cA", "prompt": "$node.worker.old"},
                            ]
                        },
                    },
                    {
                        "id": "loopB",
                        "body": {
                            "nodes": [
                                {"id": "worker", "name": "b"},
                                {"id": "cB", "prompt": "$node.worker.old"},
                            ]
                        },
                    },
                ]
            }
        }
        _rename_component_output_refs(raw, "a", {"old": "new"})
        self.assertEqual(
            raw["dag"]["nodes"][0]["body"]["nodes"][1]["prompt"], "$node.worker.new"
        )
        self.assertEqual(
            raw["dag"]["nodes"][1]["body"]["nodes"][1]["prompt"], "$node.worker.old"
        )

    def test_rename_updates_same_scope_outputs_and_parent_ref(self) -> None:
        # The legit case still works: a top-level rename updates this scope's
        # outputs AND the $parent.node form reaching down into a repeat body.
        raw = {
            "dag": {
                "nodes": [
                    {"id": "lit", "name": "a"},
                    {
                        "id": "loop",
                        "body": {
                            "nodes": [
                                {
                                    "id": "verify",
                                    "name": "b",
                                    "inputs": {"x": "$parent.node.lit.old"},
                                }
                            ]
                        },
                    },
                ],
                "outputs": {"final": "$node.lit.old"},
            }
        }
        _rename_component_output_refs(raw, "a", {"old": "new"})
        self.assertEqual(raw["dag"]["outputs"]["final"], "$node.lit.new")
        self.assertEqual(
            raw["dag"]["nodes"][1]["body"]["nodes"][0]["inputs"]["x"],
            "$parent.node.lit.new",
        )


# --- P-B3: editor JS keeps type/default when a CLI output path is renamed -----
# (The CLI file editor is retained; only the bare-schema projection was dropped.)

EDITOR_HTML = ROOT / "app" / "templates" / "dev_preset_editor.html"


def _extract_js_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    brace = source.index("{", start)
    depth = 0
    for j in range(brace, len(source)):
        ch = source[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
    raise AssertionError(f"unterminated function {name!r}")


@unittest.skipUnless(shutil.which("node"), "node not available for JS harness")
class EditorRenameB3Tests(unittest.TestCase):
    def _run_node(self) -> dict[str, Any]:
        html = EDITOR_HTML.read_text(encoding="utf-8")
        schema_fn = _extract_js_function(html, "cliOutputSchemaForRow")
        spec_fn = _extract_js_function(html, "cliOutputSpecForRow")
        script = (
            schema_fn
            + "\n"
            + spec_fn
            + "\n"
            + textwrap.dedent(
                """
                // Row saved as an artifact-path spec, then its path is renamed.
                const row = { dataset: {
                    originalOutputName: 'artifact',
                    originalOutputPath: 'artifact.bin',
                    originalOutputSchema: JSON.stringify('path'),
                    originalOutputSpec: JSON.stringify({ path: 'artifact.bin', type: 'path', default: 'X' }),
                }};
                const newPath = 'renamed.bin';
                console.log(JSON.stringify({
                    schema: cliOutputSchemaForRow(row, newPath),
                    spec: cliOutputSpecForRow(row, newPath),
                }));
                """
            )
        )
        proc = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout.strip())

    def test_rename_preserves_type_and_carries_new_path(self) -> None:
        result = self._run_node()
        # The declared type survives the rename (was flattened to 'string').
        self.assertEqual(result["schema"], "path")
        spec = result["spec"]
        self.assertIsInstance(spec, dict)
        # type/default survive and the spec carries the NEW path.
        self.assertEqual(spec["type"], "path")
        self.assertEqual(spec["default"], "X")
        self.assertEqual(spec["path"], "renamed.bin")


if __name__ == "__main__":
    unittest.main()
