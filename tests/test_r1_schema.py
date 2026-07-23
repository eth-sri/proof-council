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
    mutate_preset_yaml,
    validate_preset_yaml,
)
from proofstack.agents.configurable_prompt import ConfigurablePromptAgent  # noqa: E402


def _mutate(raw_yaml: str, operation: dict[str, Any]) -> str:
    result = mutate_preset_yaml(raw_yaml, operation)
    assert result["ok"], result["errors"]
    return result["raw_yaml"]


def _component(raw_yaml: str, name: str) -> dict[str, Any]:
    return yaml.safe_load(raw_yaml)["components"][name]


# --- P-A3: structured API output spec survives an output_schema edit ----------

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


class SchemaProjectionA3Tests(unittest.TestCase):
    def _edit_dropping_note(self) -> dict[str, Any]:
        # Re-edit the schema (dropping `note`), keeping the two structured fields.
        raw = _mutate(
            API_STRUCTURED_FIXTURE,
            {
                "op": "update_component",
                "name": "judge",
                "fields": {"output_schema": {"decision": "object", "items": "array"}},
            },
        )
        self.assertEqual(validate_preset_yaml(raw)["errors"], [])
        return _component(raw, "judge")

    def test_structured_bindings_survive(self) -> None:
        cfg = self._edit_dropping_note()
        output = cfg["output"]
        # The json/array parse bindings are preserved, not flattened to string tags.
        self.assertEqual(output.get("json_tags"), {"decision": "decision"})
        self.assertEqual(output.get("xml_lists"), {"items": "item"})
        # decision/items must NOT appear as plain xml_tags (that was the bug).
        self.assertNotIn("decision", output.get("xml_tags") or [])
        self.assertNotIn("items", output.get("xml_tags") or [])

    def test_removed_field_pruned_from_spec(self) -> None:
        cfg = self._edit_dropping_note()
        output = cfg["output"]
        self.assertNotIn("note", output.get("xml_tags") or [])
        self.assertNotEqual(output.get("default_field"), "note")
        self.assertNotIn("note", cfg["output_schema"])

    def test_runtime_parses_object_and_array(self) -> None:
        cfg = self._edit_dropping_note()
        agent = ConfigurablePromptAgent.__new__(ConfigurablePromptAgent)
        agent.component_config = cfg
        raw_resp = (
            '<decision>{"verdict": "accept", "score": 9}</decision>\n'
            "<item>a</item>\n<item>b</item>"
        )
        data = agent.parse_output(raw_resp, None).model_dump()
        self.assertIsInstance(data.get("decision"), dict)
        self.assertEqual(data["decision"], {"verdict": "accept", "score": 9})
        self.assertIsInstance(data.get("items"), list)
        self.assertEqual(data["items"], ["a", "b"])


# --- P-B2: done-only CLI component gets a synthesized file for new fields ------

CLI_DONE_ONLY_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      solver:
        contract: auto
        cmd: [claude, -p]
        prompt: solve {problem}
        input_schema:
          problem: string
          workspace: string
        output_schema:
          workspace: string
          status: string
        done_outputs:
          status: status
    dag:
      nodes:
        - id: solve
          kind: agent
          agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
          name: solver
          inputs:
            problem: $input.problem
      outputs:
        status: $node.solve.status
    """
)

API_NO_FILES_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      judge:
        model: models/openai/gpt-54
        prompt: judge {problem}
        output_schema:
          verdict: string
        output:
          xml_tags: [verdict]
          default_field: verdict
    dag:
      nodes:
        - id: j
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: judge
          inputs:
            problem: $input.problem
      outputs:
        verdict: $node.j.verdict
    """
)


class SchemaProjectionB2Tests(unittest.TestCase):
    def test_new_field_synthesizes_output_file(self) -> None:
        raw = _mutate(
            CLI_DONE_ONLY_FIXTURE,
            {
                "op": "update_component",
                "name": "solver",
                "fields": {
                    "output_schema": {
                        "workspace": "string",
                        "status": "string",
                        "answer_tex": "string",
                    }
                },
            },
        )
        self.assertEqual(validate_preset_yaml(raw)["errors"], [])
        cfg = _component(raw, "solver")
        files = cfg.get("output_files")
        self.assertIsInstance(files, dict)
        # The genuinely-new field gets a delivery file so the model is told to write it.
        self.assertIn("answer_tex", files)
        # A done_outputs-supplied field is not shadowed by a synthetic empty file.
        self.assertNotIn("status", files)

    def test_api_component_never_grows_output_files(self) -> None:
        raw = _mutate(
            API_NO_FILES_FIXTURE,
            {
                "op": "update_component",
                "name": "judge",
                "fields": {"output_schema": {"verdict": "string", "reason": "string"}},
            },
        )
        cfg = _component(raw, "judge")
        self.assertNotIn("output_files", cfg)


# --- P-B4: collision check normalizes paths (./notes.txt == notes.txt) --------

CLI_CUSTOM_PATH_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      solver:
        contract: auto
        cmd: [claude, -p]
        prompt: solve {problem}
        input_schema:
          problem: string
          workspace: string
        output_schema:
          workspace: string
          draft: string
        output_files:
          draft:
            path: ./notes.txt
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


class SchemaProjectionB4Tests(unittest.TestCase):
    @staticmethod
    def _norm(spec: Any) -> str:
        raw = spec.get("path") if isinstance(spec, dict) else spec
        return Path(str(raw)).as_posix()

    def test_new_field_avoids_normalized_collision(self) -> None:
        raw = _mutate(
            CLI_CUSTOM_PATH_FIXTURE,
            {
                "op": "update_component",
                "name": "solver",
                "fields": {
                    "output_schema": {
                        "workspace": "string",
                        "draft": "string",
                        "notes": "string",
                    }
                },
            },
        )
        self.assertEqual(validate_preset_yaml(raw)["errors"], [])
        files = _component(raw, "solver")["output_files"]
        # ./notes.txt and notes.txt are one physical file, so `notes` must dodge it.
        self.assertEqual(self._norm(files["draft"]), "notes.txt")
        self.assertEqual(self._norm(files["notes"]), "notes_2.txt")
        physical = {self._norm(spec) for spec in files.values()}
        self.assertEqual(len(physical), len(files))


# --- P-B3: editor JS keeps type/default when a CLI output path is renamed -----

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
