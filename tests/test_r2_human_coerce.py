"""R2 Round-3: human form responses honour their declared output type.

The dashboard added JSON coercion for array/object human outputs, but two gaps
remained:

  A8 — declared scalar types (bool/integer/number) were never coerced, so a
       submitted ``"false"`` reached downstream nodes as a truthy string.
  B7 — a JSON-schema-form declaration ({type: array}) was flattened to
       ``string`` by HumanAgent._output_fields before coercion ever saw it, so
       even a valid array response was stored as a raw string.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.dev import create_app  # noqa: E402
from app.dev_data import coerce_human_response_value, human_response_type_error  # noqa: E402
from proofstack.agents.human_agent import HumanAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class CoerceValueTests(unittest.TestCase):
    """A8/B7: the coercion covers scalars and containers, and stays safe."""

    def test_bool(self) -> None:
        self.assertIs(coerce_human_response_value("false", "bool"), False)
        self.assertIs(coerce_human_response_value("true", "boolean"), True)
        self.assertIs(coerce_human_response_value("", "bool"), False)
        # an unrecognised value is left for output validation, not forced
        self.assertEqual(coerce_human_response_value("maybe", "bool"), "maybe")

    def test_integer(self) -> None:
        self.assertEqual(coerce_human_response_value("2", "integer"), 2)
        self.assertEqual(coerce_human_response_value("2", "int"), 2)
        # a boolean must not satisfy an integer field (bool is an int subclass)
        self.assertEqual(coerce_human_response_value("true", "integer"), "true")
        # a non-integer literal is left verbatim
        self.assertEqual(coerce_human_response_value("1.5", "integer"), "1.5")

    def test_number(self) -> None:
        self.assertEqual(coerce_human_response_value("0.5", "number"), 0.5)
        self.assertEqual(coerce_human_response_value("3", "number"), 3)
        self.assertEqual(coerce_human_response_value("false", "number"), "false")

    def test_array_and_object(self) -> None:
        self.assertEqual(coerce_human_response_value('["a", "b"]', "array"), ["a", "b"])
        self.assertEqual(coerce_human_response_value('{"k": 1}', "object"), {"k": 1})
        # wrong shape for the declared type -> untouched
        self.assertEqual(coerce_human_response_value('{"k": 1}', "array"), '{"k": 1}')

    def test_string_passthrough(self) -> None:
        self.assertEqual(coerce_human_response_value("hello", "string"), "hello")
        self.assertEqual(coerce_human_response_value("hello", None), "hello")


class OutputFieldsTypeTests(unittest.TestCase):
    """B7: HumanAgent._output_fields keeps a JSON-schema-form declared type."""

    def _fields(self, schema: dict) -> dict:
        with tempfile.TemporaryDirectory() as td:
            ctx = RunContext.create(
                run_id="run",
                root_workdir=Path(td),
                flat=True,
                component_configs={"human": {"output_schema": schema}},
            )
            return HumanAgent(ctx, name="human")._output_fields()

    def test_string_form_scalar_kept(self) -> None:
        self.assertEqual(self._fields({"approved": "bool"}), {"approved": "bool"})

    def test_json_schema_form_type_extracted(self) -> None:
        self.assertEqual(
            self._fields({"items": {"type": "array", "items": {"type": "string"}}}),
            {"items": "array"},
        )

    def test_unknown_spec_defaults_to_string(self) -> None:
        self.assertEqual(self._fields({"x": {"no_type": 1}}), {"x": "string"})


class HumanPostCoercionTests(unittest.TestCase):
    """A8 end-to-end: a submitted bool is stored as a bool, not a truthy str."""

    def test_bool_field_is_coerced_on_submit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            inbox = run_dir / "human_inbox"
            inbox.mkdir(parents=True)
            ctx = RunContext.create(
                run_id="run",
                root_workdir=root,
                flat=True,
                component_configs={"human": {"output_schema": {"approved": "bool"}}},
            )
            declared = HumanAgent(ctx, name="human")._output_fields()
            task_path = run_dir / "task.json"
            task_path.write_text(json.dumps({"output_fields": declared}), encoding="utf-8")
            response = inbox / "task.response.json"
            (run_dir / "run-metadata.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            (run_dir / "events.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "human.waiting",
                        "payload": {
                            "task_path": str(task_path),
                            "response_path": str(response),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                client.post(
                    "/run/run/human",
                    data={"response_filename": "task.response.json", "f_approved": "false"},
                )
            written = json.loads(response.read_text(encoding="utf-8"))
            self.assertIs(written["approved"], False)


class TypeErrorTests(unittest.TestCase):
    """B5: the validator flags a coerced value that doesn't match its type."""

    def test_matches_pass(self) -> None:
        self.assertIsNone(human_response_type_error([1], "array"))
        self.assertIsNone(human_response_type_error({"k": 1}, "object"))
        self.assertIsNone(human_response_type_error(True, "bool"))
        self.assertIsNone(human_response_type_error(3, "integer"))
        self.assertIsNone(human_response_type_error(0.5, "number"))
        self.assertIsNone(human_response_type_error("anything", "string"))

    def test_mismatches_flagged(self) -> None:
        # a malformed structured value stays a raw string -> flagged
        self.assertIsNotNone(human_response_type_error('["x"]', "object"))
        self.assertIsNotNone(human_response_type_error("nope", "array"))
        self.assertIsNotNone(human_response_type_error("maybe", "bool"))
        self.assertIsNotNone(human_response_type_error("1.5", "integer"))
        # a bool must not satisfy a numeric field
        self.assertIsNotNone(human_response_type_error(True, "integer"))


class HumanPostRejectionTests(unittest.TestCase):
    """B5 end-to-end: a malformed structured answer is rejected, task pending."""

    def _setup(self, td: str, schema: dict):
        root = Path(td)
        run_dir = root / "run"
        inbox = run_dir / "human_inbox"
        inbox.mkdir(parents=True)
        ctx = RunContext.create(
            run_id="run", root_workdir=root, flat=True,
            component_configs={"human": {"output_schema": schema}},
        )
        declared = HumanAgent(ctx, name="human")._output_fields()
        task_path = run_dir / "task.json"
        task_path.write_text(json.dumps({"output_fields": declared}), encoding="utf-8")
        response = inbox / "task.response.json"
        (run_dir / "run-metadata.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            json.dumps({"kind": "human.waiting", "payload": {
                "task_path": str(task_path), "response_path": str(response)}}) + "\n",
            encoding="utf-8",
        )
        return root, response

    def test_valid_object_is_stored_as_dict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root, response = self._setup(td, {"data": "object"})
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                r = client.post("/run/run/human", data={
                    "response_filename": "task.response.json", "f_data": '{"k": 1}'})
            self.assertEqual(r.status_code, 302)
            self.assertEqual(json.loads(response.read_text())["data"], {"k": 1})

    def test_malformed_object_is_rejected_and_task_stays_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root, response = self._setup(td, {"data": "object"})
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                r = client.post("/run/run/human", data={
                    "response_filename": "task.response.json", "f_data": '["not", "an", "object"]'})
            self.assertEqual(r.status_code, 400)
            self.assertFalse(response.exists())  # nothing written -> still pending


if __name__ == "__main__":
    unittest.main()
