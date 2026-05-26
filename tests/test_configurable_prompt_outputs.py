from __future__ import annotations

import sys
import asyncio
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_prompt import ConfigurablePromptAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class _EmptyClient:
    model = "fake-empty-model"

    def run_queries(self, queries, no_tqdm=False):
        messages = list(queries[0])
        yield (
            0,
            [*messages, {"role": "assistant", "content": ""}],
            {"cost": 0.0, "input_tokens": 1, "output_tokens": 0},
        )


class _CaptureClient:
    model = "fake-capture-model"

    def __init__(self):
        self.queries = []

    def run_queries(self, queries, no_tqdm=False):
        messages = list(queries[0])
        self.queries.append(messages)
        yield (
            0,
            [*messages, {"role": "assistant", "content": "ok"}],
            {"cost": 0.0, "input_tokens": 1, "output_tokens": 1},
        )


class ConfigurablePromptOutputTests(unittest.TestCase):
    def test_named_fallback_does_not_also_emit_text_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "verifier": {
                        "output": {
                            "xml_tags": ["verification"],
                            "default_field": "verification",
                        }
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="verifier")

            out = agent.parse_output("Plain verification report.", agent.Inputs())

        self.assertEqual(out.model_dump(mode="json"), {"verification": "Plain verification report."})

    def test_default_fallback_still_emits_text_when_configured_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={"plain": {"output": {}}},
            )
            agent = ConfigurablePromptAgent(ctx, name="plain")

            out = agent.parse_output("Plain response.", agent.Inputs())

        self.assertEqual(out.model_dump(mode="json"), {"text": "Plain response."})

    def test_empty_response_fills_every_configured_output_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "verifier": {
                        "output": {
                            "xml_tags": ["verification", "verdict"],
                            "xml_lists": {"issues": "issue"},
                            "regex_fields": {"confidence": r"<confidence>(.*?)</confidence>"},
                            "default_field": "verification",
                        }
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="verifier")

            out = agent.parse_output("", agent.Inputs())

        self.assertEqual(
            out.model_dump(mode="json"),
            {"issues": [], "verification": "", "verdict": "", "confidence": ""},
        )

    def test_multiple_json_tags_are_parsed_into_named_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "critic": {
                        "output": {
                            "xml_tags": ["review"],
                            "json_tags": {"findings": "findings", "decision": "decision"},
                            "json_defaults": {
                                "findings": [],
                                "decision": {"answer_ready": False},
                            },
                            "default_field": "review",
                        }
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="critic")

            out = agent.parse_output(
                '<findings>[{"tag":"NIT","issue":"typo"}]</findings>'
                "<review>Looks fine.</review>"
                '<decision>{"answer_ready": true, "has_blockers_or_concerns": false}</decision>',
                agent.Inputs(),
            )

        self.assertEqual(out.review, "Looks fine.")
        self.assertEqual(out.findings, [{"tag": "NIT", "issue": "typo"}])
        self.assertEqual(out.decision["answer_ready"], True)
        self.assertEqual(out.decision["has_blockers_or_concerns"], False)

    def test_empty_response_continues_but_records_debug_error_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp)
            ctx = RunContext.create(
                run_id="test",
                root_workdir=run_path,
                flat=True,
                api_client_factory=lambda _cfg: _EmptyClient(),
                component_configs={
                    "verifier": {
                        "messages": [{"role": "user", "content": "verify"}],
                        "output": {
                            "xml_tags": ["verification", "verdict"],
                            "default_field": "verification",
                        },
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="verifier")

            out = asyncio.run(agent())
            events = (run_path / "events.jsonl").read_text(encoding="utf-8")

        self.assertEqual(out.model_dump(mode="json"), {"verification": "", "verdict": ""})
        self.assertIn('"kind": "model.empty_response"', events)
        self.assertIn('"type": "EmptyResponse"', events)

    def test_local_function_tool_refs_leave_tool_loop_budget_to_api_client_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "writer": {
                        "tool_refs": ["list_persisted_files"],
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="writer")

            kwargs = agent.extra_client_kwargs()

        self.assertNotIn("max_tool_calls", kwargs)
        self.assertEqual(kwargs["tools"][0][1]["function"]["name"], "list_persisted_files")
        self.assertTrue(callable(kwargs["tools"][0][0]))

    def test_provider_managed_tool_refs_do_not_force_a_local_tool_loop_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "writer": {
                        "tool_refs": ["code_interpreter", "web_search_preview"],
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="writer")

            kwargs = agent.extra_client_kwargs()

        self.assertNotIn("max_tool_calls", kwargs)
        self.assertEqual([tool[1]["type"] for tool in kwargs["tools"]], ["code_interpreter", "web_search_preview"])

    def test_persisted_file_context_is_shared_across_nodes_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clients = []

            def factory(_cfg):
                client = _CaptureClient()
                clients.append(client)
                return client

            ctx = RunContext.create(
                run_id="shared",
                root_workdir=Path(tmp),
                flat=True,
                api_client_factory=factory,
                component_configs={
                    "first": {"messages": [{"role": "user", "content": "first"}], "output": {"default_field": "text"}},
                    "second": {"messages": [{"role": "user", "content": "second"}], "output": {"default_field": "text"}},
                },
            )

            asyncio.run(ConfigurablePromptAgent(ctx, name="first")())
            asyncio.run(ConfigurablePromptAgent(ctx, name="second")())

        roots = [
            client.queries[0][0]["tool_context"]["persisted_file_root"]
            for client in clients
        ]
        self.assertEqual(roots, [str(Path(tmp) / "persisted_files"), str(Path(tmp) / "persisted_files")])

    def test_persisted_file_context_is_isolated_between_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_roots = []
            clients = []

            def factory(_cfg):
                client = _CaptureClient()
                clients.append(client)
                return client

            for run_id in ("one", "two"):
                root = Path(tmp) / run_id
                run_roots.append(root)
                ctx = RunContext.create(
                    run_id=run_id,
                    root_workdir=root,
                    flat=True,
                    api_client_factory=factory,
                    component_configs={
                        "writer": {
                            "messages": [{"role": "user", "content": run_id}],
                            "output": {"default_field": "text"},
                        }
                    },
                )
                asyncio.run(ConfigurablePromptAgent(ctx, name="writer")())

        roots = [
            client.queries[0][0]["tool_context"]["persisted_file_root"]
            for client in clients
        ]
        self.assertEqual(roots, [str(root / "persisted_files") for root in run_roots])
        self.assertNotEqual(roots[0], roots[1])


if __name__ == "__main__":
    unittest.main()
