from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.cli_usage import (
    ClaudeUsage,
    cost_for_claude_usage,
    load_cost_rates,
    parse_claude_json,
)
from proofstack.context import RunContext
from proofstack.kinds.cli import CLIDoneRecord


COST_CONFIG = "models/anthropic/opus_46"
RATES = dict(read_cost=5, write_cost=25, cache_read_cost=0.5, cache_write_cost=6.25)
TOKENS = dict(
    input_tokens=100,
    cache_creation_input_tokens=200,
    cache_read_input_tokens=300,
    output_tokens=40,
)
ESTIMATE = (100 * 5 + 200 * 6.25 + 300 * 0.5 + 40 * 25) / 1_000_000


def result(**overrides) -> str:
    return json.dumps(dict(
        {"type": "result", "num_turns": 2, "total_cost_usd": 0.42, "usage": TOKENS},
        **overrides,
    ))


def assistant(mid: str | None = "msg_a", **tokens) -> str:
    return json.dumps({"type": "assistant", "message": {
        "id": mid, "usage": tokens or TOKENS,
    }})


class ClaudePaidCostTests(unittest.TestCase):
    def test_async_reviews_use_complete_model_totals_without_double_counting(self) -> None:
        models = {"claude-fable-5-1": {
            "inputTokens": 44, "outputTokens": 14709, "cacheReadInputTokens": 209725,
            "cacheCreationInputTokens": 35283, "costUSD": 0.69332625,
        }}
        first = result(modelUsage=models)
        last = result(modelUsage=models, usage={"input_tokens": 12, "output_tokens": 3762})
        usage = parse_claude_json(first + "\n" + last)
        self.assertEqual(usage.input_tokens, 44)
        self.assertEqual(usage.output_tokens, 14709)
        self.assertEqual(usage.cache_read_input_tokens, 209725)
        self.assertEqual(usage.cache_creation_input_tokens, 35283)
        self.assertEqual(usage.metered_tokens, 259761)

    def test_unknown_cli_prices_use_configured_model_rates_including_helpers(self) -> None:
        models = {"claude-fable-5-1": {
            "costBasis": "unknown", "canonicalModel": "claude-fable-5-1",
            "inputTokens": 24, "cacheCreationInputTokens": 17318,
            "cacheReadInputTokens": 108625, "outputTokens": 8394, "costUSD": 0.37252,
        }}
        text = result(total_cost_usd=0.38252, modelUsage=models)
        usage = parse_claude_json(text)
        self.assertTrue(usage.cost_estimated)
        self.assertAlmostEqual(cost_for_claude_usage(text, cost_config="models/anthropic/fable_51"),
                               0.67357125)
        # Correct only the unknown-priced model, retaining another model's $0.01.
        self.assertEqual(usage.total_cost_usd, 0.38252)
        with self.assertRaisesRegex(ValueError, "no configured pricing"):
            cost_for_claude_usage(text, cost_config=COST_CONFIG)
        with self.assertRaisesRegex(ValueError, "exceed"):
            cost_for_claude_usage(result(total_cost_usd=0.1, modelUsage=models),
                                  cost_config="models/anthropic/fable_51")
        models["claude-fable-5-1"].pop("outputTokens")
        with self.assertRaisesRegex(ValueError, "complete per-model"):
            cost_for_claude_usage(result(modelUsage=models), cost_config="models/anthropic/fable_51")

    def test_compacted_stream_preserves_unknown_pricing_metadata(self) -> None:
        from proofstack.sandbox.subprocess import _JsonUsageCapture
        models = {"claude-fable-5-1": {"costBasis": "unknown"}}
        capture = _JsonUsageCapture()
        capture.feed(result(modelUsage=models) + "\n")
        self.assertEqual(parse_claude_json(capture.text()).model_usage, models)

    def test_result_cost_is_authoritative_over_snapshots_and_estimate(self) -> None:
        text = "\n".join([assistant(), assistant(), result(), result()])
        usage = parse_claude_json(text)
        self.assertEqual(usage.metered_tokens, 640)
        self.assertEqual(usage.num_turns, 2)
        self.assertEqual(cost_for_claude_usage(usage), 0.42)
        self.assertEqual(cost_for_claude_usage(text, cost_config=COST_CONFIG), 0.42)

    def test_pretty_printed_and_bare_results(self) -> None:
        obj = json.loads(result())
        for typed in (True, False):
            if not typed:
                obj.pop("type")
            with self.subTest(typed=typed):
                self.assertEqual(cost_for_claude_usage(json.dumps(obj, indent=2)), 0.42)

    def test_compacted_result_can_omit_turn_count(self) -> None:
        self.assertEqual(cost_for_claude_usage(result(num_turns=None)), 0.42)

    def test_interrupted_stream_uses_last_snapshot_per_message_including_caches(self) -> None:
        text = "\n".join([
            assistant(output_tokens=1), assistant(), assistant(), assistant("msg_b"),
            '{"type":"assistant","message":',
        ])
        usage = parse_claude_json(text)
        self.assertIsNone(usage.total_cost_usd)
        self.assertEqual(usage.num_turns, 2)
        self.assertEqual(usage.metered_tokens, 1280)
        self.assertAlmostEqual(cost_for_claude_usage(usage, **RATES), 2 * ESTIMATE)
        self.assertAlmostEqual(
            cost_for_claude_usage(text, cost_config=COST_CONFIG), 2 * ESTIMATE
        )

    def test_cache_only_usage_is_found(self) -> None:
        text = assistant(cache_read_input_tokens=1000, cache_creation_input_tokens=10)
        self.assertTrue(parse_claude_json(text).found)
        self.assertAlmostEqual(
            cost_for_claude_usage(text, **RATES), (1000 * 0.5 + 10 * 6.25) / 1_000_000
        )

    def test_long_context_is_priced_per_message_including_cached_input(self) -> None:
        rates = dict(
            RATES, long_context_threshold_tokens=600,
            long_context_input_multiplier=2, long_context_output_multiplier=1.5,
        )
        self.assertAlmostEqual(
            cost_for_claude_usage("\n".join([assistant(), assistant("msg_b")]), **rates),
            2 * ESTIMATE,
        )
        text = assistant(**dict(TOKENS, cache_read_input_tokens=301))
        self.assertAlmostEqual(
            cost_for_claude_usage(text, **rates),
            ((100 * 5 + 200 * 6.25 + 301 * 0.5) * 2 + 40 * 25 * 1.5) / 1_000_000,
        )

    def test_unknown_usage_is_not_free(self) -> None:
        for text in (
            "", "not json", result(usage=None), result(usage={}),
            '{"type":"result","num_turns":5}',
            '{"type":"assistant","message":{"id":"a","usage":{}}}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cost_for_claude_usage(text, **RATES)
        with self.assertRaises(ValueError):
            cost_for_claude_usage(ClaudeUsage(num_turns=2), **RATES)

    def test_result_missing_cost_does_not_fall_back_to_estimate(self) -> None:
        obj = json.loads(result())
        obj.pop("total_cost_usd")
        with self.assertRaisesRegex(ValueError, "missing total_cost_usd"):
            cost_for_claude_usage(assistant() + "\n" + json.dumps(obj), **RATES)

    def test_invalid_reported_costs_fail(self) -> None:
        for cost in (None, -1, float("nan"), float("inf"), "unknown", True):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                cost_for_claude_usage(result(total_cost_usd=cost), **RATES)

    def test_reported_zero_is_distinct_from_unknown(self) -> None:
        self.assertEqual(cost_for_claude_usage(result(total_cost_usd=0)), 0)
        self.assertIsNone(parse_claude_json(assistant()).total_cost_usd)

    def test_invalid_token_counts_fail_even_when_cost_is_reported(self) -> None:
        for field in TOKENS:
            for value in (-1, 1.5, None, True, "100", float("nan")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    cost_for_claude_usage(result(usage=dict(TOKENS, **{field: value})))

    def test_missing_message_ids_cannot_be_safely_estimated(self) -> None:
        with self.assertRaisesRegex(ValueError, "message ids"):
            cost_for_claude_usage(assistant(None) + "\n" + assistant(None), **RATES)
        self.assertEqual(cost_for_claude_usage(assistant(None) + "\n" + result()), 0.42)

    def test_interrupted_stream_requires_all_rates(self) -> None:
        for key in RATES:
            for value in (None, 0, -1, float("nan"), float("inf"), True, "5"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cost_for_claude_usage(assistant(), **dict(RATES, **{key: value}))
        with self.assertRaises(ValueError):
            cost_for_claude_usage(assistant())

    def test_strict_cost_loader_rejects_missing_and_invalid_rates(self) -> None:
        for key in RATES:
            for value in (None, 0, -1, float("nan"), float("inf"), True):
                cfg = dict(RATES, **{key: value})
                with self.subTest(key=key, value=value), mock.patch(
                    "mathagents.config_loader.load_solver_config", return_value=cfg
                ), self.assertRaises(ValueError):
                    load_cost_rates("offline", require_cache_rates=True)
            cfg = dict(RATES)
            cfg.pop(key)
            with mock.patch("mathagents.config_loader.load_solver_config", return_value=cfg):
                with self.assertRaises(ValueError):
                    load_cost_rates("offline", require_cache_rates=True)

    def test_strict_cost_loader_rejects_invalid_context_rates_and_cache_semantics(self) -> None:
        for key, value in (
            ("cache_write_tokens_in_input", True),
            ("long_context_threshold_tokens", 0),
            ("long_context_threshold_tokens", 1.5),
            ("long_context_threshold_tokens", True),
            ("long_context_input_multiplier", float("nan")),
            ("long_context_output_multiplier", 0),
        ):
            with self.subTest(key=key), mock.patch(
                "mathagents.config_loader.load_solver_config",
                return_value=dict(RATES, **{key: value}),
            ), self.assertRaises(ValueError):
                load_cost_rates("offline", require_cache_rates=True)


class ClaudePaidAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch(
            "proofstack.sandbox.subprocess._terminate_marked_processes",
            new=mock.AsyncMock(return_value=True),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, *, usage=None, **overrides) -> ConfigurableCLIAgent:
        config = {
            "cmd": ["true"], "completion_signal": "exit",
            "usage": usage if usage is not None else {
                "type": "claude_json", "auth_mode": "api", "cost_config": COST_CONFIG,
            },
            "sandbox": {"backend": "subprocess"},
            **overrides,
        }
        ctx = RunContext.create(
            run_id="offline", root_workdir=self.root, flat=True,
            component_configs={"claude": config},
        )
        agent = ConfigurableCLIAgent(ctx, name="claude")
        agent._copied_codex_auth = False
        agent.WORKSPACE_RESERVATION_BYTES = 1
        agent.WORKSPACE_MIN_FREE_BYTES = 0
        agent.WORKSPACE_RESERVATION_DIR = self.root / "leases"
        agent.WORKSPACE_RECOVERY_ENABLED = True
        return agent

    def meter(self, agent, text=result()) -> None:
        asyncio.run(agent.record_cli_usage(text, "", CLIDoneRecord()))

    def model_events(self) -> list[dict]:
        return [
            event["payload"] for line in (self.root / "events.jsonl").read_text().splitlines()
            if (event := json.loads(line))["kind"] == "model.call"
        ]

    def test_paid_usage_charges_actual_usd_and_all_tokens_ignoring_bill_false(self) -> None:
        agent = self.agent(usage={
            "type": "claude_json", "auth_mode": "api", "cost_config": COST_CONFIG,
            "bill": False, "model": "configured-claude",
        })
        self.meter(agent, assistant() + "\n" + result())
        self.assertTrue(agent.cli_usage_must_succeed())
        self.assertEqual(agent.tracker.counters.usd, 0.42)
        self.assertEqual(agent.tracker.parent.counters.usd, 0.42)
        self.assertEqual(agent.tracker.counters.tokens, 640)
        event, = self.model_events()
        self.assertEqual(event["cost_usd"], 0.42)
        self.assertEqual(event["cost_config"], COST_CONFIG)
        self.assertEqual(event["model"], "configured-claude")
        self.assertEqual(event["auth_mode"], "api")
        self.assertFalse(event["cost_estimated"])

    def test_partial_usage_is_charged_and_marked_estimated(self) -> None:
        agent = self.agent()
        self.meter(agent, assistant() + "\n" + assistant())
        self.assertAlmostEqual(agent.tracker.counters.usd, ESTIMATE)
        self.assertEqual(agent.tracker.counters.tokens, 640)
        event, = self.model_events()
        self.assertTrue(event["cost_estimated"])
        self.assertAlmostEqual(event["cost_usd"], ESTIMATE)

    def test_event_is_persisted_before_usd_or_token_budget_check(self) -> None:
        for spec in (BudgetSpec(max_usd=0.01), BudgetSpec(max_tokens=10)):
            with self.subTest(spec=spec):
                agent = self.agent()
                agent.tracker.spec = spec
                self.meter(agent)
                with self.assertRaises(BudgetExhausted):
                    agent.tracker.check()
                self.assertEqual(agent.tracker.counters.usd, 0.42)
                self.assertEqual(agent.tracker.counters.tokens, 640)
                self.assertEqual(self.model_events()[-1]["cost_usd"], 0.42)

    def test_subscription_remains_default_and_does_not_charge_usd(self) -> None:
        for usage in ({"type": "claude_json"}, {"type": "claude_json", "auth_mode": "subscription"}):
            with self.subTest(usage=usage):
                agent = self.agent(usage=usage)
                agent.tracker.spec = BudgetSpec(max_usd=0)
                self.meter(agent)
                self.assertFalse(agent.cli_usage_must_succeed())
                self.assertEqual(agent.tracker.counters.usd, 0)
                self.assertEqual(agent.tracker.counters.tokens, 640)
                self.assertNotIn("ANTHROPIC_API_KEY", agent.SANDBOX.provider_keys)
                self.meter(agent, "no usage")

    def test_invalid_auth_modes_fail_during_construction(self) -> None:
        for mode in ("paid", "API", "", None, False):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "auth_mode"):
                self.agent(usage={"type": "claude_json", "auth_mode": mode})

    def test_missing_or_invalid_cost_config_fails_before_spawn(self) -> None:
        for cfg in (None, "", "models/anthropic/does-not-exist"):
            with self.subTest(cfg=cfg), mock.patch(
                "proofstack.sandbox.subprocess.SubprocessSandbox.stream_command"
            ) as spawn, self.assertRaises(ValueError):
                self.agent(usage={"type": "claude_json", "auth_mode": "api", "cost_config": cfg})
            spawn.assert_not_called()
        with mock.patch("mathagents.config_loader.load_solver_config", return_value={
            **RATES, "cache_write_cost": float("nan"),
        }), self.assertRaisesRegex(ValueError, "before the component starts"):
            self.agent()

    def test_api_key_preserved_in_all_sandbox_environment_channels(self) -> None:
        for channel in ("provider_keys", "env_allowlist", "extra_env", "env"):
            sandbox_cfg = {"backend": "subprocess", "provider_keys": [], "env_allowlist": []}
            env = {}
            if channel == "env":
                env = {"ANTHROPIC_API_KEY": "{env:ANTHROPIC_API_KEY}"}
            elif channel == "extra_env":
                sandbox_cfg[channel] = {"ANTHROPIC_API_KEY": "offline-test-key"}
            else:
                sandbox_cfg[channel] = ["ANTHROPIC_API_KEY"]
            with self.subTest(channel=channel):
                agent = self.agent(sandbox=sandbox_cfg, env=env)
                sandbox = SimpleNamespace(root=self.root, spec=agent.SANDBOX)
                effective = agent.SANDBOX.build_env(sandbox_root=self.root)
                effective.update(agent.extra_env(sandbox, agent.Inputs()))
                self.assertEqual(effective["ANTHROPIC_API_KEY"], "offline-test-key")

    def test_api_auth_rejects_missing_or_blank_effective_key_before_spawn(self) -> None:
        for env in ({}, {"ANTHROPIC_API_KEY": "   "}):
            agent = self.agent(sandbox={"backend": "subprocess", "provider_keys": []}, env=env)
            with self.subTest(env=env), mock.patch(
                "proofstack.sandbox.subprocess.SubprocessSandbox.stream_command"
            ) as spawn, self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                asyncio.run(agent(workspace=self.root / "workspace"))
            spawn.assert_not_called()

    def test_docker_auth_validation_uses_its_environment_precedence(self) -> None:
        agent = self.agent(
            sandbox={"backend": "docker", "extra_env": {"ANTHROPIC_API_KEY": ""}},
            env={"ANTHROPIC_API_KEY": "offline-test-key"},
        )
        sandbox = SimpleNamespace(root=self.root, spec=agent.SANDBOX)
        with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
            agent.extra_env(sandbox, agent.Inputs())

    def test_api_auth_rejects_conflicting_authentication(self) -> None:
        for key in ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK"):
            agent = self.agent(env={key: "offline-conflict"})
            sandbox = SimpleNamespace(root=self.root, spec=agent.SANDBOX)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                agent.extra_env(sandbox, agent.Inputs())

    def test_paid_run_with_missing_or_invalid_usage_fails_closed(self) -> None:
        for text in ("no usage", result(total_cost_usd=-1), result(usage={})):
            agent = self.agent(cmd=[sys.executable, "-c", f"print({text!r})"])
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "required CLI usage accounting failed"):
                asyncio.run(agent(workspace=self.root / "workspace"))
            self.assertEqual(agent.tracker.counters.usd, 0)
            self.assertEqual(agent.tracker.counters.tokens, 0)

    def test_successful_local_run_bills_exactly_once(self) -> None:
        text = "\n".join([assistant(), assistant(), result(), result()])
        agent = self.agent(cmd=[sys.executable, "-c", f"print({text!r})"])
        asyncio.run(agent(workspace=self.root / "workspace"))
        self.assertEqual(agent.tracker.counters.usd, 0.42)
        self.assertEqual(agent.tracker.counters.tokens, 640)
        self.assertEqual(len(self.model_events()), 1)

    def test_over_budget_local_run_preserves_event_and_bills_once(self) -> None:
        agent = self.agent(cmd=[sys.executable, "-c", f"print({result()!r})"])
        agent.tracker.spec = BudgetSpec(max_usd=0.01)
        asyncio.run(agent(workspace=self.root / "workspace"))
        self.assertEqual(agent.tracker.counters.usd, 0.42)
        self.assertEqual(agent.tracker.counters.tokens, 640)
        self.assertEqual(len(self.model_events()), 1)

    def test_cancelled_local_stream_bills_partial_usage_exactly_once(self) -> None:
        text = "\n".join([assistant(), assistant()])
        agent = self.agent(cmd=[sys.executable, "-c", (
            f"import pathlib,time; print({text!r}, flush=True); "
            "pathlib.Path('ready').touch(); time.sleep(30)"
        )])
        workspace = self.root / "workspace"

        async def run_and_cancel():
            task = asyncio.create_task(agent(workspace=workspace))
            try:
                async with asyncio.timeout(10):
                    while not (workspace / "ready").exists():
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run_and_cancel())
        self.assertAlmostEqual(agent.tracker.counters.usd, ESTIMATE)
        self.assertEqual(agent.tracker.counters.tokens, 640)
        self.assertEqual(len(self.model_events()), 1)

    def test_cancelled_local_run_without_usage_fails_closed(self) -> None:
        agent = self.agent(cmd=[sys.executable, "-c", (
            "import pathlib,time; pathlib.Path('ready').touch(); time.sleep(30)"
        )])
        workspace = self.root / "workspace"

        async def run_and_cancel():
            task = asyncio.create_task(agent(workspace=workspace))
            try:
                async with asyncio.timeout(10):
                    while not (workspace / "ready").exists():
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaisesRegex(RuntimeError, "required CLI usage accounting failed"):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run_and_cancel())
        self.assertEqual(agent.tracker.counters.usd, 0)


if __name__ == "__main__":
    unittest.main()
