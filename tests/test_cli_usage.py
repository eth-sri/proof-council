import json
import unittest

from proofstack.budget import BudgetExhausted, BudgetSpec, BudgetTracker
from proofstack.cli_usage import (
    CodexUsage,
    cost_for_codex_usage,
    load_cost_rates,
    parse_claude_json,
    parse_codex_jsonl,
)


class CodexUsageTests(unittest.TestCase):
    def test_parser_collects_cache_write_tokens_when_cli_exposes_them(self) -> None:
        usage = parse_codex_jsonl(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1000,
                        "cached_input_tokens": 200,
                        "cache_write_tokens": 300,
                        "output_tokens": 10,
                    },
                }
            )
        )

        self.assertEqual(usage.cache_write_input_tokens, 300)

    def test_explicit_cache_write_tokens_use_cache_write_rate(self) -> None:
        usage = CodexUsage(
            input_tokens=1000,
            cached_input_tokens=200,
            cache_write_input_tokens=300,
            output_tokens=10,
        )

        cost = cost_for_codex_usage(
            usage,
            read_cost=5,
            cache_read_cost=0.5,
            cache_write_cost=6.25,
            cache_write_tokens_in_input=True,
            write_cost=30,
        )

        expected = (500 * 5 + 200 * 0.5 + 300 * 6.25 + 10 * 30) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_missing_cli_cache_write_count_is_conservatively_costed(self) -> None:
        usage = CodexUsage(input_tokens=1000, cached_input_tokens=200)

        cost = cost_for_codex_usage(
            usage,
            read_cost=5,
            cache_read_cost=0.5,
            cache_write_cost=6.25,
            cache_write_tokens_in_input=True,
            write_cost=30,
        )

        expected = (200 * 0.5 + 800 * 6.25) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_sol_cost_config_loads_cache_write_rate(self) -> None:
        rates = load_cost_rates("models/openai/gpt-56-sol-pro")

        self.assertEqual(rates["cache_write_cost"], 6.25)
        self.assertTrue(rates["cache_write_tokens_in_input"])


class ParseClaudeJsonTests(unittest.TestCase):
    def test_single_result_object(self) -> None:
        text = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 7,
                "total_cost_usd": 0.42,
                "result": "proof written",
                "usage": {
                    "input_tokens": 1200,
                    "cache_creation_input_tokens": 300,
                    "cache_read_input_tokens": 5000,
                    "output_tokens": 800,
                },
            }
        )
        usage = parse_claude_json(text)
        self.assertTrue(usage.found)
        self.assertEqual(usage.input_tokens, 1200)
        self.assertEqual(usage.cache_read_input_tokens, 5000)
        self.assertEqual(usage.output_tokens, 800)
        self.assertEqual(usage.num_turns, 7)
        self.assertAlmostEqual(usage.total_cost_usd, 0.42)
        self.assertEqual(usage.total_tokens, 2000)

    def test_stream_json_lines_picks_result(self) -> None:
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps({"type": "assistant", "message": {"content": "..."}}),
            json.dumps(
                {
                    "type": "result",
                    "num_turns": 3,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        ]
        usage = parse_claude_json("\n".join(lines))
        self.assertTrue(usage.found)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.num_turns, 3)

    def _assistant(self, mid: str, **usage) -> str:
        return json.dumps({"type": "assistant", "message": {"id": mid, "usage": usage}})

    def test_stream_json_completed_uses_result_total(self) -> None:
        # The real stream emits several assistant snapshots per turn (same id)
        # plus a final result with the CUMULATIVE total. The result is
        # authoritative — duplicate snapshots must not inflate the count.
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            self._assistant("msg_a", input_tokens=10, cache_read_input_tokens=5000, output_tokens=0),
            self._assistant("msg_a", input_tokens=10, cache_read_input_tokens=5000, output_tokens=200),
            self._assistant("msg_b", input_tokens=12, cache_read_input_tokens=5200, output_tokens=300),
            json.dumps(
                {
                    "type": "result",
                    "num_turns": 2,
                    "total_cost_usd": 0.03,
                    "usage": {
                        "input_tokens": 22,
                        "cache_read_input_tokens": 10200,
                        "output_tokens": 500,
                    },
                }
            ),
        ]
        usage = parse_claude_json("\n".join(lines))
        self.assertEqual(usage.input_tokens, 22)
        self.assertEqual(usage.cache_read_input_tokens, 10200)
        self.assertEqual(usage.output_tokens, 500)
        self.assertEqual(usage.num_turns, 2)
        self.assertAlmostEqual(usage.total_cost_usd, 0.03)
        self.assertEqual(usage.metered_tokens, 22 + 10200 + 500)

    def test_stream_json_killed_dedupes_and_sums_turns(self) -> None:
        # No final `result` (killed by a timeout). Reconstruct from per-turn
        # usage, deduping the repeated streaming snapshots by message id so a
        # turn is counted once — and still metered (not zero).
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            self._assistant("msg_a", input_tokens=10, cache_creation_input_tokens=8000, cache_read_input_tokens=50000, output_tokens=900),
            self._assistant("msg_a", input_tokens=10, cache_creation_input_tokens=8000, cache_read_input_tokens=50000, output_tokens=1500),
            self._assistant("msg_b", input_tokens=12, cache_read_input_tokens=60000, output_tokens=1800),
        ]
        usage = parse_claude_json("\n".join(lines))
        self.assertTrue(usage.found)
        self.assertEqual(usage.num_turns, 2)  # two distinct message ids
        self.assertEqual(usage.output_tokens, 3300)  # 1500 (last msg_a) + 1800
        self.assertEqual(usage.metered_tokens, 22 + 8000 + 110000 + 3300)

    def test_empty_or_garbage_is_not_found(self) -> None:
        for text in ("", "   ", "not json at all", "plain stdout\nno usage here"):
            usage = parse_claude_json(text)
            self.assertFalse(usage.found, text)
            self.assertEqual(usage.total_tokens, 0)

    def test_metered_tokens_includes_cache(self) -> None:
        usage = parse_claude_json(
            json.dumps(
                {
                    "type": "result",
                    "num_turns": 6,
                    "usage": {
                        "input_tokens": 44,
                        "cache_creation_input_tokens": 11332,
                        "cache_read_input_tokens": 118080,
                        "output_tokens": 3039,
                    },
                }
            )
        )
        # The backstop must count cache reads, which dominate an agentic loop;
        # input+output alone would undercount this real call ~43x.
        self.assertEqual(usage.total_tokens, 3083)
        self.assertEqual(usage.metered_tokens, 132495)

    def test_metered_tokens_trip_max_tokens_budget(self) -> None:
        text = json.dumps(
            {
                "type": "result",
                "num_turns": 1,
                "usage": {
                    "input_tokens": 44,
                    "cache_read_input_tokens": 5000,
                    "output_tokens": 600,
                },
            }
        )
        usage = parse_claude_json(text)
        # input+output (644) stays under the cap; the full metered total (5644)
        # trips it. The backstop only works because cache reads are counted.
        tracker = BudgetTracker(scope="run", spec=BudgetSpec(max_tokens=1000))
        tracker.add_tokens(usage.metered_tokens)
        with self.assertRaises(BudgetExhausted):
            tracker.check()


if __name__ == "__main__":
    unittest.main()
