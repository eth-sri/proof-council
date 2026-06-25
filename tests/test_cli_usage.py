import json
import unittest

from proofstack.budget import BudgetExhausted, BudgetSpec, BudgetTracker
from proofstack.cli_usage import parse_claude_json


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

    def test_empty_or_garbage_is_not_found(self) -> None:
        for text in ("", "   ", "not json at all", "plain stdout\nno usage here"):
            usage = parse_claude_json(text)
            self.assertFalse(usage.found, text)
            self.assertEqual(usage.total_tokens, 0)

    def test_parsed_tokens_trip_max_tokens_budget(self) -> None:
        text = json.dumps(
            {"type": "result", "num_turns": 1, "usage": {"input_tokens": 600, "output_tokens": 600}}
        )
        usage = parse_claude_json(text)
        tracker = BudgetTracker(scope="run", spec=BudgetSpec(max_tokens=1000))
        tracker.add_tokens(usage.input_tokens + usage.output_tokens)
        with self.assertRaises(BudgetExhausted):
            tracker.check()


if __name__ == "__main__":
    unittest.main()
