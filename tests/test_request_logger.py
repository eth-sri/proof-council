from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathagents.request_logger import RequestLogger  # noqa: E402


class RequestLoggerTests(unittest.TestCase):
    def test_log_response_recovers_malformed_request_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logger = RequestLogger()
            logger.log_dir = td
            log_path = Path(td) / "uninitialized" / "ts_idx0.json"
            log_path.parent.mkdir(parents=True)
            log_path.write_text('{"request": ', encoding="utf-8")

            logger.log_response("ts", 0, response={"exception": "boom"})

            data = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(data["response"], {"exception": "boom"})
            self.assertEqual(
                data["request_log_recovery"]["type"],
                "JSONDecodeError",
            )

    def test_log_request_stringifies_unusual_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logger = RequestLogger()
            logger.log_dir = td

            logger.log_request("ts", 0, request={"odd": object()})

            log_path = Path(td) / "uninitialized" / "ts_idx0.json"
            data = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertIn("object object", data["request"]["odd"])


if __name__ == "__main__":
    unittest.main()
