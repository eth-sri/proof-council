from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UIReviewContractTests(unittest.TestCase):
    def test_wrapper_and_python_use_dashboard_default_and_support_probe(self) -> None:
        wrapper = (ROOT / "scripts" / "ui_review.sh").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "ui_review.py").read_text(encoding="utf-8")

        self.assertIn("PORT=5005", wrapper)
        self.assertIn('if [[ "${1:-}" =~ ^[0-9]+$ ]]', wrapper)
        self.assertIn('parser.add_argument("--port", type=int, default=5005)', script)
        self.assertIn('"--probe"', script)
        self.assertIn("editor_count == 0", script)

    def test_uv_helpers_support_the_standard_user_install_location(self) -> None:
        for name in ("run_dashboard.sh", "ui_review.sh", "validate_preset.sh"):
            wrapper = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('export PATH="$HOME/.local/bin:$PATH"', wrapper, name)

    def test_dev_dependencies_cover_tests_and_browser_review(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)

        self.assertIn("playwright>=1.61.0", project["dependency-groups"]["dev"])
        self.assertIn("pytest>=8.3.0", project["dependency-groups"]["dev"])


if __name__ == "__main__":
    unittest.main()
