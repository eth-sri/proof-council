"""Drive the dashboard headlessly and screenshot key UI states for review.

Usage: uv run python scripts/ui_review.py [--port 5002]

Writes PNGs to outputs/ui_review/ (gitignored). Read-only: it loads pages but
does not save settings or launch runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "ui_review"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shots: list[str] = []

    def shot(page, name: str, selector: str | None = None) -> None:
        path = OUT_DIR / f"{name}.png"
        target = page.locator(selector) if selector else page
        target.screenshot(path=str(path))
        shots.append(str(path.relative_to(REPO_ROOT)))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1400})
        page.goto(f"{base}/run-agent", wait_until="networkidle")
        page.wait_for_timeout(500)
        shot(page, "run_agent_full")

        page.goto(f"{base}/runs", wait_until="networkidle")
        page.wait_for_timeout(300)
        shot(page, "runs_list")
        browser.close()

    print("screenshots:")
    for s in shots:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
