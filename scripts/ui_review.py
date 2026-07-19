"""Drive the dashboard headlessly and screenshot key UI states for review.

Usage: uv run python scripts/ui_review.py [--port 5002] [--probe]

Writes PNGs to outputs/ui_review/ (gitignored). Read-only by default: it
loads pages and switches presets but does not save settings or launch runs.
--probe additionally clicks "Probe now" (hits the real usage endpoint; ~1s,
no token cost, refreshes the shared probe cache timestamp).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "ui_review"

PACING_PRESETS = ["claude_subscription_min", "codex_subscription_min", "author_critic"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--probe", action="store_true", help="also click 'Probe now'")
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
        page.wait_for_timeout(500)  # panel renders after the status fetch
        shot(page, "run_agent_full")

        select = page.locator("#run-agent-preset")
        for preset in PACING_PRESETS:
            if select.locator(f'option[value="{preset}"]').count() == 0:
                print(f"note: preset {preset!r} not found; skipped", file=sys.stderr)
                continue
            select.select_option(preset)
            page.wait_for_timeout(300)
            shot(page, f"pacing_{preset}", "#pacing-panel")

        if args.probe:
            select.select_option("claude_subscription_min")
            page.wait_for_timeout(300)
            page.locator("#pacing-probe-button").click()
            page.wait_for_timeout(4000)
            shot(page, "pacing_after_probe", "#pacing-panel")

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
