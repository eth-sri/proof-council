"""Drive the dashboard headlessly and screenshot key UI states for review.

Usage: uv run python scripts/ui_review.py [--port 5005] [--probe]

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
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fail on HTTP errors or uncaught browser-page exceptions.",
    )
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
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1400})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def visit(path: str, *, settle_ms: int = 300) -> None:
                response = page.goto(f"{base}{path}", wait_until="networkidle")
                if args.probe and (response is None or not response.ok):
                    status = response.status if response is not None else "no response"
                    raise RuntimeError(f"{path} returned {status}")
                page.wait_for_timeout(settle_ms)

            visit("/run-agent", settle_ms=500)
            shot(page, "run_agent_full")

            visit("/runs")
            shot(page, "runs_list")

            visit("/presets")
            shot(page, "presets_list")
            editor_links = page.locator("a.agent-link")
            editor_count = editor_links.count()
            if args.probe and editor_count == 0:
                raise RuntimeError("/presets did not expose an agent editor link")
            if editor_count:
                editor_link = editor_links.first
                href = editor_link.get_attribute("href")
                if args.probe and not href:
                    raise RuntimeError("agent editor link has no href")
                if href:
                    visit(href, settle_ms=700)
                    shot(page, "preset_editor")

            if args.probe and page_errors:
                raise RuntimeError("browser page error(s): " + "; ".join(page_errors))
        finally:
            browser.close()

    print("screenshots:")
    for s in shots:
        print(f"  {s}")
    if args.probe:
        print("probe: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
