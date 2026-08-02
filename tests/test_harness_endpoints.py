from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.dev import create_app  # noqa: E402
from app.dev_data import load_pending_human_tasks  # noqa: E402

SHARE_FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "chatgpt_share_sample.json").read_text(encoding="utf-8")
)
SHARE_URL = "https://chatgpt.com/share/6a6ef011-902c-83eb-b009-7c85c80b2324"


def _clean_fixture() -> dict:
    # The tracked fixture deliberately contains sandbox artifacts (they must
    # warn); the clean-path test needs a variant without them.
    data = json.loads(json.dumps(SHARE_FIXTURE))
    for node in data["linear_conversation"]:
        msg = node.get("message") or {}
        if (msg.get("author") or {}).get("role") == "assistant":
            msg.get("metadata", {}).pop("attachments", None)
            parts = (msg.get("content") or {}).get("parts") or []
            msg["content"]["parts"] = [
                p.split("\nDownload:")[0] if isinstance(p, str) else p for p in parts
            ]
    return data


def _make_run(outputs_root: Path, *, expected_slugs: list[str]) -> tuple[str, str]:
    run_dir = outputs_root / "run1"
    inbox = run_dir / "human_inbox"
    packet = inbox / "echo__abc123def456.packet"
    packet.mkdir(parents=True)
    (packet / "instruction.txt").write_text("DO THE THING", encoding="utf-8")
    zip_path = inbox / "echo__abc123def456.packet.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(packet / "instruction.txt", arcname="instruction.txt")
    stem = "echo__abc123def456"
    response_path = inbox / f"{stem}.response.json"
    task = {
        "agent": "echo",
        "run_id": "run1",
        "type": "browser_call",
        "prompt": "DO THE THING",
        "response_path": str(response_path),
        "packet_dir": str(packet),
        "packet_zip": str(zip_path),
        "harness_copy_dir": None,
        "files": [{"name": "instruction.txt", "bytes": 12}],
        "browser": {
            "model": "fake-browser-model",
            "display_model": "Fake Browser",
            "service": "chatgpt",
            "settings_hint": "pick Fake",
            "chat_url": "https://chatgpt.com/",
            "expected_model_slugs": expected_slugs,
        },
        "expected": {},
        "output_fields": {},
    }
    (inbox / f"{stem}.task.json").write_text(
        json.dumps(task, ensure_ascii=False), encoding="utf-8"
    )
    events = [
        {"kind": "run.start", "payload": {}},
        {
            "kind": "human.waiting",
            "agent": "echo",
            "payload": {
                "type": "browser_call",
                "task_path": str(inbox / f"{stem}.task.json"),
                "response_path": str(response_path),
                "packet_dir": str(packet),
            },
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return stem, f"{stem}.response.json"


class HarnessEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.outputs_root = Path(self._tmp.name)
        self.stem, self.response_filename = _make_run(
            self.outputs_root, expected_slugs=["gpt-5-6-sol"]
        )
        self.run_dir = self.outputs_root / "run1"
        app = create_app((self.outputs_root,))
        app.testing = True
        self.client = app.test_client()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pending_task_passthrough(self) -> None:
        tasks = load_pending_human_tasks(self.run_dir)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["type"], "browser_call")
        self.assertEqual(task["browser"]["display_model"], "Fake Browser")
        self.assertEqual(task["files"][0]["name"], "instruction.txt")

    def test_run_detail_renders_harness_card(self) -> None:
        resp = self.client.get("/run/run1")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Browser call", html)
        self.assertIn("Fetch &amp; Continue", html)
        self.assertIn("harness/fetch-share", html)
        self.assertIn("harness/manual", html)

    def test_fetch_share_clean_writes_response(self) -> None:
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=_clean_fixture(),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={
                    "response_filename": self.response_filename,
                    "share_url": SHARE_URL,
                    "operator_comments": "note",
                },
            )
        self.assertEqual(resp.status_code, 302)
        payload = json.loads(
            (self.run_dir / "human_inbox" / self.response_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["transport"], "share_link")
        self.assertIn("<answer>The lemma holds.</answer>", payload["assistant_text"])
        self.assertEqual(payload["model_slug"], "gpt-5-6-sol-pro")
        self.assertEqual(payload["operator_comments"], "note")

    def test_fetch_share_with_warnings_needs_confirmation(self) -> None:
        # The fixture's assistant generated sandbox files -> warning page.
        stem2, filename2 = self.stem, self.response_filename
        task_path = self.run_dir / "human_inbox" / f"{stem2}.task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["browser"]["expected_model_slugs"] = ["some-other-model"]
        task_path.write_text(json.dumps(task), encoding="utf-8")

        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": filename2, "share_url": SHARE_URL},
            )
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("Use anyway", html)
            self.assertFalse(
                (self.run_dir / "human_inbox" / filename2).exists()
            )
            resp2 = self.client.post(
                "/run/run1/harness/fetch-share",
                data={
                    "response_filename": filename2,
                    "share_url": SHARE_URL,
                    "confirmed": "1",
                },
            )
        self.assertEqual(resp2.status_code, 302)
        self.assertTrue((self.run_dir / "human_inbox" / filename2).exists())

    def test_fetch_share_error_is_422(self) -> None:
        from proofstack.harness.chatgpt_share import ShareFetchError

        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            side_effect=ShareFetchError("share link not found"),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={
                    "response_filename": self.response_filename,
                    "share_url": SHARE_URL,
                },
            )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("share link not found", resp.get_data(as_text=True))

    def test_manual_submit_with_upload(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "pasted answer",
                "operator_comments": "",
                "files": (io.BytesIO(b"TEX BODY"), "answer.tex"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 302)
        payload = json.loads(
            (self.run_dir / "human_inbox" / self.response_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["transport"], "manual")
        self.assertEqual(
            payload["uploaded_files"]["answer.tex"],
            f"human_inbox/{self.stem}.uploads/answer.tex",
        )
        saved = (
            self.run_dir / "human_inbox" / f"{self.stem}.uploads" / "answer.tex"
        ).read_text(encoding="utf-8")
        self.assertEqual(saved, "TEX BODY")

    def test_manual_submit_empty_is_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "  "},
        )
        self.assertEqual(resp.status_code, 400)

    def test_manual_upload_bad_filename_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "x",
                "files": (io.BytesIO(b"evil"), "we ird$.txt"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_non_harness_response_filename_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": "nope.response.json", "assistant_text": "x"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_blob_download_param(self) -> None:
        resp = self.client.get(
            "/run/run1/blob",
            query_string={
                "ref": f"human_inbox/{self.stem}.packet.zip",
                "download": "1",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))


if __name__ == "__main__":
    unittest.main()
