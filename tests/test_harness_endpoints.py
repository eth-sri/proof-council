from __future__ import annotations

import io
import json
import sys
import tempfile
import time
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
        "instruction_file": f"instruction_{stem}.txt",
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
        # The card must show the real tokenized instruction filename, not a
        # hardcoded instruction.txt.
        self.assertIn(f"instruction_{self.stem}.txt", html)

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

    def test_fetch_share_with_warnings_stages_then_confirms(self) -> None:
        # The fixture's assistant generated sandbox files -> warning page.
        stem2, filename2 = self.stem, self.response_filename
        task_path = self.run_dir / "human_inbox" / f"{stem2}.task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["browser"]["expected_model_slugs"] = ["some-other-model"]
        task_path.write_text(json.dumps(task), encoding="utf-8")

        from proofstack.harness.chatgpt_share import ShareFetchError

        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), mock.patch(
            "proofstack.harness.chatgpt_share.resolve_shared_file",
            side_effect=ShareFetchError("resolver down"),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": filename2, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Use fetched answer", html)
        self.assertIn("could not auto-download", html)
        self.assertFalse((self.run_dir / "human_inbox" / filename2).exists())
        staged_path = self.run_dir / "human_inbox" / f"{stem2}.staged.json"
        staged = json.loads(staged_path.read_text(encoding="utf-8"))

        # Confirming applies the staged version — no re-fetch happens.
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            side_effect=AssertionError("confirm must not re-fetch"),
        ), mock.patch(
            "proofstack.harness.chatgpt_share.resolve_shared_file",
            side_effect=AssertionError("confirm must not re-resolve files"),
        ):
            resp2 = self.client.post(
                "/run/run1/harness/confirm",
                data={
                    "response_filename": filename2,
                    "digest": staged["digest"],
                    "operator_comments": "steer",
                    "files": (io.BytesIO(b"DOWNLOADED BODY"), "answer.tex"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(resp2.status_code, 302)
        payload = json.loads(
            (self.run_dir / "human_inbox" / filename2).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["transport"], "share_link")
        self.assertEqual(payload["operator_comments"], "steer")
        rel = payload["uploaded_files"]["answer.tex"]
        self.assertTrue(rel.startswith(f"human_inbox/{stem2}.uploads-"))
        self.assertTrue(rel.endswith("/answer.tex"))
        self.assertFalse(staged_path.exists())

    def test_confirm_with_stale_digest_is_409(self) -> None:
        from proofstack.harness.chatgpt_share import ShareFetchError

        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), mock.patch(
            "proofstack.harness.chatgpt_share.resolve_shared_file",
            side_effect=ShareFetchError("resolver down"),
        ):
            self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        resp = self.client.post(
            "/run/run1/harness/confirm",
            data={"response_filename": self.response_filename, "digest": "not-the-digest"},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertFalse(
            (self.run_dir / "human_inbox" / self.response_filename).exists()
        )

    def test_double_submit_is_409(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "one"},
        )
        self.assertEqual(resp.status_code, 302)
        resp2 = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "two"},
        )
        self.assertEqual(resp2.status_code, 409)
        payload = json.loads(
            (self.run_dir / "human_inbox" / self.response_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["assistant_text"], "one")

    def test_inflight_reservation_blocks_submission(self) -> None:
        # A fresh empty response file is another request's in-flight
        # fallback write.
        target = self.run_dir / "human_inbox" / self.response_filename
        target.write_text("", encoding="utf-8")
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "x"},
        )
        self.assertEqual(resp.status_code, 409)

    def test_aged_empty_reservation_is_recoverable(self) -> None:
        # A crashed publication must not brick the task: an empty file
        # older than the grace period may be replaced.
        import os as _os

        target = self.run_dir / "human_inbox" / self.response_filename
        target.write_text("", encoding="utf-8")
        old = time.time() - 3600
        _os.utime(target, (old, old))
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "recovered"},
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8"))["assistant_text"],
            "recovered",
        )

    def test_confirm_with_mutated_staged_file_is_409(self) -> None:
        stem2, filename2 = self.stem, self.response_filename
        task_path = self.run_dir / "human_inbox" / f"{stem2}.task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["browser"]["expected_model_slugs"] = ["some-other-model"]
        task_path.write_text(json.dumps(task), encoding="utf-8")

        p1, p2 = self._mock_download(b"ORIGINAL BYTES")
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), p1, p2:
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": filename2, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 200)
        staged = json.loads(
            (self.run_dir / "human_inbox" / f"{stem2}.staged.json").read_text(
                encoding="utf-8"
            )
        )
        # Mutate the staged bytes (what a concurrent fetch could do before
        # nonce dirs; the digest must catch any such drift).
        rel = next(iter(staged["file_hashes"]))
        (self.run_dir / rel).write_bytes(b"SWAPPED BYTES")
        resp2 = self.client.post(
            "/run/run1/harness/confirm",
            data={"response_filename": filename2, "digest": staged["digest"]},
        )
        self.assertEqual(resp2.status_code, 409)
        self.assertFalse((self.run_dir / "human_inbox" / filename2).exists())

    def test_generic_human_endpoint_refuses_browser_tasks(self) -> None:
        resp = self.client.post(
            "/run/run1/human",
            data={"response_filename": self.response_filename, "f_answer": "sneaky"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(
            (self.run_dir / "human_inbox" / self.response_filename).exists()
        )

    def test_losing_submission_leaves_no_upload_directory(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "winner",
                "files": (io.BytesIO(b"WINNER BYTES"), "answer.tex"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 302)
        resp2 = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "loser",
                "files": (io.BytesIO(b"LOSER BYTES"), "answer.tex"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp2.status_code, 409)
        dirs = list((self.run_dir / "human_inbox").glob(f"{self.stem}.uploads-*"))
        self.assertEqual(len(dirs), 1)
        self.assertEqual((dirs[0] / "answer.tex").read_bytes(), b"WINNER BYTES")

    def test_failed_fetch_leaves_no_staging_directory(self) -> None:
        from proofstack.harness.chatgpt_share import ShareFetchError

        p1, p2 = self._mock_download(b"DOWNLOADED")
        # Downloads succeed, but validation then hard-fails (empty answer).
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), p1, p2, mock.patch(
            "proofstack.harness.chatgpt_share.validate_result",
            side_effect=ShareFetchError("no answer"),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(
            list((self.run_dir / "human_inbox").glob(f"{self.stem}.staged-*")), []
        )

    def test_ambiguous_share_offers_no_transfer(self) -> None:
        inbox = self.run_dir / "human_inbox"
        task_path = inbox / f"{self.stem}.task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["task_token"] = "tokA"
        task["instruction_file"] = "instruction_tokA.txt"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        other = {
            "agent": "critic",
            "run_id": "run1",
            "type": "browser_call",
            "task_token": "tokB",
            "instruction_file": "instruction_tokB.txt",
            "response_path": str(inbox / "critic__b.response.json"),
            "browser": {"display_model": "Fake B"},
            "expected": {},
        }
        (inbox / "critic__b.task.json").write_text(json.dumps(other), encoding="utf-8")

        fixture = _clean_fixture()
        for node in fixture["linear_conversation"]:
            msg = node.get("message") or {}
            meta = msg.get("metadata") or {}
            if (msg.get("author") or {}).get("role") == "user" and not meta.get(
                "is_visually_hidden_from_conversation"
            ):
                msg["content"]["parts"] = [
                    "[ProofCouncil task tokB]\n[ProofCouncil task tokC]\ndo it"
                ]
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=fixture,
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("ambiguous", html)
        self.assertNotIn("belong to a different task", html)

    def test_broken_response_may_be_replaced(self) -> None:
        target = self.run_dir / "human_inbox" / self.response_filename
        target.write_text("{not json", encoding="utf-8")
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "fixed"},
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8"))["assistant_text"], "fixed"
        )

    def test_share_reuse_across_tasks_warns(self) -> None:
        other = self.run_dir / "human_inbox" / "other__000.response.json"
        other.write_text(
            json.dumps({"share_url": SHARE_URL, "assistant_text": "x"}),
            encoding="utf-8",
        )
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=_clean_fixture(),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("already submitted for another task", resp.get_data(as_text=True))
        self.assertFalse(
            (self.run_dir / "human_inbox" / self.response_filename).exists()
        )

    def test_unexpected_extraction_error_renders_fallback_not_500(self) -> None:
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            side_effect=KeyError("schema drift"),
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("manual paste fallback", resp.get_data(as_text=True))

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
        rel = payload["uploaded_files"]["answer.tex"]
        self.assertTrue(rel.startswith(f"human_inbox/{self.stem}.uploads-"))
        self.assertTrue(rel.endswith("/answer.tex"))
        self.assertEqual((self.run_dir / rel).read_text(encoding="utf-8"), "TEX BODY")

    def test_manual_submit_empty_is_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={"response_filename": self.response_filename, "assistant_text": "  "},
        )
        self.assertEqual(resp.status_code, 400)

    def test_manual_upload_binary_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "x",
                "files": (io.BytesIO(b"\x89PNG\x00\xff"), "plot.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(
            (self.run_dir / "human_inbox" / self.response_filename).exists()
        )

    def test_manual_upload_fence_conflict_rejected(self) -> None:
        resp = self.client.post(
            "/run/run1/harness/manual",
            data={
                "response_filename": self.response_filename,
                "assistant_text": "x",
                "files": (io.BytesIO(b"ok\n```\nfence"), "notes.md"),
            },
            content_type="multipart/form-data",
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

    def _mock_download(self, body: bytes):
        def fake_resolve(sid, message_id, sandbox_path, **kw):
            return {
                "download_url": "https://sub.oaiusercontent.com/signed",
                "file_name": sandbox_path.rsplit("/", 1)[-1],
                "file_size_bytes": len(body),
            }

        def fake_download(info, dest_dir, **kw):
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / info["file_name"]
            dest.write_bytes(body)
            return dest

        return mock.patch(
            "proofstack.harness.chatgpt_share.resolve_shared_file", fake_resolve
        ), mock.patch(
            "proofstack.harness.chatgpt_share.download_shared_file", fake_download
        )

    def test_generated_text_file_is_auto_downloaded_and_merged(self) -> None:
        p1, p2 = self._mock_download(b"GENERATED TEXT")
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), p1, p2:
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        # Everything auto-downloaded -> no sandbox warning -> direct write.
        self.assertEqual(resp.status_code, 302)
        payload = json.loads(
            (self.run_dir / "human_inbox" / self.response_filename).read_text(
                encoding="utf-8"
            )
        )
        rel = payload["uploaded_files"]["report.pdf"]
        self.assertTrue(rel.startswith(f"human_inbox/{self.stem}.staged-"))
        self.assertTrue(rel.endswith("/report.pdf"))
        self.assertEqual((self.run_dir / rel).read_bytes(), b"GENERATED TEXT")

    def test_generated_binary_file_is_stored_not_merged(self) -> None:
        p1, p2 = self._mock_download(b"\x89PNG\x00\xffbinary")
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=SHARE_FIXTURE,
        ), p1, p2:
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 302)
        payload = json.loads(
            (self.run_dir / "human_inbox" / self.response_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertIsNone(payload["uploaded_files"])
        self.assertEqual(payload["stored_files"][0]["name"], "report.pdf")
        self.assertFalse(payload["stored_files"][0]["merged"])

    def test_wrong_task_share_offers_transfer(self) -> None:
        inbox = self.run_dir / "human_inbox"
        task_path = inbox / f"{self.stem}.task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["task_token"] = "tokA"
        task["instruction_file"] = "instruction_tokA.txt"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        other = {
            "agent": "critic",
            "run_id": "run1",
            "type": "browser_call",
            "task_token": "tokB",
            "instruction_file": "instruction_tokB.txt",
            "response_path": str(inbox / "critic__b.response.json"),
            "browser": {"display_model": "Fake B"},
            "expected": {},
        }
        (inbox / "critic__b.task.json").write_text(json.dumps(other), encoding="utf-8")

        fixture = _clean_fixture()
        for node in fixture["linear_conversation"]:
            msg = node.get("message") or {}
            meta = msg.get("metadata") or {}
            if (msg.get("author") or {}).get("role") == "user" and not meta.get(
                "is_visually_hidden_from_conversation"
            ):
                msg.setdefault("metadata", {})["attachments"] = [
                    {"name": "instruction_tokB.txt"}
                ]
        with mock.patch(
            "proofstack.harness.chatgpt_share.fetch_share",
            return_value=fixture,
        ):
            resp = self.client.post(
                "/run/run1/harness/fetch-share",
                data={"response_filename": self.response_filename, "share_url": SHARE_URL},
            )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("belong to a different task", html)
        self.assertIn('value="critic__b.response.json"', html)
        self.assertIn("most recent ProofCouncil task", html)

    def test_waiting_after_timeout_resurfaces_task(self) -> None:
        # waiting -> timeout -> (resume) waiting again: the later wait must
        # clear the earlier resolution or the card stays hidden forever.
        events_path = self.run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        waiting = next(e for e in events if e["kind"] == "human.waiting")
        events.append(
            {"kind": "human.timeout", "payload": {"response_path": waiting["payload"]["response_path"]}}
        )
        events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        self.assertEqual(load_pending_human_tasks(self.run_dir), [])
        events.append(waiting)
        events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        self.assertEqual(len(load_pending_human_tasks(self.run_dir)), 1)

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
