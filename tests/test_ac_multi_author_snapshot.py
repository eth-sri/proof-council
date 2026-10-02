from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.agents.ac.sandbox_carry import SandboxCarryState, SandboxSnapshot
from proofstack.context import RunContext


def _author(root: str, **delegation) -> MultiAuthor:
    ctx = RunContext.create(
        run_id="snapshot-test", root_workdir=root, flat=True,
        component_configs={"Author": {"delegation": {
            "enabled": True, "sandbox_carry_over": True,
            "container_keepalive_s": 0, **delegation,
        }}},
    )
    author = MultiAuthor(ctx, name="Author")
    author._begin_turn(Author.Inputs(problem="P", round=1, n_rounds=2))
    author._sandbox_carry.bind_container({"type": "auto", "file_ids": ["file-original"]})
    return author


def _client(create, delete, names=("answer.tex",), *, contents=None, empty_size=0):
    contents = contents if contents is not None else dict.fromkeys(names, b"tex")
    return SimpleNamespace(
        files=SimpleNamespace(create=create, delete=delete),
        containers=SimpleNamespace(files=SimpleNamespace(
            list=lambda cid: [SimpleNamespace(id=name, path=f"/mnt/data/{name}",
                                             bytes=len(contents[name]) if contents[name] else empty_size)
                              for name in contents],
            content=SimpleNamespace(retrieve=lambda fid, **kwargs: SimpleNamespace(read=lambda: contents[fid])),
        )),
    )


@pytest.mark.parametrize("empty_size", [0, None, 123])
@pytest.mark.parametrize("empty_first", [True, False])
def test_empty_file_does_not_abort_snapshot(tmp_path, empty_size, empty_first):
    async def scenario():
        author = _author(str(tmp_path))
        descriptor = {"type": "auto", "file_ids": ["file-original"]}
        author._sandbox_carry.bind_container(descriptor)
        contents = {"answer.tex": b"Current proof", "research_notes.tex": b" \n", "references.bib": b""}
        if empty_first:
            contents = {"references.bib": b"", **contents}
        created, deleted = [], []

        def create(*, file, purpose):
            if not file[1]:
                raise ValueError("File is empty.")
            created.append(file)
            return SimpleNamespace(id=f"file-copy{len(created)}")

        client = _client(create, deleted.append, contents=contents, empty_size=empty_size)
        with patch.object(author, "_openai", lambda **kwargs: client), patch("proofstack.agents.ac.multi_author._SNAPSHOT_ATTEMPTS", 1):
            result = await author._snapshot_sandbox("old", deadline=time.monotonic() + 1)
            assert result["error"] is None
            assert created == [("answer.tex", b"Current proof"), ("research_notes.tex", b" \n")]
            assert result["empty_files"] == ["references.bib"]
            assert result["files"] == [("answer.tex", "/mnt/data/file-copy1-answer.tex"),
                                       ("research_notes.tex", "/mnt/data/file-copy2-research_notes.tex")]
            assert descriptor["file_ids"] == ["file-original", "file-copy1", "file-copy2"]
            note = author._render_sandbox_note(result)
            assert "/mnt/data/file-copy1-answer.tex" in note
            assert "/mnt/data/references.bib" in note
            assert "zero-byte" in note
            assert "older attachments" in note
            events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
            assert events[-1]["payload"]["empty_files"] == ["references.bib"]
            await author._close_sandbox_carry()
        assert deleted == ["file-copy1", "file-copy2"]
    asyncio.run(scenario())


def test_empty_snapshot_replaces_stale_copy_without_uploading_placeholder(tmp_path):
    async def scenario():
        author = _author(str(tmp_path))
        descriptor = {"type": "auto", "file_ids": ["file-original"]}
        author._sandbox_carry.bind_container(descriptor)
        contents = {"references.bib": b"Old bibliography"}
        created, deleted = [], []

        def create(*, file, purpose):
            assert file[1]
            created.append(file)
            return SimpleNamespace(id=f"file-copy{len(created)}")

        client = _client(create, deleted.append, contents=contents)
        with patch.object(author, "_openai", lambda **kwargs: client), patch("proofstack.agents.ac.multi_author._SNAPSHOT_ATTEMPTS", 1):
            assert (await author._snapshot_sandbox("old"))["error"] is None
            assert descriptor["file_ids"] == ["file-original", "file-copy1"]
            contents["references.bib"] = b""
            result = await author._snapshot_sandbox("cleared")
            assert result["error"] is None
            assert result["files"] == []
            assert result["empty_files"] == ["references.bib"]
            assert len(created) == 1
            assert descriptor["file_ids"] == ["file-original"]
            assert not author._carried
            note = author._render_sandbox_note(result)
            assert "nothing was carried over" not in note
            assert "zero-byte" in note and "/mnt/data/references.bib" in note
            contents["references.bib"] = b"New bibliography"
            result = await author._snapshot_sandbox("rewritten")
            assert result["error"] is None
            assert result["empty_files"] == []
            assert descriptor["file_ids"] == ["file-original", "file-copy2"]
            assert author._carried == {"references.bib": "file-copy2"}
            await author._close_sandbox_carry()
        assert created == [("references.bib", b"Old bibliography"), ("references.bib", b"New bibliography")]
        assert deleted == ["file-copy1", "file-copy2"]
    asyncio.run(scenario())


def test_failed_snapshot_does_not_commit_empty_file(tmp_path):
    async def scenario():
        author = _author(str(tmp_path))
        descriptor = {"type": "auto", "file_ids": ["file-original", "file-prior"]}
        author._sandbox_carry.bind_container(descriptor)
        author._carried["references.bib"] = "file-prior"
        contents = {"references.bib": b"", "answer.tex": b"Current proof"}

        def fail(**kwargs):
            raise RuntimeError("Upload failed")

        client = _client(fail, None, contents=contents)
        with patch.object(author, "_openai", lambda **kwargs: client), patch("proofstack.agents.ac.multi_author._SNAPSHOT_ATTEMPTS", 1):
            result = await author._snapshot_sandbox("old", deadline=time.monotonic() + 1)
        assert "Upload failed" in result["error"]
        assert not result.get("empty_files")
        assert descriptor["file_ids"] == ["file-original", "file-prior"]
        assert author._carried == {"references.bib": "file-prior"}
    asyncio.run(scenario())


@pytest.mark.parametrize("closed", [False, True])
def test_abandoned_empty_snapshot_cannot_remove_current_attachment(closed):
    state = SandboxCarryState()
    descriptor = {"type": "auto", "file_ids": ["file-current"]}
    state.bind_container(descriptor)
    state.carried["references.bib"] = "file-current"
    snapshot = SandboxSnapshot(empty_files=["references.bib"])
    if closed:
        state.close()
    else:
        state.abandon(snapshot)
    assert not state.commit(snapshot)
    assert descriptor["file_ids"] == ["file-current"]
    assert state.carried == {"references.bib": "file-current"}


def test_slow_snapshot_keeps_completed_reports_and_wave_deadline():
    async def scenario(root):
        author = _author(root, job_timeout_s=0.04)
        cancelled = asyncio.Event()

        async def seat(self, inp):
            return self.Outputs(report="Completed proof survives")

        async def snapshot(self, container_id, **kwargs):
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()

        started = time.monotonic()
        with patch.object(SubAuthorSeat, "run", seat), patch.object(MultiAuthor, "_snapshot_sandbox", snapshot):
            result = await asyncio.to_thread(
                author._delegate, tasks=[{"role": "prover", "task": "Prove P"}],
                messages=[{"type": "code_interpreter_call", "container_id": "old"}],
            )
        assert time.monotonic() - started < 1
        assert "Completed proof survives" in result
        assert "could not copy" in result
        assert cancelled.is_set()
        assert author._delegation_log[0]["report"] == "Completed proof survives"
        assert (author.workdir / "subagents" / "wave1-prover1.md").exists()

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_cancelled_upload_cannot_attach_to_next_turn():
    async def scenario(root):
        author = _author(root)
        old_state = author._sandbox_carry
        entered, release, deleted = threading.Event(), threading.Event(), threading.Event()

        def create(**kwargs):
            entered.set()
            assert release.wait(2)
            return SimpleNamespace(id="file-old-turn")

        def delete(file_id):
            assert file_id == "file-old-turn"
            deleted.set()

        with patch.object(author, "_openai", lambda **kwargs: _client(create, delete)):
            snapshot = asyncio.create_task(author._snapshot_sandbox("old", deadline=time.monotonic() + 1))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                snapshot.cancel()
                await asyncio.gather(snapshot, return_exceptions=True)
                await author._close_sandbox_carry(old_state)
                author._begin_turn(Author.Inputs(problem="P", round=2, n_rounds=2))
                descriptor = {"type": "auto", "file_ids": ["file-next-turn"]}
                author._sandbox_carry.bind_container(descriptor)
                release.set()
                assert await asyncio.to_thread(deleted.wait, 1)
                assert descriptor["file_ids"] == ["file-next-turn"]
                assert not author._carried
                assert not author._carry_ids_to_delete
            finally:
                release.set()
                await asyncio.gather(snapshot, return_exceptions=True)

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_timed_out_upload_cannot_replace_a_newer_snapshot():
    async def scenario(root):
        author = _author(root)
        descriptor = {"type": "auto", "file_ids": ["file-original"]}
        author._sandbox_carry.bind_container(descriptor)
        entered, release, deleted = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def create(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                assert release.wait(2)
                return SimpleNamespace(id="file-stale")
            return SimpleNamespace(id="file-current")

        def delete(file_id):
            if file_id == "file-stale":
                deleted.set()

        with patch.object(author, "_openai", lambda **kwargs: _client(create, delete)):
            stale = asyncio.create_task(author._snapshot_sandbox("old", deadline=time.monotonic() + 0.04))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                assert (await stale)["error"] is not None
                fresh = await author._snapshot_sandbox("new", deadline=time.monotonic() + 1)
                assert fresh["error"] is None
                release.set()
                assert await asyncio.to_thread(deleted.wait, 1)
                assert descriptor["file_ids"] == ["file-original", "file-current"]
                assert author._carried == {"answer.tex": "file-current"}
                await author._close_sandbox_carry()
            finally:
                release.set()
                await asyncio.gather(stale, return_exceptions=True)

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_partial_snapshot_failure_keeps_upload_registered_for_cleanup():
    async def scenario(root):
        author = _author(root)
        descriptor = {"type": "auto", "file_ids": ["file-original"]}
        author._sandbox_carry.bind_container(descriptor)
        calls, deleted = [], []

        def create(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("second upload failed")
            return SimpleNamespace(id="file-partial")

        client = _client(create, deleted.append, names=("answer.tex", "research_notes.tex"))
        with patch.object(author, "_openai", lambda **kwargs: client), patch("proofstack.agents.ac.multi_author._SNAPSHOT_ATTEMPTS", 1):
            result = await author._snapshot_sandbox("old", deadline=time.monotonic() + 1)
            assert "second upload failed" in result["error"]
            assert descriptor["file_ids"] == ["file-original"]
            assert author._carry_ids_to_delete == ["file-partial"]
            await author._close_sandbox_carry()
        assert deleted == ["file-partial"]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


@pytest.mark.parametrize("exception_type", [RuntimeError, asyncio.CancelledError])
def test_query_failure_cleans_captured_turn_uploads(exception_type):
    async def scenario(root):
        author = _author(root)
        author._carry_ids_to_delete.append("file-carry")
        deleted = []

        async def fail(*args, **kwargs):
            raise exception_type("provider stopped")

        with patch.object(Author, "_query", fail), patch.object(author, "_delete_platform_files", deleted.extend):
            with pytest.raises(exception_type):
                await author._query(None, [], None)
        assert deleted == ["file-carry"]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_successful_turn_cleans_only_after_parent_finishes_download():
    async def scenario(root):
        author = _author(root)
        order = []

        async def parent_run(self, inp):
            self._carry_ids_to_delete.append("file-carry")
            order.append("download")
            return self.Outputs(answer_tex="downloaded proof")

        def delete(ids):
            assert ids == ["file-carry"]
            order.append("cleanup")

        with patch.object(Author, "run", parent_run), patch.object(author, "_delete_platform_files", delete):
            result = await author.run(author._current_inp)
        assert result.answer_tex == "downloaded proof"
        assert order == ["download", "cleanup"]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_delete_failure_does_not_replace_downloaded_proof():
    async def scenario(root):
        author = _author(root)
        attempted = []

        async def parent_run(self, inp):
            self._carry_ids_to_delete.append("file-carry")
            return self.Outputs(answer_tex="downloaded proof")

        def delete(file_id):
            attempted.append(file_id)
            raise RuntimeError("Files API unavailable")

        client = _client(None, delete)
        with patch.object(Author, "run", parent_run), patch.object(author, "_openai", lambda **kwargs: client):
            result = await author.run(author._current_inp)
        assert result.answer_tex == "downloaded proof"
        assert attempted == ["file-carry"]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))


def test_slow_delete_does_not_hold_completed_turn_past_cleanup_limit():
    async def scenario(root):
        author = _author(root)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        async def parent_run(self, inp):
            self._carry_ids_to_delete.append("file-carry")
            return self.Outputs(answer_tex="downloaded proof")

        def delete(file_id):
            entered.set()
            try:
                assert release.wait(2)
            finally:
                finished.set()

        client = _client(None, delete)
        with patch.object(Author, "run", parent_run), patch.object(author, "_openai", lambda **kwargs: client), patch("proofstack.agents.ac.multi_author._CLEANUP_WAIT_S", 0.05):
            try:
                started = time.monotonic()
                result = await author.run(author._current_inp)
                assert time.monotonic() - started < 1
                assert entered.is_set()
                assert not finished.is_set()
                assert result.answer_tex == "downloaded proof"
            finally:
                release.set()
                assert await asyncio.to_thread(finished.wait, 1)

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(scenario(root))
