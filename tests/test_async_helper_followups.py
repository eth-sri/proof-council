import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.agents.ac import async_helpers, delegation_context, delegation_snapshot
from proofstack.agents.ac.async_helpers import HelperSession, helper_scope
from proofstack.agents.ac.delegation_context import DelegationContext
from proofstack.agents.ac.multi_author import Author, SubAuthorSeat
from test_async_helpers import lead_at, tool


class Files:
    def __init__(self, bodies):
        self.bodies = bodies
        self.reads = []
        self.content = SimpleNamespace(with_streaming_response=SimpleNamespace(retrieve=self.retrieve))

    def list(self, container, limit=100):
        return [SimpleNamespace(path="/mnt/data/" + name, id=name, bytes=len(body))
                for name, body in self.bodies.items()]

    def retrieve(self, *, container_id, file_id):
        self.reads.append(file_id)
        body = self.bodies[file_id]
        class Stream:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def iter_bytes(self, chunk_size):
                yield body
        return Stream()


def messages():
    return [{"type": "code_interpreter_call", "container_id": "current-container", "id": "ci-current"}]


def test_launch_uses_latest_saved_files_without_changing_turn_identity(tmp_path):
    async def scenario():
        files = Files({"answer.tex": b"new lemma", "research_notes.tex": b"new notes", "references.bib": b"new refs"})
        inputs = []
        async def work(seat, inp):
            inputs.append(inp)
            assert "new lemma" in seat.context_view.read("round/answer.tex")
            return seat.Outputs(report="checked")
        async with helper_scope():
            lead = lead_at(tmp_path, max_tasks_per_turn=2)
            with patch.object(lead, "_openai", return_value=SimpleNamespace(containers=SimpleNamespace(files=files))), patch.object(SubAuthorSeat, "run", work):
                first = (await tool(lead._launch_helpers, [{"role": "prover", "task": "Check current lemma"}], messages=messages()))["helpers"][0]
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                files.bodies["research_notes.tex"] = b"a newer note"
                second = (await tool(lead._launch_helpers, [{"role": "checker", "task": "Check revised notes"}], messages=messages()))["helpers"][0]
                await tool(lead._wait_helpers, ["helper2"], timeout_s=1)
                refused = await tool(lead._launch_helpers, [{"role": "prover", "task": "third"}], messages=messages())
                assert "allowance" in refused["error"]
            assert inputs[0].answer_tex == "new lemma" and inputs[1].research_notes_tex == "a newer note"
            assert first["input_hash"] == second["input_hash"]
            assert first["snapshot"]["sha256"] != second["snapshot"]["sha256"]
            assert all(f["source"] == "launch_sandbox" for f in first["snapshot"]["files"].values())
            assert lead._current_inp.answer_tex == "draft zero"
            assert lead._context.files["round/answer.tex"] == "draft zero"
            assert "current lemma" in (await tool(lead._read_context, first["task_path"]))["content"]
            lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=3), "")
            notice = str(lead._with_guide([{"role": "developer", "content": "lead"}]))
            assert "Check current lemma" in notice and first["snapshot"]["sha256"] in notice
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["missing", "listing", "size", "utf8", "no_container"])
def test_snapshot_failures_are_explicit_turn_start_fallbacks(tmp_path, failure):
    async def scenario():
        files = Files({"research_notes.tex": b"current notes", "references.bib": b"refs"})
        if failure == "listing":
            files.list = lambda *a, **kw: (_ for _ in ()).throw(OSError("provider secret must not be echoed"))
        elif failure == "size":
            files.bodies["answer.tex"] = b"x" * (delegation_context.MAX_FILE_BYTES + 1)
        elif failure == "utf8":
            files.bodies["answer.tex"] = b"\xff"
        seen = []
        async def work(seat, inp):
            seen.append(inp)
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(lead, "_openai", return_value=SimpleNamespace(containers=SimpleNamespace(files=files))), patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}],
                                  messages=[] if failure == "no_container" else messages()))["helpers"][0]
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
            assert row["snapshot"]["files"]["answer.tex"]["source"] == "turn_start"
            assert row["snapshot"]["warnings"] and "turn-start" in seen[0].context_notice
            assert seen[0].answer_tex == "draft zero"
            assert "provider secret" not in seen[0].context_notice
            if failure == "size":
                assert "answer.tex" not in files.reads
    asyncio.run(scenario())


def test_blind_launch_does_not_fetch_or_share_current_manuscript(tmp_path):
    async def scenario():
        async def work(seat, inp):
            assert not inp.answer_tex and "launch_snapshot.json" not in seat.context_view.store.files
            assert "error" in json.loads(seat.context_view.read("round/answer.tex"))
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(lead, "_openai", side_effect=AssertionError("blind launch fetched files")), patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "L", "include_workspace": False}], messages=messages()))["helpers"][0]
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                assert row["snapshot"] == {"kind": "blind"}
    asyncio.run(scenario())


def test_closed_turn_cannot_admit_a_late_snapshot(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def capture(*args):
            started.set()
            await release.wait()
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch("proofstack.agents.ac.multi_author.launch_snapshot", capture):
                request = asyncio.create_task(tool(lead._launch_helpers, [{"role": "prover", "task": "L"}], messages=messages()))
                await started.wait()
                lead._async_turn["open"] = False
                release.set()
                result = await request
            assert "closed" in result["error"] and not lead._async_session.records
    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_optional_history_leaves_output_headroom_and_required_history_is_not_dropped(tmp_path, monkeypatch, limit):
    if limit == "bytes":
        monkeypatch.setattr(async_helpers, "INPUT_MAX_BYTES", 4000)
    else:
        monkeypatch.setattr(async_helpers, "INPUT_MAX_FILES", 12)
    async def scenario():
        async def work(seat, inp):
            if inp.task == "first":
                for i in range(6):
                    assert json.loads(seat.context_view.publish(f"proof{i}.tex", "p" * 5000))["status"] == "published"
            else:
                assert seat.context_view.store.omissions
                assert json.loads(seat.context_view.publish("own.tex", "own proof"))["status"] == "published"
            return seat.Outputs(report="final report")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "first"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                await tool(lead._launch_helpers, [{"role": "prover", "task": "second"}])
                row = (await tool(lead._wait_helpers, ["helper2"], timeout_s=1))["helpers"][0]
                assert row["status"] == "completed" and row["report_path"] and row["context_omission_count"]
                omissions = json.loads((await tool(lead._read_context, row["context_omissions_path"]))["content"])
                assert omissions["context_omission_count"] == row["context_omission_count"]
                assert omissions["details"]
                refused = await tool(lead._launch_helpers, [{"role": "checker", "task": "check", "depends_on": ["helper1"]}])
                assert "Required helper artifact" in refused["error"]
    asyncio.run(scenario())


def test_published_task_file_never_shadows_assignment_including_after_resume(tmp_path):
    async def scenario():
        async def work(seat, inp):
            if inp.task == "original assignment":
                assert json.loads(seat.context_view.publish("task.txt", "published evidence"))["status"] == "published"
            else:
                assert json.loads(seat.context_view.read("helpers/helper1/task.txt"))["content"] == "published evidence"
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "original assignment"}])
                row = (await tool(lead._wait_helpers, ["helper1"], timeout_s=1))["helpers"][0]
                assert row["task_path"] == "helper-inputs/helper1/task.txt"
                assert (await tool(lead._read_context, row["task_path"]))["content"] == "original assignment"
                assert (await tool(lead._read_context, "helpers/helper1/task.txt"))["content"] == "published evidence"
                manifest = json.loads((await tool(lead._read_context, "manifest.json"))["content"])
                assert any(f["path"] == "helpers/helper1/task.txt" for f in manifest["files"])
                await tool(lead._launch_helpers, [{"role": "checker", "task": "check", "depends_on": ["helper1"]}])
                await tool(lead._wait_helpers, ["helper2"], timeout_s=1)
        resumed = HelperSession(lead.ctx, lead._async_session.key)
        assert not resumed.failure
        assert json.loads(resumed.read(row["task_path"]))["content"] == "original assignment"
        assert json.loads(resumed.read("helpers/helper1/task.txt"))["content"] == "published evidence"
        assert "error" in json.loads(resumed.read("helper-inputs/helper1/round/critic.md"))
    asyncio.run(scenario())


def test_artifact_heavy_jobs_keep_omissions_out_of_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(async_helpers, "INPUT_MAX_FILES", 12)
    async def scenario():
        async def work(seat, inp):
            for i in range(24):
                assert json.loads(seat.context_view.publish(f"{i}-{'x' * 120}.txt", "evidence"))["status"] == "published"
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            sizes = []
            old_details_bytes = 0
            with patch.object(SubAuthorSeat, "run", work):
                for i in range(45):
                    result = await tool(lead._launch_helpers, [{"role": "prover", "task": f"task {i}"}])
                    assert not result.get("error"), result
                    row = result["helpers"][0]
                    ident = row["agent_id"]
                    await tool(lead._wait_helpers, [ident], timeout_s=1)
                    session = lead._async_session
                    store = session.stores[ident]
                    omitted = [p for source in list(session.records)[:-1] for p in session.artifacts(source)
                               if p not in store.metadata]
                    assert row["context_omission_count"] == len(omitted)
                    old_details_bytes += len(json.dumps([p + ": input capacity reached; output headroom is reserved" for p in omitted]))
                    ledger = (session.root / "state.json").read_bytes()
                    assert b'"context_omissions"' not in ledger
                    sizes.append(len(ledger))
                    manifest = json.loads((store.root / "manifest.json").read_text())
                    assert manifest["omissions"] == store.omissions
                    assert len(store.omissions) <= async_helpers.MAX_OMISSION_DETAILS + 1
            assert old_details_bytes > async_helpers.MAX_FILE_BYTES
            assert sizes[-1] < 200_000 and sizes[-1] < 3 * sizes[21]
        restored = HelperSession(lead.ctx, session.key)
        assert not restored.failure and len(restored.records) == 45
        assert restored.status([ident])["helpers"][0]["context_omission_count"] == row["context_omission_count"]
        assert restored.stores[ident].omissions == store.omissions
        pieces, offset = [], 0
        while offset is not None:
            page = json.loads(restored.read(row["context_omissions_path"], offset, 500))
            pieces.append(page["content"])
            offset = page["next_offset"]
        details = json.loads("".join(pieces))
        assert details["details"] == store.omissions
        assert details["context_omission_count"] == len(omitted)
    asyncio.run(scenario())


def test_legacy_omission_lists_migrate_without_resetting_launch_allowances(tmp_path):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path, max_tasks_per_turn=1)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "original"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
        session = lead._async_session
        state_path = session.root / "state.json"
        state = json.loads(state_path.read_text())
        state["jobs"][0].pop("context_omission_count")
        state["jobs"][0]["context_omissions"] = [f"omission {i}: " + "x" * 2000 for i in range(200)]
        state_path.write_text(json.dumps(state))
        manifest_path = session.stores["helper1"].root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("omissions", None)
        manifest_path.write_text(json.dumps(manifest))
        resumed = HelperSession(lead.ctx, session.key)
        assert not resumed.failure
        row = resumed.status(["helper1"])["helpers"][0]
        assert row["context_omission_count"] == 200 and "context_omissions" not in row
        assert "truncated" in resumed.stores["helper1"].omissions[0]
        assert "72 further omissions" in resumed.stores["helper1"].omissions[-1]
        assert "context_omissions" not in json.loads(state_path.read_text())["jobs"][0]
        twice = HelperSession(lead.ctx, session.key)
        assert not twice.failure and twice.stores["helper1"].omissions == resumed.stores["helper1"].omissions
        with pytest.raises(ValueError, match="allowance"):
            await twice.launch(lead, [{"role": "prover", "task": "retry"}], "")
    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_publications_preserve_final_report_capacity(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(delegation_context, "MAX_TOTAL_BYTES", 100 if limit == "bytes" else 1000)
    monkeypatch.setattr(delegation_context, "MAX_FILE_BYTES", 20)
    monkeypatch.setattr(delegation_context, "MAX_FILES", 6 if limit == "bytes" else 5)
    store = DelegationContext(tmp_path, round=0)
    view = store.view(publisher="helpers/helper1", reserve_report=True)
    for i in range(4):
        assert json.loads(view.publish(f"p{i}.txt", "x" * 20))["status"] == "published"
    assert "reserved" in json.loads(view.publish("more.txt", "y"))["error"]
    assert view.retain_report("r" * 20) in store.files


def test_expired_manifest_revision_requires_restart(tmp_path):
    store = DelegationContext(tmp_path, round=0)
    view = store.view()
    first = json.loads(view.read("manifest.json", max_chars=5))
    for i in range(5):
        store.put(f"file{i}.txt", "new", source="test")
        view.read("manifest.json", max_chars=5)
    assert "expired" in json.loads(view.read("manifest.json", offset=5, revision=first["revision"]))["error"]
    assert "content" in json.loads(view.read("manifest.json", offset=0))


def test_late_snapshot_download_cannot_overwrite_fallback_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(delegation_snapshot, "SNAPSHOT_TIMEOUT_S", .02)
    released, returned = threading.Event(), threading.Event()
    def capture(*args):
        released.wait(2)
        returned.set()
        return {"answer.tex": "late result"}, []
    async def scenario():
        async def work(seat, inp):
            assert inp.answer_tex == "draft zero"
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(delegation_snapshot, "_capture", capture), patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}], messages=messages()))["helpers"][0]
                assert "timed out" in str(row["snapshot"]["warnings"])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                ledger = (lead._async_session.root / "state.json").read_bytes()
                released.set()
                assert await asyncio.to_thread(returned.wait, 1)
                await asyncio.sleep(0)
                assert (lead._async_session.root / "state.json").read_bytes() == ledger
    try:
        asyncio.run(scenario())
    finally:
        released.set()


@pytest.mark.parametrize("reader", ["lead", "helper"])
def test_manifest_pages_are_stable_across_publications(tmp_path, reader):
    async def scenario():
        async with helper_scope():
            lead = lead_at(tmp_path)
            view = lead._context.view(publisher="helpers/helper1")
            async def read(**kw):
                return await tool(lead._read_context, **kw) if reader == "lead" else json.loads(view.read(**kw))
            first = await read(path="manifest.json", max_chars=50)
            original = (await read(path="manifest.json", revision=first["revision"], max_chars=24000))["content"]
            view.publish("new.txt", "new evidence")
            pieces = [first["content"]]
            offset = first["next_offset"]
            while offset is not None:
                page = await read(path="manifest.json", offset=offset, max_chars=50, revision=first["revision"])
                pieces.append(page["content"])
                offset = page["next_offset"]
            assert "".join(pieces) == original
            assert "new.txt" not in original
            fresh = await read(path="manifest.json")
            assert fresh["revision"] != first["revision"] and "new.txt" in fresh["content"]
            assert "error" in await read(path="manifest.json", offset=50)
            assert "error" in await read(path="manifest.json", offset=50, revision="unknown")
    asyncio.run(scenario())


def test_rendered_async_prompts_describe_only_current_operations(tmp_path):
    lead = lead_at(tmp_path)
    seat = SubAuthorSeat(lead.ctx)
    content = str(lead._with_guide([{"role": "developer", "content": "Author"}]))
    content += str(seat.render_messages(seat.Inputs(role="prover", task="L", problem="P", asynchronous=True,
                                                  briefing="B", remaining_seconds=14000, deadline_utc="deadline")))
    content += lead._render_sandbox_note({"error": None, "files": [("answer.tex", "/mnt/data/file-answer.tex")]})
    for phrase in ("two-wave", "50-minute", "50 minutes", "pre-wave", "before the wave", "in this wave", "long reports are cut", "receives a summary", "no prescribed number"):
        assert phrase not in content
    assert "can take hours" in content and "Do not cancel merely" in content
    assert "Leave relevant tasks running" in content and "ANY selected" in content
    assert "turn-start" in content and "revision" in content


@pytest.mark.parametrize("target", ["scope", "cancel"])
def test_shutdown_deadline_fences_late_artifacts_and_preserves_usage(tmp_path, monkeypatch, target):
    monkeypatch.setattr(async_helpers, "SHUTDOWN_GRACE_S", .03)
    async def scenario():
        ready, release = asyncio.Event(), asyncio.Event()
        seen = []
        work_tasks = []
        async def work(seat, inp):
            seen.append(seat)
            work_tasks.append(asyncio.current_task())
            seat.interrupted_report = "Latched paid proof"
            seat.context_view.publish("proof.tex", "partial proof")
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Model a transport/finalizer that ignores even forced local
                # cancellation and completes after the research phase exits.
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                assert "closed" in json.loads(seat.context_view.publish("late.tex", "too late"))["error"]
                seat.tracker.add_usd(.75)
                seat.tracker.add_tokens(123)
        with patch.object(SubAuthorSeat, "run", work):
            async with helper_scope():
                lead = lead_at(tmp_path)
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await ready.wait()
                if target == "cancel":
                    result = await asyncio.wait_for(lead._async_session.cancel(), 1)
                    assert result["helpers"][0]["shutdown_incomplete"]
            session = lead._async_session
            row = session.records["helper1"]
            assert row["shutdown_incomplete"] and row["usage_unresolved"]
            assert "Latched paid proof" in json.loads(session.read(row["report_path"]))["content"]
            assert session.failure and seen[0].context_view.closed
            ledger = (session.root / "state.json").read_bytes()
            with pytest.raises(ValueError, match="closed"):
                session.stores["helper1"].put("unsafe.txt", "late", source="test")
            release.set()
            await asyncio.gather(*session.tasks.values(), return_exceptions=True)
            await asyncio.gather(*work_tasks, return_exceptions=True)
            await asyncio.sleep(0)
            assert (session.root / "state.json").read_bytes() == ledger
            receipts = list((session.root / "late-usage").glob("*.json"))
            assert len(receipts) == 1
            receipt = json.loads(receipts[0].read_text())
            assert receipt["cost_usd"] == .75 and receipt["tokens"] == 123 and receipt["usage_unresolved"]
            assert lead.tracker.counters.usd == .75
            reloaded = HelperSession(lead.ctx, session.key)
            assert reloaded.failure
    asyncio.run(scenario())


def test_stalled_cancellable_finalizer_is_interrupted_after_salvage(tmp_path, monkeypatch):
    monkeypatch.setattr(async_helpers, "SHUTDOWN_GRACE_S", .02)
    async def scenario():
        ready = asyncio.Event()
        work_tasks = []
        async def work(seat, inp):
            work_tasks.append(asyncio.current_task())
            seat.interrupted_report = "Paid result"
            seat.tracker.add_usd(.2)
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.Event().wait()
        with patch.object(SubAuthorSeat, "run", work):
            async with helper_scope():
                lead = lead_at(tmp_path)
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await ready.wait()
            await asyncio.wait_for(asyncio.gather(*work_tasks, return_exceptions=True), 1)
        row = lead._async_session.records["helper1"]
        assert row["usage_unresolved"] and row["cost_usd"] == .2
        assert "Paid result" in json.loads(lead._async_session.read(row["report_path"]))["content"]
    asyncio.run(scenario())
