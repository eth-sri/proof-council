import asyncio
import json
import time
from unittest.mock import patch

import pytest

from proofstack.agents.ac.async_helpers import HelperSession, helper_scope
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext


def lead_at(root, **settings):
    ctx = RunContext.create(root_workdir=root, flat=True,
        run_budget=BudgetSpec(max_wallclock_s=80000, max_usd=10),
        component_configs={"Author": {"delegation": {"enabled": True, "asynchronous": True,
            "sandbox_carry_over": False, **settings}}})
    lead = MultiAuthor(ctx, name="Author")
    lead._render_container_messages(Author.Inputs(problem="P", round=0, n_rounds=3, answer_tex="draft zero"), "")
    return lead


async def tool(fn, *args, **kwargs):
    return json.loads(await asyncio.to_thread(fn, *args, **kwargs))


def test_launch_is_nonblocking_and_wait_timeout_does_not_cancel(tmp_path):
    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()
        async def work(seat, inp):
            started.set()
            await finish.wait()
            return seat.Outputs(report="Proved L.")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                launched = await tool(lead._launch_helpers, [{"role": "prover", "task": "Prove L"}])
                ident = launched["helpers"][0]["agent_id"]
                await started.wait()
                assert launched["helpers"][0]["status"] == "running"
                assert (await tool(lead._wait_helpers, [ident], timeout_s=.01))["helpers"][0]["status"] == "running"
                finish.set()
                done = await tool(lead._wait_helpers, [ident], timeout_s=1)
                assert done["helpers"][0]["status"] == "completed"
                path = done["helpers"][0]["report_path"]
                assert (await tool(lead._read_context, path))["content"] == "Proved L."
    asyncio.run(scenario())


def test_finished_helper_contents_are_disk_backed_and_verified_on_read(tmp_path):
    async def scenario():
        async def work(seat, inp):
            seat.context_view.publish("proof.tex", "a lemma")
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                store = lead._async_session.stores["helper1"]
                assert not store.files
                assert (await tool(lead._read_context, "helpers/helper1/proof.tex"))["content"] == "a lemma"
                assert not store.files
                (store.root / "helpers/helper1/proof.tex").write_text("tampered")
                with pytest.raises(ValueError, match="digest mismatch"):
                    lead._async_session.read("helpers/helper1/proof.tex")
                assert not store.files
    asyncio.run(scenario())


def test_admission_failure_does_not_leave_phantom_jobs_or_start_paid_work(tmp_path):
    async def scenario():
        from proofstack.agents.ac.delegation_context import DelegationContext
        original = DelegationContext.put
        calls = []
        async def work(seat, inp):
            calls.append(inp.task)
            return seat.Outputs(report="done")
        def put(store, name, content, **kwargs):
            if name == "task.txt" and content == "second":
                raise ValueError("test capacity failure")
            return original(store, name, content, **kwargs)
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                with patch.object(DelegationContext, "put", put):
                    result = await tool(lead._launch_helpers, [
                        {"role": "prover", "task": "first"}, {"role": "checker", "task": "second"}])
                assert "capacity failure" in result["error"]
                assert not lead._async_session.records and not lead._async_session.tasks
                assert not calls
                await tool(lead._launch_helpers, [{"role": "prover", "task": "retry"}])
                row = (await tool(lead._wait_helpers, ["helper1"], timeout_s=1))["helpers"][0]
                assert row["status"] == "completed" and calls == ["retry"]
    asyncio.run(scenario())


def test_required_dependency_copy_failure_refuses_launch(tmp_path):
    async def scenario():
        from proofstack.agents.ac.delegation_context import DelegationContext
        original = DelegationContext.put
        async def work(seat, inp):
            seat.context_view.publish("proof.tex", "a lemma")
            return seat.Outputs(report="done")
        def put(store, name, content, **kwargs):
            if "helper2" in store.root.parts and name == "helpers/helper1/proof.tex":
                raise ValueError("test capacity failure")
            return original(store, name, content, **kwargs)
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                with patch.object(DelegationContext, "put", put):
                    result = await tool(lead._launch_helpers, [{
                        "role": "checker", "task": "Check L", "depends_on": ["helper1"]}])
                assert "Required helper artifact" in result["error"]
                assert list(lead._async_session.records) == ["helper1"]
    asyncio.run(scenario())


def test_scope_joins_all_sessions_even_when_one_close_fails(caplog):
    from proofstack.agents.ac.async_helpers import _SESSIONS
    async def scenario():
        joined = []
        class Failing:
            async def close(self):
                raise RuntimeError("ledger failure")
        class Slow:
            async def close(self):
                await asyncio.sleep(.02)
                joined.append(True)
        async with helper_scope():
            _SESSIONS.get().update(failing=Failing(), slow=Slow())
        assert joined == [True]
        assert "ledger failure" in caplog.text
    asyncio.run(scenario())


def test_cancellation_retains_tex_code_and_data_without_markdown(tmp_path):
    async def scenario():
        published = asyncio.Event()
        async def work(seat, inp):
            for name, content in [("proof.tex", "proof"), ("check.py", "assert 1"), ("result.json", '{"ok":true}')]:
                assert "published" in seat.context_view.publish(name, content)
            published.set()
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "Prove L"}]))["helpers"][0]
                await published.wait()
                row = (await tool(lead._cancel_helpers, [row["agent_id"]]))["helpers"][0]
                assert row["status"] == "cancelled"
                report = (await tool(lead._read_context, row["report_path"]))["content"]
                assert "proof.tex" in report and "check.py" in report and "result.json" in report
                assert "unreviewed" in report
    asyncio.run(scenario())


def test_helpers_survive_author_turn_and_do_not_change_reviewed_draft(tmp_path):
    async def scenario():
        finish = asyncio.Event()
        async def work(seat, inp):
            assert inp.answer_tex == "draft zero"
            await finish.wait()
            return seat.Outputs(report="Late proof for draft zero")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "Prove L"}]))["helpers"][0]
                old_session = lead._async_session
                lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=3, answer_tex="reviewed draft one"), "")
                assert lead._async_session is old_session
                messages = lead._with_guide([{"role": "developer", "content": "author"}])
                assert "Existing asynchronous helpers" in messages[-1]["content"]
                assert row["agent_id"] in messages[-1]["content"]
                finish.set()
                done = (await tool(lead._wait_helpers, [row["agent_id"]], timeout_s=1))["helpers"][0]
                assert done["round"] == 0 and done["input_hash"] == row["input_hash"]
                assert lead._current_inp.answer_tex == "reviewed draft one"
                manifest = json.loads((await tool(lead._read_context, "manifest.json"))["content"])
                assert any(e["path"] == done["report_path"] for e in manifest["files"])
    asyncio.run(scenario())


def test_global_concurrency_includes_previous_round_jobs(tmp_path):
    async def scenario():
        async def work(seat, inp):
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path, max_threads=1)
            with patch.object(SubAuthorSeat, "run", work):
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}]))["helpers"][0]
                lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=3), "")
                rejected = await tool(lead._launch_helpers, [{"role": "checker", "task": "C"}])
                assert "slots" in rejected["error"]
                assert len(lead._async_session.records) == 1
                await tool(lead._cancel_helpers, [row["agent_id"]])
                assert "helpers" in await tool(lead._launch_helpers, [{"role": "checker", "task": "C"}])
    asyncio.run(scenario())


def test_continuation_has_new_id_and_reads_original_artifacts(tmp_path):
    async def scenario():
        async def work(seat, inp):
            if inp.task == "initial":
                seat.context_view.publish("proof.tex", "original lemma")
            else:
                assert "original lemma" in seat.context_view.read("helpers/helper1/proof.tex")
                assert "first report" in seat.context_view.read("helpers/helper1/final-response.md")
            return seat.Outputs(report="first report" if inp.task == "initial" else "improved report")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "initial"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                row = (await tool(lead._launch_helpers, [{"role": "prover", "task": "continue", "agent_id": "helper1"}]))["helpers"][0]
                assert row["agent_id"] == "helper2" and row["continued_from"] == "helper1"
                assert (await tool(lead._wait_helpers, ["helper2"], timeout_s=1))["helpers"][0]["status"] == "completed"
                assert (await tool(lead._read_context, "helpers/helper1/final-response.md"))["content"] == "first report"
    asyncio.run(scenario())


def test_scope_closes_and_accounts_before_cleanup(tmp_path):
    async def scenario():
        started = asyncio.Event()
        async def work(seat, inp):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                seat.tracker.add_usd(.25)
        with patch.object(SubAuthorSeat, "run", work):
            async with helper_scope():
                lead = lead_at(tmp_path)
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await started.wait()
            session = lead._async_session
            assert session.closed and all(t.done() for t in session.tasks.values())
            assert lead.tracker.counters.usd == .25
            assert session.records["helper1"]["cost_usd"] == .25
    asyncio.run(scenario())


def test_empty_reply_is_failure_not_success(tmp_path):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                row = (await tool(lead._wait_helpers, ["helper1"], timeout_s=1))["helpers"][0]
                assert row["status"] == "failed" and "EmptyResponse" in row["error"]
    asyncio.run(scenario())


def test_recovery_retains_published_files_and_consumed_allowance(tmp_path):
    async def scenario():
        published = asyncio.Event()
        async def work(seat, inp):
            seat.context_view.publish("proof.tex", "partial")
            published.set()
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path, max_tasks_per_turn=1)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await published.wait()
                state = json.loads((lead._async_session.root / "state.json").read_text())
                assert state["jobs"][0]["status"] == "running"
                # Simulate a crash snapshot without racing the original job's files.
                import shutil
                copy = tmp_path / "recovered"
                shutil.copytree(lead._async_session.root, copy / "async-helpers" / lead._async_session.key)
                ctx = RunContext.create(root_workdir=copy, flat=True)
                restored = HelperSession(ctx, lead._async_session.key)
                row = restored.records["helper1"]
                assert row["status"] == "interrupted"
                assert "proof.tex" in json.loads(restored.read(row["report_path"]))["content"]
                assert "partial" in restored.read("helpers/helper1/proof.tex")
                assert not restored.tasks
                with pytest.raises(ValueError, match="allowance"):
                    await restored.launch(lead, [{"role": "prover", "task": "repeat"}], "")
    asyncio.run(scenario())


def test_deadline_uses_research_not_fifty_minutes_or_short_lead_request(tmp_path):
    async def scenario():
        seen = []
        async def work(seat, inp):
            seen.append(inp)
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}],
                           call_deadline_monotonic_s=time.monotonic()+30)
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
        # The Astra model's 14,000-second invocation ceiling still applies.
        assert 13900 < seen[0].remaining_seconds <= 14000
        assert seen[0].asynchronous
    asyncio.run(scenario())


def test_shared_budget_cancels_siblings(tmp_path):
    async def scenario():
        async def work(seat, inp):
            if inp.role == "checker":
                seat.tracker.add_usd(11)
                return seat.Outputs(report="paid result")
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}, {"role": "checker", "task": "C"}])
                await asyncio.wait_for(asyncio.gather(*lead._async_session.tasks.values(), return_exceptions=True), 3)
                assert lead.tracker.counters.usd == 11
                assert all(t.done() for t in lead._async_session.tasks.values())
                assert "error" in await tool(lead._launch_helpers, [{"role": "prover", "task": "more"}])
    asyncio.run(scenario())


def test_research_deadline_produces_durable_partial_handoff(tmp_path):
    async def scenario():
        async def work(seat, inp):
            seat.context_view.publish("calculation.json", '{"partial":true}')
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path, helper_timeout_s=1)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                row = (await tool(lead._wait_helpers, ["helper1"], timeout_s=2))["helpers"][0]
                assert row["status"] == "incomplete"
                assert "deadline" in row["error"]
                assert "calculation.json" in (await tool(lead._read_context, row["report_path"]))["content"]
    asyncio.run(scenario())


def test_blind_helper_gets_only_explicit_dependencies(tmp_path):
    async def scenario():
        async def work(seat, inp):
            if inp.task == "first":
                seat.context_view.publish("proof.tex", "public lemma")
            else:
                assert not inp.answer_tex
                assert "error" in json.loads(seat.context_view.read("round/answer.tex"))
                assert "public lemma" in seat.context_view.read("helpers/helper1/proof.tex")
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "first"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                await tool(lead._launch_helpers, [{"role": "checker", "task": "check", "include_workspace": False, "depends_on": ["helper1"]}])
                assert (await tool(lead._wait_helpers, ["helper2"], timeout_s=1))["helpers"][0]["status"] == "completed"
                blocked = await tool(lead._launch_helpers, [{"role": "checker", "task": "blind", "include_workspace": False, "agent_id": "helper1"}])
                assert "cannot forget" in blocked["error"]
    asyncio.run(scenario())


def test_scope_cancels_on_exception_and_keeps_original_failure(tmp_path):
    async def scenario():
        async def work(seat, inp):
            await asyncio.Event().wait()
        with patch.object(SubAuthorSeat, "run", work), pytest.raises(RuntimeError, match="outer failure"):
            async with helper_scope():
                lead = lead_at(tmp_path)
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                raise RuntimeError("outer failure")
        assert all(t.done() for t in lead._async_session.tasks.values())
        assert lead._async_session.records["helper1"]["status"] == "cancelled"
    asyncio.run(scenario())


def test_cancel_before_job_starts_has_terminal_state(tmp_path):
    async def scenario():
        async with helper_scope():
            lead = lead_at(tmp_path)
            session = lead._async_session
            await session.launch(lead, [{"role": "prover", "task": "L"}], "")
            await session.cancel(["helper1"])
            assert session.records["helper1"]["status"] == "cancelled"
            assert session.records["helper1"]["report_path"]
            assert session.records["helper1"]["cost_usd"] == 0
    asyncio.run(scenario())


def test_launch_from_worker_without_context_still_nests_under_author(tmp_path):
    import concurrent.futures
    from proofstack.agent import _AGENT_PATH, _PARENT_CALL_ID
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="done")
        async with helper_scope():
            path = _AGENT_PATH.set(("Research", "Author"))
            parent = _PARENT_CALL_ID.set("author-call")
            try:
                lead = lead_at(tmp_path)
            finally:
                _AGENT_PATH.reset(path)
                _PARENT_CALL_ID.reset(parent)
            with patch.object(SubAuthorSeat, "run", work), concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                result = json.loads(await asyncio.wrap_future(future))
                assert result["helpers"][0]["agent_id"] == "helper1"
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        event = next(e for e in events if e["kind"] == "agent.start" and e["agent"].startswith("SubAuthor."))
        assert event["agent_path"] == "Research.Author.SubAuthor.helper1"
        assert event["parent_call_id"] == "author-call"
    asyncio.run(scenario())


def test_every_async_local_tool_preserves_lead_sandbox_before_return(tmp_path):
    async def scenario():
        snapshots = []
        async def snapshot(self, container, **kwargs):
            snapshots.append(container)
            return {"error": None, "files": [["answer.tex", "/mnt/data/file-proof-answer.tex"]]}
        async def work(seat, inp):
            return seat.Outputs(report="done")
        async with helper_scope():
            lead = lead_at(tmp_path, sandbox_carry_over=True)
            messages = [{"type": "code_interpreter_call", "container_id": "cntr-current"}]
            with patch.object(SubAuthorSeat, "run", work), patch.object(MultiAuthor, "_snapshot_sandbox", snapshot):
                results = [await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}], messages=messages)]
                results.append(await tool(lead._helper_status, messages=messages))
                results.append(await tool(lead._wait_helpers, ["helper1"], timeout_s=1, messages=messages))
                results.append(await tool(lead._read_context, "helpers/helper1/final-response.md", messages=messages))
                results.append(await tool(lead._cancel_helpers, ["helper1"], messages=messages))
                assert snapshots == ["cntr-current"] * 5
                assert all("file-proof-answer.tex" in r["sandbox_note"] for r in results)
    asyncio.run(scenario())


def test_corrupted_artifacts_are_quarantined_without_resetting_launch_allowance(tmp_path):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="original")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
        session = lead._async_session
        (session.stores["helper1"].root / session.records["helper1"]["report_path"]).write_text("tampered")
        restored = HelperSession(lead.ctx, session.key)
        assert not restored.records and not restored.failure
        assert "error" in json.loads(restored.read("helpers/helper1/final-response.md"))
        quarantines = list(session.root.parent.glob(session.key + ".quarantine-*"))
        assert len(quarantines) == 1
        assert (quarantines[0] / "helper1/context/helpers/helper1/final-response.md").read_text() == "tampered"
        assert restored.retired_launches[session.records["helper1"]["input_hash"]] == 1
        assert restored.next_id == 2
        with patch.object(lead, "_delegation_cfg", return_value={**lead._delegation_cfg(), "max_tasks_per_turn": 1}):
            with pytest.raises(ValueError, match="allowance"):
                await restored.launch(lead, [{"role": "prover", "task": "again"}], "")
        with patch.object(SubAuthorSeat, "run", work):
            launched = await restored.launch(lead, [{"role": "prover", "task": "new work"}], "")
            assert launched["helpers"][0]["agent_id"] == "helper2"
            await restored.wait(["helper2"], 1)
        await restored.close()
        reloaded = HelperSession(lead.ctx, session.key)
        assert list(reloaded.records) == ["helper2"] and reloaded.next_id == 3
        assert reloaded.retired_launches == restored.retired_launches
    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["cancel", "close", "scope", "helper"])
def test_repeated_cancellation_drains_paid_handoff_and_accounting(tmp_path, target):
    async def scenario():
        ready, finalizing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        saved = []
        async def work(seat, inp):
            seat.interrupted_report = "Paid lemma latched before cancellation"
            seat.context_view.publish("proof.tex", "partial proof")
            saved.append(seat)
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalizing.set()
                await release.wait()
                seat.tracker.add_usd(.25)
                seat.tracker.add_tokens(100)
        async def research():
            async with helper_scope():
                lead = lead_at(tmp_path)
                saved.append(lead)
                session = lead._async_session
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await ready.wait()
                if target == "scope":
                    return "accepted"
                if target == "cancel":
                    await session.cancel(["helper1"])
                elif target == "close":
                    await session.close()
                else:
                    session.tasks["helper1"].cancel()
                    await asyncio.wait([session.tasks["helper1"]])
        with patch.object(SubAuthorSeat, "run", work):
            outer = asyncio.create_task(research())
            await asyncio.wait_for(finalizing.wait(), 3)
            lead, seat = saved
            victim = lead._async_session.tasks["helper1"] if target == "helper" else outer
            victim.cancel()
            await asyncio.sleep(0)
            victim.cancel()
            await asyncio.sleep(0)
            assert not outer.done()
            release.set()
            if target == "helper":
                await asyncio.wait_for(outer, 3)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(outer, 3)
        session = lead._async_session
        assert session.closed and all(t.done() for t in session.tasks.values())
        assert seat.context_view.closed and not session.failure
        assert lead.tracker.counters.usd == .25 and lead.tracker.counters.tokens == 100
        rec = json.loads((session.root / "state.json").read_text())["jobs"][0]
        assert rec["status"] == "cancelled" and rec["cost_usd"] == .25
        handoff = json.loads(session.read(rec["report_path"]))["content"]
        assert "Paid lemma latched" in handoff and "proof.tex" in handoff
    asyncio.run(scenario())


@pytest.mark.parametrize("original_failure", [False, True])
@pytest.mark.parametrize("event_failure", [False, True])
def test_helper_bookkeeping_failure_preserves_research_outcome(tmp_path, caplog, original_failure, event_failure):
    async def scenario():
        accepted = {"answer": "accepted manuscript", "critic_accepted": True}
        expected_error = ValueError("original research failure")
        async def research():
            async with helper_scope():
                lead = lead_at(tmp_path)
                session = lead._async_session
                def fail_save():
                    raise OSError("ledger disk write failed")
                async def fail_emit(*args, **kwargs):
                    raise OSError("events disk write failed")
                session._save = fail_save
                session.failure = "one helper's bookkeeping failed"
                if event_failure:
                    lead.ctx.events.emit = fail_emit
                if original_failure:
                    raise expected_error
                return accepted
        if original_failure:
            with pytest.raises(ValueError) as raised:
                await research()
            assert raised.value is expected_error
        else:
            assert await research() is accepted
        assert "ledger disk write failed" in caplog.text
        assert "one helper's bookkeeping failed" in caplog.text
        if not event_failure:
            events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
            assert any(e["kind"] == "ac.author.helper_warning" for e in events)
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["json", "shape", "record", "missing_ledger", "missing_manifest"])
def test_damaged_checkpoint_does_not_block_later_author_turns(tmp_path, damage):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="old helper result")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
        root = lead._async_session.root
        ledger = root / "state.json"
        if damage == "missing_ledger":
            ledger.unlink()
        elif damage == "missing_manifest":
            (root / "helper1/context/manifest.json").unlink()
        elif damage == "record":
            state = json.loads(ledger.read_text())
            state["jobs"] = [None]
            ledger.write_text(json.dumps(state))
        else:
            ledger.write_text("not JSON" if damage == "json" else "[]")
        async with helper_scope():
            recovered_lead = lead_at(tmp_path)
            session = recovered_lead._async_session
            assert not session.records
            for round in (1, 2):
                messages = recovered_lead._render_container_messages(
                    Author.Inputs(problem="P", round=round, n_rounds=3, answer_tex="current manuscript"), "")
                assert "checkpoint rejected" in str(messages)
                assert recovered_lead._async_session is session
                assert recovered_lead._current_inp.answer_tex == "current manuscript"
            if damage != "missing_manifest":
                rejected = await tool(recovered_lead._launch_helpers, [{"role": "prover", "task": "repeat"}])
                assert "delegation is disabled" in rejected["error"]
        # The quarantine decision, including unknown launch history, survives resume.
        reloaded = HelperSession(recovered_lead.ctx, session.key)
        assert bool(reloaded.failure) == (damage != "missing_manifest")
        assert len(list(root.parent.glob(session.key + ".quarantine-*"))) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["rename", "parent_symlink", "replacement_write"])
def test_unmovable_checkpoint_disables_helpers_without_overwriting_it(tmp_path, failure):
    from pathlib import Path
    from proofstack.agents.ac.delegation_recovery import problem_key
    async def scenario():
        parent = tmp_path / "async-helpers"
        if failure == "parent_symlink":
            outside = tmp_path / "outside"
            outside.mkdir()
            parent.symlink_to(outside, target_is_directory=True)
        root = parent / problem_key("P")
        root.mkdir(parents=True)
        state = root / "state.json"
        state.write_text("damaged evidence")
        async with helper_scope():
            if failure == "rename":
                with patch.object(Path, "rename", side_effect=PermissionError("denied")):
                    lead = lead_at(tmp_path)
            elif failure == "replacement_write":
                with patch.object(HelperSession, "_save", side_effect=OSError("disk full")):
                    lead = lead_at(tmp_path)
            else:
                lead = lead_at(tmp_path)
            assert lead._async_session.failure and lead._async_session._persistence_disabled
            lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=3), "")
        if failure == "replacement_write":
            quarantines = list(parent.glob(root.name + ".quarantine-*"))
            assert len(quarantines) == 1
            assert (quarantines[0] / "state.json").read_text() == "damaged evidence"
            # Renaming succeeded but saving the replacement did not. A restart
            # must not mistake the missing ledger for a new allowance.
            reloaded = HelperSession(lead.ctx, root.name)
            assert reloaded.failure and reloaded._persistence_disabled
            with pytest.raises(ValueError, match="delegation is disabled"):
                await reloaded.launch(lead, [{"role": "prover", "task": "repeat"}], "")
        else:
            assert state.read_text() == "damaged evidence"
        assert not lead._async_session.tasks
    asyncio.run(scenario())


def test_completed_reply_survives_cancellation_during_logging(tmp_path):
    from proofstack.kinds.api_call import APICallAgent
    async def scenario():
        latched = asyncio.Event()
        async def paid_response(seat, inp):
            seat._on_response("Paid proof before log await", inp)
            latched.set()
            await asyncio.Event().wait()
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(APICallAgent, "run", paid_response):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await latched.wait()
                row = (await tool(lead._cancel_helpers, ["helper1"]))["helpers"][0]
                assert "Paid proof before log await" in (await tool(lead._read_context, row["report_path"]))["content"]
    asyncio.run(scenario())


def test_failed_provider_output_is_not_success_even_with_partial_text(tmp_path):
    async def scenario():
        async def work(seat, inp):
            seat.provider_outcomes = [{"status": "failed"}]
            return seat.Outputs(report="Partial explanation before invalid_prompt")
        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                row = (await tool(lead._wait_helpers, ["helper1"], timeout_s=1))["helpers"][0]
                assert row["status"] == "failed"
                assert "Partial explanation" in (await tool(lead._read_context, row["report_path"]))["content"]
    asyncio.run(scenario())


@pytest.mark.parametrize("dag", [False, True])
@pytest.mark.parametrize("bookkeeping_failure", [False, True])
def test_real_workflow_wrappers_own_helper_lifetime(tmp_path, dag, bookkeeping_failure):
    from proofstack.agents.ac.ac_workflow import ACWorkflow, ACDAGWorkflow
    from proofstack.agents.dag_workflow import DAGWorkflow
    async def scenario():
        saved = []
        started = asyncio.Event()
        async def work(seat, inp):
            started.set()
            await asyncio.Event().wait()
        async def research(workflow, inp):
            lead = lead_at(tmp_path)
            saved.append(lead._async_session)
            await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
            await started.wait()
            assert not saved[0].closed
            if bookkeeping_failure:
                def fail_save():
                    raise OSError("ledger write failed")
                saved[0]._save = fail_save
            return "research result"
        cls = ACDAGWorkflow if dag else ACWorkflow
        base, method = (DAGWorkflow, "run") if dag else (ACWorkflow, "_run_research")
        ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                                component_configs={"ACDAGWorkflow": {"dag": {"nodes": [], "outputs": {}}}})
        workflow = cls(ctx)
        with patch.object(base, method, research), patch.object(SubAuthorSeat, "run", work):
            assert await workflow.run(None) == "research result"
        assert saved[0].closed
        assert all(t.done() for t in saved[0].tasks.values())
        assert saved[0].records["helper1"]["status"] == "cancelled"
        assert bool(saved[0].failure) == bookkeeping_failure
    asyncio.run(scenario())


def test_standalone_author_result_survives_helper_close_failure(tmp_path):
    async def scenario():
        lead = lead_at(tmp_path)
        accepted = Author.Outputs(answer_tex="accepted manuscript")
        async def author_run(self, inp):
            return accepted
        def fail_save():
            raise OSError("ledger write failed")
        lead._async_session._save = fail_save
        lead._async_session.failure = "helper bookkeeping failed"
        with patch.object(Author, "run", author_run):
            assert await lead.run(lead._current_inp) is accepted
        assert lead._async_session.closed
    asyncio.run(scenario())
