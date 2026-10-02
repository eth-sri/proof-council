import asyncio
import json
from unittest.mock import patch

import pytest

from proofstack.agents.ac.async_helpers import helper_scope
from proofstack.agents.ac.multi_author import Author, SubAuthorSeat
from test_async_helpers import lead_at, tool


def test_smoke_four_fresh_helpers_accept_empty_ids(tmp_path):
    async def scenario():
        started, all_started, release = [], asyncio.Event(), asyncio.Event()

        async def work(seat, inp):
            started.append(inp.task)
            if len(started) == 4:
                all_started.set()
            await release.wait()
            return seat.Outputs(report="verified")

        async with helper_scope():
            lead = lead_at(tmp_path, max_threads=4, max_tasks_per_turn=4)
            tasks = [{"include_workspace": True, "agent_id": "", "task": f"Assigned n={n}.",
                      "depends_on": [], "role": role}
                     for n, role in zip((12, 13, 14, 15), ("explorer", "prover", "checker", "prover"))]
            with patch.object(SubAuthorSeat, "run", work):
                try:
                    result = await tool(lead._launch_helpers, tasks, briefing="Verify the binomial identity.")
                    assert not result.get("error"), result
                    await asyncio.wait_for(all_started.wait(), timeout=2)
                    assert len(result["helpers"]) == 4 and len(started) == 4
                    assert all(row["continued_from"] is None for row in result["helpers"])
                    ids = [row["agent_id"] for row in result["helpers"]]
                    release.set()
                    await asyncio.gather(*lead._async_session.tasks.values())
                    assert all(row["status"] == "completed" for row in (await tool(lead._helper_status, ids))["helpers"])
                    refused = await tool(lead._launch_helpers, tasks[:1])
                    assert "allowance" in refused["error"]
                finally:
                    release.set()
    asyncio.run(scenario())


@pytest.mark.parametrize("optional_id", [{}, {"agent_id": None}, {"agent_id": ""}])
def test_fresh_helper_optional_id_forms_and_real_continuation(tmp_path, optional_id):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="a lemma")

        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                first = await tool(lead._launch_helpers, [{"role": "prover", "task": "Prove L", **optional_id}])
                assert first["helpers"][0]["continued_from"] is None
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                continued = await tool(lead._launch_helpers, [{"role": "checker", "task": "Extend L", "agent_id": "helper1"}])
                assert continued["helpers"][0]["continued_from"] == "helper1"
                assert continued["helpers"][0]["agent_id"] == "helper2"
                await tool(lead._wait_helpers, ["helper2"], timeout_s=1)
    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_id", [False, 0, [], {}, "unknown", " "])
def test_invalid_helper_ids_do_not_launch_or_consume_allowance(tmp_path, invalid_id):
    async def scenario():
        async with helper_scope():
            lead = lead_at(tmp_path, max_tasks_per_turn=1)
            with patch.object(SubAuthorSeat, "run") as work:
                result = await tool(lead._launch_helpers, [{"role": "prover", "task": "L", "agent_id": invalid_id}])
            assert "error" in result and not lead._async_session.records
            work.assert_not_called()
    asyncio.run(scenario())


def test_empty_status_selection_means_all_including_previous_round(tmp_path):
    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="a lemma")

        async with helper_scope():
            lead = lead_at(tmp_path)
            assert (await tool(lead._helper_status, []))["helpers"] == []
            with patch.object(SubAuthorSeat, "run", work):
                await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}])
                await tool(lead._wait_helpers, ["helper1"], timeout_s=1)
                lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=3), "")
                await tool(lead._launch_helpers, [{"role": "checker", "task": "M"}])
                await tool(lead._wait_helpers, ["helper2"], timeout_s=1)
            for selection in (None, []):
                assert [r["agent_id"] for r in (await tool(lead._helper_status, selection))["helpers"]] == ["helper1", "helper2"]
            assert [r["agent_id"] for r in (await tool(lead._helper_status, ["helper2"]))["helpers"]] == ["helper2"]
    asyncio.run(scenario())


@pytest.mark.parametrize("selection", [None, [], "", {}, 0, ["unknown"], ["helper1", "unknown"]])
def test_wait_and_cancel_require_explicit_nonempty_valid_ids(tmp_path, selection):
    async def scenario():
        release = asyncio.Event()

        async def work(seat, inp):
            await release.wait()
            return seat.Outputs(report="done")

        async with helper_scope():
            lead = lead_at(tmp_path)
            with patch.object(SubAuthorSeat, "run", work):
                try:
                    await tool(lead._launch_helpers, [{"role": "prover", "task": "L"}, {"role": "checker", "task": "M"}])
                    assert (await tool(lead._wait_helpers, selection, timeout_s=0)).get("error")
                    assert (await tool(lead._cancel_helpers, selection)).get("error")
                    assert all(not t.done() and not t.cancelling() for t in lead._async_session.tasks.values())
                    cancelled = await tool(lead._cancel_helpers, ["helper1"])
                    assert cancelled["helpers"][0]["status"] == "cancelled"
                    assert not lead._async_session.tasks["helper2"].done()
                finally:
                    release.set()
    asyncio.run(scenario())


@pytest.mark.parametrize("reader", ["lead", "helper"])
@pytest.mark.parametrize("optional_revision", [{}, {"revision": None}, {"revision": ""}])
def test_manifest_initial_revision_forms_preserve_pagination(tmp_path, reader, optional_revision):
    async def scenario():
        async with helper_scope():
            lead = lead_at(tmp_path)
            view = lead._context.view(publisher="helpers/helper1")

            async def read(**kw):
                return await tool(lead._read_context, **kw) if reader == "lead" else json.loads(view.read(**kw))

            first = await read(path="manifest.json", offset=0, max_chars=50, **optional_revision)
            assert "error" not in first
            original = (await read(path="manifest.json", revision=first["revision"], max_chars=24000))["content"]
            view.publish("new.txt", "published after the first page")
            pieces, offset = [first["content"]], first["next_offset"]
            while offset is not None:
                page = await read(path="manifest.json", offset=offset, max_chars=50, revision=first["revision"])
                pieces.append(page["content"])
                offset = page["next_offset"]
            assert "".join(pieces) == original and "new.txt" not in original
            fresh = await read(path="manifest.json", **optional_revision)
            assert "new.txt" in fresh["content"] and fresh["revision"] != first["revision"]
            for invalid in (None, "", "unknown", "latest", "0", False, [], {}):
                assert "error" in await read(path="manifest.json", offset=50, revision=invalid)
            for invalid in ("unknown", "latest", "0", False, [], {}):
                assert "error" in await read(path="manifest.json", offset=0, revision=invalid)
    asyncio.run(scenario())


def test_optional_tool_schemas_allow_null_but_controls_require_ids(tmp_path):
    lead = lead_at(tmp_path)
    descriptions = {desc["function"]["name"]: desc["function"]
                    for _, desc in lead._context_tools() + lead._helper_tools()}
    revision = descriptions["read_context"]["parameters"]["properties"]["revision"]
    helper_id = descriptions["delegate"]["parameters"]["properties"]["tasks"]["items"]["properties"]["agent_id"]
    selection = descriptions["helper_status"]["parameters"]["properties"]["agent_ids"]
    assert revision["type"] == helper_id["type"] == ["string", "null"]
    assert selection["type"] == ["array", "null"]
    assert "offset 0" in revision["description"] and "empty" in helper_id["description"]
    for name in ("wait_helpers", "cancel_helpers"):
        parameters = descriptions[name]["parameters"]
        assert "agent_ids" in parameters["required"]
        assert parameters["properties"]["agent_ids"]["type"] == "array"
        assert parameters["properties"]["agent_ids"]["minItems"] == 1
