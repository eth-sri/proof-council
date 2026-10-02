from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mathagents.config_loader import _safe_deepcopy  # noqa: E402
from proofstack.agents.ac.ac_workflow import ACWorkflow  # noqa: E402
from proofstack.agents.ac.author import Author  # noqa: E402
from proofstack.agents.ac.container_files import find_container_id  # noqa: E402
from proofstack.agents.ac.multi_author import (  # noqa: E402
    DELEGATION_DEFAULTS,
    ROLE_DESCRIPTIONS,
    MultiAuthor,
    SubAuthorSeat,
)
from proofstack.agent import _AGENT_PATH, _PARENT_CALL_ID  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


def _ctx(temp_dir: str, delegation: dict | None = None, **author_cfg) -> RunContext:
    components: dict = {}
    if delegation is not None or author_cfg:
        components["Author"] = {**author_cfg}
        if delegation is not None:
            components["Author"]["delegation"] = delegation
    return RunContext.create(
        run_id="test",
        root_workdir=temp_dir,
        flat=True,
        component_configs=components,
    )


def _inp(**kw) -> Author.Inputs:
    base = dict(problem="Prove P.", round=1, n_rounds=3, answer_tex="A", research_notes_tex="N", references_bib="B")
    base.update(kw)
    return Author.Inputs(**base)


def _tool_names(kwargs: dict) -> list[str]:
    names = []
    for fn, desc in kwargs["tools"]:
        names.append(desc.get("function", {}).get("name") if desc.get("type") == "function" else desc["type"])
    return names


class MultiAuthorDisabledTests(unittest.TestCase):
    def test_disabled_matches_plain_author(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(td)
            plain = Author(ctx)
            multi = MultiAuthor(ctx, name="Author")
            self.assertEqual(multi.name, "Author")
            self.assertFalse(multi.delegation_enabled())
            self.assertEqual(multi.extra_client_kwargs(), plain.extra_client_kwargs())
            self.assertEqual(
                multi._render_container_messages(_inp(), "listing"),
                plain._render_container_messages(_inp(), "listing"),
            )
            captured: list[dict] = []
            ctx.api_client_factory = lambda cfg: captured.append(cfg) or object()
            multi._build_api_client_with_file_ids(["file-1"])
            self.assertEqual(_tool_names(captured[-1]), ["code_interpreter", "web_search_preview"])
            self.assertEqual(captured[-1]["max_tool_calls"], Author.MAX_TOOL_CALLS)

    def test_workflow_wires_multi_author_under_author_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(td, delegation={"enabled": True}, model="models/openai/gpt-54-mini")
            wf = ACWorkflow(ctx)
            self.assertIsInstance(wf.author, MultiAuthor)
            self.assertEqual(wf.author.name, "Author")
            self.assertEqual(wf.author.MODEL, "models/openai/gpt-54-mini")
            self.assertTrue(wf.author.delegation_enabled())


class MultiAuthorEnabledTests(unittest.TestCase):
    def test_enabled_adds_delegate_tool_and_guide(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(td, delegation={"enabled": True, "max_threads": 3, "roles": ["prover", "checker"]})
            multi = MultiAuthor(ctx, name="Author")
            kwargs = multi.extra_client_kwargs()
            self.assertEqual(_tool_names(kwargs), ["code_interpreter", "web_search_preview", "read_context", "delegate"])
            fn, desc = kwargs["tools"][-1]
            self.assertTrue(callable(fn))
            self.assertEqual(desc["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]["role"]["enum"], ["prover", "checker"])

            messages = multi._render_container_messages(_inp(), "listing")
            dev = messages[0]["content"]
            self.assertIn("Delegation (multi-agent mode)", dev)
            self.assertIn("up to 3 independent tasks", dev)
            self.assertIn("- prover:", dev)
            self.assertNotIn("- explorer:", dev)
            plain_dev = Author(ctx)._render_container_messages(_inp(), "listing")[0]["content"]
            self.assertTrue(dev.startswith(plain_dev))

            captured: list[dict] = []
            ctx.api_client_factory = lambda cfg: captured.append(cfg) or object()
            multi._build_api_client_with_file_ids(["file-1"])
            self.assertEqual(_tool_names(captured[-1]), ["code_interpreter", "web_search_preview", "read_context", "delegate"])
            self.assertEqual(captured[-1]["tools"][0][1]["container"]["file_ids"], ["file-1"])

    def test_wave_runs_seats_and_formats_reports(self) -> None:
        seen: list[SubAuthorSeat.Inputs] = []
        active = {"n": 0, "max": 0}

        async def fake_run(self, inp):
            seen.append(inp)
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
            await asyncio.sleep(0.05)
            active["n"] -= 1
            rendered = self.render_messages(inp)
            return self.Outputs(
                report=f"## Summary\nreport for {inp.role}: {inp.task}",
                messages_after=rendered + [{"role": "assistant", "content": "prev"}],
            )

        async def scenario(td: str) -> None:
            ctx = _ctx(
                td,
                delegation={"enabled": True, "max_threads": 2, "max_waves": 2, "subagent_model": "models/openai/gpt-54-mini"},
            )
            multi = MultiAuthor(ctx, name="Author")
            # Simulate being inside Author.__call__: seats must nest under it.
            _AGENT_PATH.set(("ACWorkflow", "Author"))
            _PARENT_CALL_ID.set("lead01")
            multi._render_container_messages(_inp(), "listing")  # captures loop, ctxvars + inputs

            fn, _ = multi._delegate_tool()
            tasks = [
                {"role": "prover", "task": "prove L1"},
                {"role": "checker", "task": "check S2", "include_workspace": False},
                {"role": "explorer", "task": "find refs"},
            ]
            # The real APIClient runs the tool synchronously in a worker thread.
            out = await asyncio.to_thread(fn, tasks=tasks, messages=[])
            self.assertIn("Delegation wave 1 of 2", out)
            self.assertIn("### prover1 (role=prover, model=gpt-54-mini", out)
            self.assertIn("### checker2 (role=checker", out)
            self.assertIn("### explorer3 (role=explorer", out)
            self.assertIn("report for prover: prove L1", out)
            self.assertIn("Waves remaining this turn: 1", out)
            self.assertEqual(active["max"], 2)
            self.assertEqual(len(seen), 3)
            by_role = {s.role: s for s in seen}
            self.assertEqual(by_role["prover"].answer_tex, "A")
            self.assertEqual(by_role["checker"].answer_tex, "")
            self.assertFalse(by_role["checker"].include_workspace)
            self.assertEqual(by_role["prover"].problem, "Prove P.")

            # Follow-up continues the seat's conversation.
            out2 = await asyncio.to_thread(fn, tasks=json.dumps([{"role": "prover", "task": "fix step 3", "agent_id": "prover1"}]), messages=[])
            self.assertIn("Delegation wave 2 of 2", out2)
            self.assertIn("### prover1 (role=prover", out2)
            self.assertTrue(seen[-1].prior_messages)
            self.assertEqual(seen[-1].prior_messages[-1]["content"], "prev")
            self.assertEqual(seen[-1].task, "fix step 3")

            out3 = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "again"}], messages=[])
            self.assertIn("No delegation waves left", out3)
            self.assertEqual(len(seen), 4)

            self.assertIn("4 subagent call(s) in 2 wave(s)", multi.delegation_summary())
            sub = multi.workdir / "subagents"
            self.assertTrue((sub / "wave1-prover1.md").exists())
            self.assertTrue((sub / "wave2-prover1.md").exists())
            log = json.loads((sub / "delegation_log.json").read_text())
            self.assertEqual([r["agent_id"] for r in log], ["prover1", "checker2", "explorer3", "prover1"])

            events = [json.loads(l) for l in (Path(td) / "events.jsonl").read_text().splitlines() if l.strip()]
            seat_starts = [e for e in events if e["kind"] == "agent.start" and e["agent"] == "SubAuthor.prover1"]
            self.assertTrue(seat_starts)
            self.assertEqual(seat_starts[0]["agent_path"], "ACWorkflow.Author.SubAuthor.prover1")
            self.assertEqual(seat_starts[0]["parent_call_id"], "lead01")
            waves = [e for e in events if e["kind"] == "ac.author.delegate.wave_done"]
            self.assertEqual(len(waves), 2)
            self.assertEqual(waves[0]["agent_path"], "ACWorkflow.Author")

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run):
            asyncio.run(scenario(td))

    def test_delegate_validates_arguments(self) -> None:
        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "max_tasks_per_wave": 2})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            self.assertIn("Error", fn(tasks=[]))
            self.assertIn("unknown role", fn(tasks=[{"role": "poet", "task": "x"}]))
            self.assertIn("at most 2 tasks", fn(tasks=[{"role": "prover", "task": "x"}] * 3))
            self.assertIn("unknown agent_id", fn(tasks=[{"role": "prover", "task": "x", "agent_id": "nope"}]))
            self.assertIn("no `task` text", fn(tasks=[{"role": "prover", "task": "  "}]))
            self.assertEqual(multi._waves_done, 0)

        with tempfile.TemporaryDirectory() as td:
            asyncio.run(scenario(td))

    def test_seat_failures_become_error_entries(self) -> None:
        async def slow_run(self, inp):
            if inp.role == "checker":
                raise RuntimeError("provider down")
            await asyncio.sleep(5)
            return self.Outputs(report="late")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "job_timeout_s": 0.2})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            out = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}, {"role": "checker", "task": "c"}])
            self.assertIn("(error: cancelled: wave deadline 0.2 s reached)", out)
            self.assertIn("(error: RuntimeError: provider down)", out)

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", slow_run):
            asyncio.run(scenario(td))

    def test_role_models_and_default_model_selection(self) -> None:
        models: list[str] = []

        async def fake_run(self, inp):
            models.append(str(self.MODEL))
            return self.Outputs(report="r")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "role_models": {"checker": "models/openai/gpt-54-mini"}}, model="models/openai/gpt-6-astra-max")
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}, {"role": "checker", "task": "c"}])

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run):
            asyncio.run(scenario(td))
        self.assertEqual(sorted(models), ["models/openai/gpt-54-mini", "models/openai/gpt-6-astra-max"])

    def test_duplicate_agent_id_in_wave_rejected(self) -> None:
        async def fake_run(self, inp):
            return self.Outputs(report="r", messages_after=[{"role": "assistant", "content": "r"}])

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "max_waves": 3})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}])
            out = await asyncio.to_thread(
                fn, tasks=[{"role": "prover", "task": "a", "agent_id": "prover1"}, {"role": "prover", "task": "b", "agent_id": "prover1"}]
            )
            self.assertIn("appears twice", out)
            self.assertEqual(multi._waves_done, 1)

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run):
            asyncio.run(scenario(td))

    def test_run_budget_exhaustion_cancels_siblings_and_stops_delegation(self) -> None:
        from proofstack.budget import BudgetExhausted

        cancelled = {"n": 0}

        async def fake_run(self, inp):
            if inp.role == "checker":
                await asyncio.sleep(0.05)
                raise BudgetExhausted("run", "usd", 10.0, 10.5)
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled["n"] += 1
                raise
            return self.Outputs(report="late")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            t0 = asyncio.get_running_loop().time()
            out = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}, {"role": "checker", "task": "c"}])
            self.assertLess(asyncio.get_running_loop().time() - t0, 3.0)
            self.assertIn("RUN BUDGET EXHAUSTED", out)
            self.assertIn("BudgetExhausted(run)", out)
            self.assertIn("(error: cancelled: run budget exhausted)", out)
            self.assertEqual(cancelled["n"], 1)
            out2 = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}])
            self.assertIn("budget is exhausted", out2)
            self.assertEqual(multi._waves_done, 1)

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run):
            asyncio.run(scenario(td))

    def test_wallclock_reserve_refuses_wave(self) -> None:
        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "synthesis_reserve_s": 1000})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            with patch.object(type(multi.tracker), "remaining_wallclock_s", lambda self: 900.0):
                fn, _ = multi._delegate_tool()
                out = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}])
            self.assertIn("Not enough wallclock", out)
            self.assertEqual(multi._waves_done, 0)

        with tempfile.TemporaryDirectory() as td:
            asyncio.run(scenario(td))

    def test_briefing_reaches_every_seat(self) -> None:
        seen: list[SubAuthorSeat.Inputs] = []

        async def fake_run(self, inp):
            seen.append(inp)
            return self.Outputs(report="r")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, desc = multi._delegate_tool()
            self.assertIn("briefing", desc["function"]["parameters"]["properties"])
            await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}, {"role": "explorer", "task": "e"}], briefing="shared B")

        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run):
            asyncio.run(scenario(td))
        self.assertEqual([s.briefing for s in seen], ["shared B", "shared B"])

    def test_truncation_keeps_caveats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            multi = MultiAuthor(_ctx(td, delegation={"enabled": True}), name="Author")
            ordered = "## Summary\ns\n## Caveats\nCAV\n## Findings\n" + "x" * 500
            out = multi._truncate_report(ordered, 100, "subagents/w.md")
            self.assertTrue(out.startswith("## Summary"))
            self.assertIn("truncated by the harness", out)
            self.assertIn("subagents/w.md", out)
            self.assertNotIn("x" * 200, out)
            misordered = "## Summary\ns\n## Findings\n" + "y" * 500 + "\n## Caveats\nIMPORTANT CAVEAT"
            out2 = multi._truncate_report(misordered, 100, "subagents/w.md")
            self.assertIn("IMPORTANT CAVEAT", out2)
            self.assertNotIn("y" * 200, out2)
            self.assertEqual(multi._truncate_report("short", 100, "a"), "short")

    def test_container_keepalive_during_wave(self) -> None:
        calls: list[str] = []

        class _Containers:
            def retrieve(self, cid):
                calls.append(cid)
                return types.SimpleNamespace(status="running")

        class _FakeOpenAI:
            def __init__(self, **kw):
                self.containers = _Containers()

        async def fake_run(self, inp):
            await asyncio.sleep(0.35)
            return self.Outputs(report="r")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "container_keepalive_s": 0.1, "sandbox_carry_over": False})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            fn, _ = multi._delegate_tool()
            conv = [{"type": "code_interpreter_call", "container_id": "cntr_live"}]
            await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}], messages=conv)
            events = [json.loads(l) for l in (Path(td) / "events.jsonl").read_text().splitlines() if l.strip()]
            ka = [e for e in events if e["kind"] == "ac.author.container_keepalive"]
            self.assertGreaterEqual(len(ka), 2)
            self.assertEqual(ka[0]["payload"]["container_id"], "cntr_live")
            self.assertEqual(ka[0]["payload"]["status"], "running")
            starts = [e for e in events if e["kind"] == "ac.author.delegate.wave_start"]
            self.assertEqual(starts[0]["payload"]["container_id"], "cntr_live")

        fake_mod = types.SimpleNamespace(OpenAI=_FakeOpenAI)
        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run), \
                patch.dict(sys.modules, {"openai": fake_mod}), patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            asyncio.run(scenario(td))
        self.assertGreaterEqual(len(calls), 2)
        self.assertLessEqual(len(calls), 4)

    def test_sandbox_carry_over_reattaches_written_files(self) -> None:
        created: list[tuple[str, bytes]] = []
        deleted: list[str] = []

        class _Content:
            def retrieve(self, file_id, container_id):
                return types.SimpleNamespace(read=lambda: {"cf1": b"\\documentclass{article}", "cf3": b"notes"}[file_id])

        class _ContainerFiles:
            content = _Content()

            def list(self, container_id):
                assert container_id == "cntr_old"
                return [
                    types.SimpleNamespace(id="cf1", path="/mnt/data/answer.tex", bytes=22),
                    types.SimpleNamespace(id="cf2", path="/mnt/data/file-abc-answer.tex", bytes=5),
                    types.SimpleNamespace(id="cf3", path="/mnt/data/research_notes.tex", bytes=5),
                    types.SimpleNamespace(id="cf4", path="/mnt/data/answer.pdf", bytes=5000),
                ]

        class _Files:
            def create(self, file, purpose):
                created.append(file)
                return types.SimpleNamespace(id=f"file-new{len(created)}")

            def delete(self, fid):
                deleted.append(fid)

        class _FakeOpenAI:
            def __init__(self, **kw):
                self.containers = types.SimpleNamespace(files=_ContainerFiles(), retrieve=lambda cid: types.SimpleNamespace(status="running"))
                self.files = _Files()

        async def fake_run(self, inp):
            return self.Outputs(report="r")

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True, "sandbox_carry_over": True, "container_keepalive_s": 0})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            # Like the real factory: load_solver_config deep-copies cfg, and the
            # APIClient keeps the copied descriptors in ``tool_descriptions``.
            def deep_copying_factory(cfg):
                copied = _safe_deepcopy(cfg)
                return types.SimpleNamespace(tool_descriptions=[d for _, d in copied["tools"]])

            ctx.api_client_factory = deep_copying_factory
            client = multi._build_api_client_with_file_ids(["file-orig1", "file-orig2", "file-orig3"])
            container = client.tool_descriptions[0]["container"]
            fn, _ = multi._delegate_tool()
            conv = [{"type": "code_interpreter_call", "container_id": "cntr_old"}]
            out = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}], messages=conv)
            self.assertIn("Sandbox note", out)
            self.assertIn("- answer.tex: `/mnt/data/file-new1-answer.tex`", out)
            self.assertIn("- research_notes.tex: `/mnt/data/file-new2-research_notes.tex`", out)
            self.assertNotIn("answer.pdf", out)
            self.assertEqual([f[0] for f in created], ["answer.tex", "research_notes.tex"])
            self.assertEqual(container["file_ids"], ["file-orig1", "file-orig2", "file-orig3", "file-new1", "file-new2"])
            # A second wave replaces the earlier copies rather than piling up.
            out2 = await asyncio.to_thread(fn, tasks=[{"role": "prover", "task": "p"}], messages=conv)
            self.assertIn("file-new3-answer.tex", out2)
            self.assertEqual(container["file_ids"], ["file-orig1", "file-orig2", "file-orig3", "file-new3", "file-new4"])
            events = [json.loads(l) for l in (Path(td) / "events.jsonl").read_text().splitlines() if l.strip()]
            co = [e for e in events if e["kind"] == "ac.author.sandbox_carry_over"]
            self.assertEqual(len(co), 2)
            self.assertIsNone(co[0]["payload"]["error"])
            # Query return must leave files available for the final download.
            async def fake_query(self_, client, messages, query, *, call_id=None):
                return (0, [], {})
            with patch.object(Author, "_query", fake_query):
                await multi._query(None, [], None)
            self.assertEqual(deleted, [])
            await multi._close_sandbox_carry()
            self.assertEqual(sorted(deleted), ["file-new1", "file-new2", "file-new3", "file-new4"])

        fake_mod = types.SimpleNamespace(OpenAI=_FakeOpenAI)
        with tempfile.TemporaryDirectory() as td, patch.object(SubAuthorSeat, "run", fake_run), \
                patch.dict(sys.modules, {"openai": fake_mod}), patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            asyncio.run(scenario(td))

    def test_snapshot_retries_transient_failures(self) -> None:
        import proofstack.agents.ac.multi_author as mod

        calls: list[int] = []

        def flaky(self_, container_id, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("Error code: 404")
            return [("answer.tex", "/mnt/data/file-x-answer.tex")]

        async def scenario(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            rec = await multi._snapshot_sandbox("cntr_old")
            self.assertIsNone(rec["error"])
            self.assertEqual(len(calls), 3)
            calls.clear()
            rec = await multi._snapshot_sandbox("cntr_old")
            self.assertIsNone(rec["error"])

        def always_fails(self_, container_id, **kwargs):
            calls.append(1)
            raise RuntimeError("Error code: 404")

        async def scenario_fail(td: str) -> None:
            ctx = _ctx(td, delegation={"enabled": True})
            multi = MultiAuthor(ctx, name="Author")
            multi._render_container_messages(_inp(), "listing")
            rec = await multi._snapshot_sandbox("cntr_old")
            self.assertEqual(rec["error"], "RuntimeError: Error code: 404")
            self.assertEqual(len(calls), mod._SNAPSHOT_ATTEMPTS)

        with tempfile.TemporaryDirectory() as td, patch.object(mod, "_SNAPSHOT_RETRY_S", 0.0):
            with patch.object(MultiAuthor, "_snapshot_sandbox_sync", flaky):
                asyncio.run(scenario(td))
            calls.clear()
            with patch.object(MultiAuthor, "_snapshot_sandbox_sync", always_fails):
                asyncio.run(scenario_fail(td))

    def test_carry_over_auto_follows_pro_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            captured: list[dict] = []
            for model, expect in (("models/openai/gpt-6-astra-pro", True), ("models/openai/gpt-6-astra-max", False)):
                ctx = _ctx(td, delegation={"enabled": True}, model=model)
                multi = MultiAuthor(ctx, name="Author")
                ctx.api_client_factory = lambda cfg: captured.append(cfg) or object()
                multi._build_api_client_with_file_ids(["f"])
                self.assertEqual(multi._carry_over_enabled(), expect, model)


class SubAuthorSeatTests(unittest.TestCase):
    def test_render_messages_by_role_and_followup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(td)
            seat = SubAuthorSeat(ctx, model_ref="models/openai/gpt-54-mini", name="SubAuthor.prover1")
            inp = SubAuthorSeat.Inputs(role="prover", task="Prove L.", problem="P", round=2, answer_tex="AAA")
            msgs = seat.render_messages(inp)
            self.assertEqual([m["role"] for m in msgs], ["developer", "user"])
            self.assertIn("Role: prover.", msgs[0]["content"])
            self.assertIn("## Summary", msgs[0]["content"])
            self.assertIn("AAA", msgs[1]["content"])
            self.assertIn("(round 2)", msgs[1]["content"])

            no_ws = seat.render_messages(inp.model_copy(update={"include_workspace": False}))
            self.assertNotIn("AAA", no_ws[1]["content"])

            out = seat.parse_output("<thought>hidden</thought>\n## Summary\nok", inp)
            self.assertEqual(out.report, "## Summary\nok")
            self.assertEqual(out.messages_after[-1], {"role": "assistant", "content": "<thought>hidden</thought>\n## Summary\nok"})

            follow = seat.render_messages(SubAuthorSeat.Inputs(role="prover", task="Fix.", problem="P", prior_messages=out.messages_after))
            self.assertEqual(len(follow), len(out.messages_after) + 1)
            self.assertIn("Follow-up from the lead", follow[-1]["content"])

    def test_all_roles_have_instructions(self) -> None:
        from proofstack.agents.ac.multi_author import ROLE_INSTRUCTIONS

        self.assertEqual(set(ROLE_INSTRUCTIONS), set(ROLE_DESCRIPTIONS))
        self.assertEqual(DELEGATION_DEFAULTS["roles"], ["explorer", "prover", "checker"])
        self.assertTrue(set(DELEGATION_DEFAULTS["roles"]) <= set(ROLE_DESCRIPTIONS))

    def test_briefing_section_only_when_given(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            seat = SubAuthorSeat(_ctx(td), name="SubAuthor.x")
            base = dict(role="checker", task="t", problem="P")
            self.assertNotIn("Shared briefing", seat.render_messages(SubAuthorSeat.Inputs(**base))[1]["content"])
            with_b = seat.render_messages(SubAuthorSeat.Inputs(**base, briefing="NOTATION X"))[1]["content"]
            self.assertIn("Shared briefing from the lead", with_b)
            self.assertIn("NOTATION X", with_b)


class FindContainerIdTests(unittest.TestCase):
    def test_returns_last_container(self) -> None:
        conv = [
            {"type": "code_interpreter_call", "container_id": "cntr_a"},
            {"type": "function_call", "name": "delegate"},
            {"type": "code_interpreter_call", "container_id": "cntr_b"},
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(find_container_id(conv), "cntr_b")
        self.assertIsNone(find_container_id([{"role": "assistant", "content": "x"}]))


if __name__ == "__main__":
    unittest.main()
