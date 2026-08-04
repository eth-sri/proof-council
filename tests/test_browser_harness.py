from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.ac.author import Author  # noqa: E402
from proofstack.agents.ac.blocks import parse_author_output  # noqa: E402
from proofstack.agents.ac.critic import ACCritic  # noqa: E402
from proofstack.agents.human_agent import wait_for_response_file  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.harness.browser_call import (  # noqa: E402
    commit_operator_comments,
    peek_operator_comments,
    task_stem,
)
from proofstack.kinds.api_call import (  # noqa: E402
    APICallAgent,
    _flatten_messages_for_harness,
)

BROWSER_MODEL = {
    "api": "browser",
    "model": "fake-browser-model",
    "display_model": "Fake Browser",
    "service": "chatgpt",
    "chat_url": "https://chatgpt.com/",
    "settings_hint": "pick Fake",
    "expected_model_slugs": ["fake"],
    "instruction_addendum": "PRINT FILES INLINE.",
}


class EchoAgent(APICallAgent):
    SYSTEM_PROMPT = "SYS PROMPT"
    USER_PROMPT = "USER: {problem}"

    class Inputs(BaseModel):
        problem: str

    class Outputs(BaseModel):
        text: str = ""


class AddendumAgent(EchoAgent):
    def render_harness_addendum(self, inp, *, task_token: str, default: str) -> str:
        return f"DOWNLOAD AS answer___{task_token}.tex (default was: {default})"


async def _fast_wait(path, **kwargs):
    kwargs["poll_interval_s"] = 0.01
    return await wait_for_response_file(path, **kwargs)


def _ctx(tmp: str, harness_dir: str) -> RunContext:
    os.environ["PROOFCOUNCIL_HARNESS_DIR"] = harness_dir
    return RunContext.create(
        run_id="test-run",
        root_workdir=tmp,
        flat=True,
        component_configs={"echo": {"model": BROWSER_MODEL}},
    )


class FlattenTests(unittest.TestCase):
    def test_system_user_pair_concatenates(self) -> None:
        text = _flatten_messages_for_harness(
            [
                {"role": "developer", "content": "SYS"},
                {"role": "user", "content": "USR"},
            ]
        )
        self.assertEqual(text, "SYS\n\nUSR")
        self.assertNotIn("###", text)

    def test_longer_conversations_get_role_labels(self) -> None:
        text = _flatten_messages_for_harness(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": [{"type": "output_text", "text": "B"}]},
                {"role": "user", "content": "C"},
            ]
        )
        self.assertIn("### user ###\nA", text)
        self.assertIn("### assistant ###\nB", text)


class AuthorPacketTests(unittest.TestCase):
    def _author(self, tmp: str) -> Author:
        ctx = RunContext.create(run_id="t", root_workdir=tmp, flat=True)
        return Author(ctx)

    def test_round0_omits_empty_canonical_files(self) -> None:
        # ChatGPT rejects zero-byte uploads ("Something went wrong"), so
        # empty round-0 files are noted in the instruction, not attached.
        with tempfile.TemporaryDirectory() as tmp:
            author = self._author(tmp)
            instruction, attachments = author.render_harness_packet(
                Author.Inputs(problem="P?", round=0, n_rounds=3)
            )
        # The round-0 template has no per-file placeholders — it already
        # instructs the model to produce all three files from scratch — so
        # only the attachment suppression is observable here.
        self.assertEqual(attachments, {})
        self.assertIn("P?", instruction)

    def test_loop_round_moves_files_to_attachments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author = self._author(tmp)
            zip_src = Path(tmp) / "compute.zip"
            with zipfile.ZipFile(zip_src, "w") as zf:
                zf.writestr("notes.txt", "hi")
            instruction, attachments = author.render_harness_packet(
                Author.Inputs(
                    problem="P?",
                    round=2,
                    n_rounds=3,
                    answer_tex="ANSWER BODY",
                    references_bib="BIB BODY",
                    compute_zip_path=zip_src,
                )
            )
        self.assertEqual(attachments["answer.tex"], "ANSWER BODY")
        self.assertEqual(attachments["references.bib"], "BIB BODY")
        self.assertNotIn("research_notes.tex", attachments)
        self.assertIsInstance(attachments["compute_workspace.zip"], bytes)
        self.assertIn("(attached as answer.tex)", instruction)
        self.assertIn("(attached as references.bib)", instruction)
        self.assertIn(
            "(research_notes.tex is currently empty; no attachment)", instruction
        )
        self.assertNotIn("ANSWER BODY", instruction)

    def test_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author = self._author(tmp)
            expected = author.harness_expectations(
                Author.Inputs(problem="P?", round=1, n_rounds=3)
            )
            critic = ACCritic(author.ctx)
            critic_expected = critic.harness_expectations(
                ACCritic.Inputs(problem="P?")
            )
        self.assertIn("answer.tex", expected["fenced_files"])
        self.assertEqual(critic_expected, {"markers": ["<answer_ready>"]})


class BrowserCallEndToEndTests(unittest.TestCase):
    def test_prewritten_response_consumed_and_events_emitted(self) -> None:
        # A response written before the node runs (crash-resume scenario)
        # is picked up immediately thanks to the deterministic stem.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = EchoAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(EchoAgent.Inputs(problem='P')))
            inbox = ctx.root_workdir / "human_inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / f"{stem}.response.json").write_text(
                json.dumps(
                    {
                        "status": "done",
                        "transport": "manual",
                        "assistant_text": "HELLO",
                        "operator_comments": "steer left",
                    }
                ),
                encoding="utf-8",
            )
            out = asyncio.run(agent(problem="P"))

            self.assertEqual(out.text, "HELLO")
            task = json.loads((inbox / f"{stem}.task.json").read_text(encoding="utf-8"))
            self.assertEqual(task["type"], "browser_call")
            self.assertEqual(task["browser"]["display_model"], "Fake Browser")
            self.assertEqual(task["task_token"], stem)
            self.assertEqual(task["instruction_file"], f"instruction_{stem}.txt")
            instruction = (
                inbox / f"{stem}.packet" / f"instruction_{stem}.txt"
            ).read_text(encoding="utf-8")
            self.assertIn(f"[ProofCouncil task {stem}]", instruction)
            self.assertIn("SYS PROMPT", instruction)
            self.assertIn("USER: P", instruction)
            self.assertIn("PRINT FILES INLINE.", instruction)
            self.assertTrue((inbox / f"{stem}.packet.zip").exists())
            # The harness_packages mirror is cleaned up after consumption.
            self.assertFalse((Path(hdir) / "test-run" / stem).exists())

            events = [
                json.loads(line)
                for line in (ctx.root_workdir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            kinds = [e.get("kind") for e in events]
            self.assertIn("human.waiting", kinds)
            self.assertIn("human.submitted", kinds)
            call = next(e for e in events if e.get("kind") == "model.call")
            self.assertEqual(call["payload"]["cost_usd"], 0.0)
            self.assertEqual(call["payload"]["via"], "browser_harness")

            comments, offset = peek_operator_comments(ctx.root_workdir)
            self.assertIn("steer left", comments)
            commit_operator_comments(ctx.root_workdir, offset)
            self.assertEqual(peek_operator_comments(ctx.root_workdir)[0], "")

    def test_addendum_override_hook_replaces_config_addendum(self) -> None:
        # A code-level override must win over the YAML addendum (editing the
        # YAML mid-run would re-key pending tasks) and receive the stem.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = AddendumAgent(ctx, name="echo")
            stem = task_stem(
                ctx.run_id, "echo", agent._cache_key(AddendumAgent.Inputs(problem="P"))
            )
            inbox = ctx.root_workdir / "human_inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / f"{stem}.response.json").write_text(
                json.dumps({"status": "done", "assistant_text": "HELLO"}),
                encoding="utf-8",
            )
            asyncio.run(agent(problem="P"))
            instruction = (
                inbox / f"{stem}.packet" / f"instruction_{stem}.txt"
            ).read_text(encoding="utf-8")
        self.assertIn(f"DOWNLOAD AS answer___{stem}.tex", instruction)
        self.assertIn("(default was: PRINT FILES INLINE.)", instruction)
        self.assertNotIn("\nPRINT FILES INLINE.", instruction)

    def test_token_tagged_uploads_merge_under_canonical_name(self) -> None:
        # Auto-downloaded/hand-uploaded files arrive under token-tagged
        # names; the fenced merge must map them back to answer.tex.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = EchoAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(EchoAgent.Inputs(problem="P")))
            inbox = ctx.root_workdir / "human_inbox"
            uploads = inbox / f"{stem}.uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            tagged = f"answer___{stem}.tex"
            (uploads / tagged).write_text("TAGGED BODY", encoding="utf-8")
            (inbox / f"{stem}.response.json").write_text(
                json.dumps(
                    {
                        "status": "done",
                        "assistant_text": "ok",
                        "uploaded_files": {
                            tagged: f"human_inbox/{stem}.uploads/{tagged}"
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = asyncio.run(agent(problem="P"))
        parsed = parse_author_output(out.text)
        self.assertEqual(parsed.files["answer.tex"], "TAGGED BODY")

    def test_foreign_tagged_fence_never_becomes_canonical(self) -> None:
        # Blocker: a confirmed-past-warning foreign file must STILL not
        # parse into this task's answer.tex.
        text = "```file path=answer___tokB.tex\nSTALE FOREIGN\n```"
        parsed = parse_author_output(text, task_token="tokA")
        self.assertNotIn("answer.tex", parsed.files)
        self.assertTrue(
            any("different task" in w for w in parsed.parse_warnings)
        )
        # Own-token tags canonicalize normally.
        parsed_own = parse_author_output(
            "```file path=answer___tokA.tex\nMINE\n```", task_token="tokA"
        )
        self.assertEqual(parsed_own.files["answer.tex"], "MINE")
        # Tokenless parsing (non-harness paths) keeps the legacy stripping.
        parsed_no_token = parse_author_output(text)
        self.assertEqual(parsed_no_token.files["answer.tex"], "STALE FOREIGN")

    def test_foreign_tagged_upload_stays_tagged_in_merged_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = EchoAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(EchoAgent.Inputs(problem='P')))
            inbox = ctx.root_workdir / "human_inbox"
            uploads = inbox / f"{stem}.uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            foreign = "answer___Author__deadbeef0000.tex"
            (uploads / foreign).write_text("FOREIGN BODY", encoding="utf-8")
            (inbox / f"{stem}.response.json").write_text(
                json.dumps(
                    {
                        "status": "done",
                        "assistant_text": "ok",
                        "uploaded_files": {
                            foreign: f"human_inbox/{stem}.uploads/{foreign}"
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = asyncio.run(agent(problem="P"))
        self.assertIn(f"path={foreign}", out.text)
        parsed = parse_author_output(out.text, task_token=stem)
        self.assertNotIn("answer.tex", parsed.files)

    def test_second_call_replays_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = EchoAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(EchoAgent.Inputs(problem='P')))
            inbox = ctx.root_workdir / "human_inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / f"{stem}.response.json").write_text(
                json.dumps({"status": "done", "assistant_text": "HELLO"}),
                encoding="utf-8",
            )
            out1 = asyncio.run(agent(problem="P"))
            out2 = asyncio.run(EchoAgent(ctx, name="echo")(problem="P"))
            self.assertEqual(out1.text, out2.text)
            events_text = (ctx.root_workdir / "events.jsonl").read_text(encoding="utf-8")
            self.assertEqual(events_text.count('"human.waiting"'), 1)
            self.assertIn("agent.cache_hit", events_text)

    def test_uploaded_files_appended_and_win_over_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            agent = EchoAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(EchoAgent.Inputs(problem='P')))
            inbox = ctx.root_workdir / "human_inbox"
            uploads = inbox / f"{stem}.uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            (uploads / "answer.tex").write_text("UPLOADED BODY", encoding="utf-8")
            (inbox / f"{stem}.response.json").write_text(
                json.dumps(
                    {
                        "status": "done",
                        "assistant_text": "```file path=answer.tex\nINLINE BODY\n```",
                        "uploaded_files": {
                            "answer.tex": f"human_inbox/{stem}.uploads/answer.tex"
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = asyncio.run(agent(problem="P"))
            parsed = parse_author_output(out.text)
            self.assertEqual(parsed.files["answer.tex"], "UPLOADED BODY")
            self.assertTrue(
                any("duplicate" in w for w in parsed.parse_warnings)
            )

    def test_late_response_via_wait_loop(self) -> None:
        async def scenario(ctx: RunContext) -> object:
            agent = EchoAgent(ctx, name="echo")
            task = asyncio.create_task(agent(problem="P"))
            inbox = ctx.root_workdir / "human_inbox"
            response_path = None
            for _ in range(400):
                found = list(inbox.glob("*.task.json")) if inbox.exists() else []
                if found:
                    payload = json.loads(found[0].read_text(encoding="utf-8"))
                    response_path = Path(payload["response_path"])
                    break
                await asyncio.sleep(0.005)
            assert response_path is not None
            response_path.write_text(
                json.dumps({"status": "done", "assistant_text": "LATE"}),
                encoding="utf-8",
            )
            return await asyncio.wait_for(task, timeout=5.0)

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            with mock.patch(
                "proofstack.harness.browser_call.wait_for_response_file", _fast_wait
            ):
                out = asyncio.run(scenario(ctx))
        self.assertEqual(out.text, "LATE")


class PauseAccountingTests(unittest.TestCase):
    def test_parallel_waiters_credit_union_not_sum(self) -> None:
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        child1 = BudgetTracker(scope="a", parent=root)
        child2 = BudgetTracker(scope="b", parent=root)
        child1.begin_pause(now=100.0)
        child2.begin_pause(now=105.0)
        child1.end_pause(now=110.0)
        child2.end_pause(now=115.0)
        self.assertAlmostEqual(child1.counters.paused_s, 10.0)
        self.assertAlmostEqual(child2.counters.paused_s, 10.0)
        # Root sees the union [100, 115], not 20 summed seconds.
        self.assertAlmostEqual(root.counters.paused_s, 15.0)

    def test_disjoint_wait_episodes_sum(self) -> None:
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        root.begin_pause(now=10.0)
        root.end_pause(now=20.0)
        root.begin_pause(now=30.0)
        root.end_pause(now=35.0)
        self.assertAlmostEqual(root.counters.paused_s, 15.0)

    def test_nested_wait_fully_contained_counts_once(self) -> None:
        # Analog of the earlier high-water counterexample: a long wait
        # containing a shorter one counts the full outer span exactly once.
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        root.begin_pause(now=100.0)  # long waiter
        root.begin_pause(now=105.0)  # short waiter, contained
        root.end_pause(now=110.0)
        root.end_pause(now=120.0)
        self.assertAlmostEqual(root.counters.paused_s, 20.0)

    def test_state_is_constant_size(self) -> None:
        # Reference counting keeps O(1) state per node: no per-poll history.
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        for i in range(10_000):
            root.begin_pause(now=float(i))
            root.end_pause(now=float(i) + 0.5)
        self.assertAlmostEqual(root.counters.paused_s, 5000.0)
        self.assertEqual(root.counters.active_pauses, 0)

    def test_ongoing_pause_excluded_from_wallclock(self) -> None:
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        start = root.counters.started_at
        root.begin_pause(now=start + 5.0)
        # 100s in, with a pause running since t=5: only 5s count as compute.
        self.assertAlmostEqual(
            root.counters.wallclock_s(now=start + 100.0), 5.0
        )
        root.end_pause(now=start + 100.0)
        self.assertAlmostEqual(
            root.counters.wallclock_s(now=start + 100.0), 5.0
        )

    def test_unbalanced_end_pause_never_touches_the_parent(self) -> None:
        # A double-end on one child must not decrement the parent's
        # reference count while a sibling's pause is active — that would
        # end the sibling's span early and undercount the union.
        from proofstack.budget import BudgetTracker

        root = BudgetTracker(scope="run")
        a = BudgetTracker(scope="a", parent=root)
        b = BudgetTracker(scope="b", parent=root)
        a.begin_pause(now=0.0)
        a.end_pause(now=4.0)
        b.begin_pause(now=10.0)
        a.end_pause(now=11.0)  # unbalanced: must be a no-op everywhere
        b.end_pause(now=20.0)
        self.assertAlmostEqual(root.counters.paused_s, 14.0)
        self.assertEqual(root.counters.active_pauses, 0)


class TaskStemScopingTests(unittest.TestCase):
    def test_stem_is_stable_within_a_run_but_distinct_across_runs(self) -> None:
        stem1 = task_stem("run-a", "echo", "cachekey123")
        stem2 = task_stem("run-a", "echo", "cachekey123")
        stem3 = task_stem("run-b", "echo", "cachekey123")
        self.assertEqual(stem1, stem2)
        self.assertNotEqual(stem1, stem3)
        self.assertTrue(stem1.startswith("echo__"))

    def test_unsafe_agent_names_are_sanitized(self) -> None:
        stem = task_stem("run-a", "week/../ird name", "k")
        self.assertNotIn("/", stem)
        self.assertNotIn(" ", stem)


class RejectedResponseTests(unittest.TestCase):
    def test_unparseable_response_requeues_and_consumes_correction(self) -> None:
        # First submission fails parse -> quarantined, task re-waits; a
        # corrected submission then completes the call. The run never dies.
        class PickyAgent(EchoAgent):
            def parse_output(self, raw_text, inp):
                if "BAD" in raw_text:
                    raise ValueError("cannot parse this")
                return super().parse_output(raw_text, inp)

        async def scenario(ctx: RunContext):
            agent = PickyAgent(ctx, name="echo")
            stem = task_stem(ctx.run_id, "echo", agent._cache_key(PickyAgent.Inputs(problem='P')))
            inbox = ctx.root_workdir / "human_inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            response_path = inbox / f"{stem}.response.json"
            response_path.write_text(
                json.dumps({"status": "done", "assistant_text": "BAD ANSWER"}),
                encoding="utf-8",
            )
            task = asyncio.create_task(agent(problem="P"))
            for _ in range(800):
                if (inbox / f"{stem}.response.rejected-1.json").exists():
                    break
                await asyncio.sleep(0.005)
            response_path.write_text(
                json.dumps({"status": "done", "assistant_text": "GOOD ANSWER"}),
                encoding="utf-8",
            )
            out = await asyncio.wait_for(task, timeout=5.0)
            return out, stem, inbox

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as hdir:
            ctx = _ctx(tmp, hdir)
            with mock.patch(
                "proofstack.harness.browser_call.wait_for_response_file", _fast_wait
            ):
                out, stem, inbox = asyncio.run(scenario(ctx))
            self.assertEqual(out.text, "GOOD ANSWER")
            self.assertTrue((inbox / f"{stem}.response.rejected-1.json").exists())
            events = (ctx.root_workdir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("harness.response_rejected", events)
            self.assertIn("rejected_error", events)
            # waiting was re-emitted after the rejection.
            self.assertEqual(events.count('"human.waiting"'), 2)


class MalformedResponseTimeoutTests(unittest.TestCase):
    def test_permanently_malformed_response_still_times_out(self) -> None:
        # A broken response file must fall through to the timeout/pause
        # path, not spin forever on the fast retry branch.
        class TrackerStub:
            def begin_pause(self) -> None:
                pass

            def end_pause(self) -> None:
                pass

        class EventsStub:
            async def emit(self, *a: object, **k: object) -> None:
                pass

        async def scenario(tmp: str):
            path = Path(tmp) / "x.response.json"
            path.write_text("{not json", encoding="utf-8")
            return await wait_for_response_file(
                path,
                events=EventsStub(),
                tracker=TrackerStub(),
                timeout_s=0.05,
                poll_interval_s=0.01,
            )

        with tempfile.TemporaryDirectory() as tmp:
            out = asyncio.run(asyncio.wait_for(scenario(tmp), timeout=5.0))
        self.assertIsNone(out)


class CacheKeyTransportTests(unittest.TestCase):
    def test_edited_model_config_changes_cache_key(self) -> None:
        # The full parsed effective config is hashed: changing any resolved
        # value (here the instruction addendum) invalidates the cache key
        # even when the model ref itself is unchanged.
        cfg_v1 = {"api": "browser", "model": "m", "instruction_addendum": "A"}
        cfg_v2 = {"api": "browser", "model": "m", "instruction_addendum": "B"}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(run_id="t", root_workdir=tmp, flat=True)
            agent = EchoAgent(ctx, name="echo")
            with mock.patch("mathagents.load_solver_config", return_value=cfg_v1):
                key1 = agent._cache_key(EchoAgent.Inputs(problem="P"))
            with mock.patch("mathagents.load_solver_config", return_value=cfg_v2):
                key2 = agent._cache_key(EchoAgent.Inputs(problem="P"))
        self.assertNotEqual(key1, key2)

    def test_model_override_to_browser_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx_api = RunContext.create(run_id="t1", root_workdir=tmp, flat=True)
            ctx_browser = RunContext.create(
                run_id="t2",
                root_workdir=tmp,
                flat=True,
                model_overrides={"echo": BROWSER_MODEL},
            )
            key_api = EchoAgent(ctx_api, name="echo")._cache_key(
                EchoAgent.Inputs(problem="P")
            )
            key_browser = EchoAgent(ctx_browser, name="echo")._cache_key(
                EchoAgent.Inputs(problem="P")
            )
        self.assertNotEqual(key_api, key_browser)


class CouncilAllowlistTests(unittest.TestCase):
    def _workflow(self, tmp: str):
        from proofstack.agents.ac.ac_workflow import ACWorkflow

        ctx = RunContext.create(run_id="t", root_workdir=tmp, flat=True)
        return ACWorkflow(ctx)

    def test_author_cannot_authorize_models_outside_allowlist(self) -> None:
        allowed = ["models/browser/chatgpt-gpt56-sol-pro", "models/browser/claude-opus"]
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._workflow(tmp)
            used = asyncio.run(
                wf._council_member_models(
                    round=1,
                    requested=["models/openai/gpt-56-sol-pro", "claude-opus"],
                    allowed=allowed,
                )
            )
        self.assertEqual(used, ["models/browser/claude-opus"])

    def test_no_match_falls_back_to_full_allowlist(self) -> None:
        allowed = ["models/browser/claude-opus"]
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._workflow(tmp)
            used = asyncio.run(
                wf._council_member_models(
                    round=1, requested=["models/openai/o5"], allowed=allowed
                )
            )
        self.assertEqual(used, allowed)

    def test_empty_request_uses_configured_list(self) -> None:
        allowed = ["models/browser/claude-opus", "models/browser/gemini-pro"]
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._workflow(tmp)
            used = asyncio.run(
                wf._council_member_models(round=0, requested=[], allowed=allowed)
            )
        self.assertEqual(used, allowed)


class OperatorCommentRoutingTests(unittest.TestCase):
    def test_comments_fold_into_author_feedback_and_commit_after_success(self) -> None:
        from proofstack.agents.ac.ac_workflow import ACWorkflow

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(run_id="t", root_workdir=tmp, flat=True)
            wf = ACWorkflow(ctx)
            comments_path = ctx.root_workdir / "harness" / "operator_comments.jsonl"
            comments_path.parent.mkdir(parents=True, exist_ok=True)
            comments_path.write_text(
                json.dumps({"agent": "Author", "stem": "s1", "text": "try the dual"})
                + "\n",
                encoding="utf-8",
            )
            calls: list[dict] = []

            async def fake_author(**kwargs):
                calls.append(kwargs)
                return "AUTHOR_OUT"

            wf.author = fake_author
            out = asyncio.run(
                wf._call_author_with_operator_comments(workflow_feedback="compile ok")
            )
            self.assertEqual(out, "AUTHOR_OUT")
            self.assertIn("try the dual", calls[0]["workflow_feedback"])
            self.assertIn("compile ok", calls[0]["workflow_feedback"])
            # Committed: a second Author round no longer sees the comment.
            calls.clear()
            asyncio.run(wf._call_author_with_operator_comments(workflow_feedback=""))
            self.assertNotIn("try the dual", str(calls[0].get("workflow_feedback")))

    def test_visual_author_block_routes_through_comment_helper(self) -> None:
        import inspect

        from proofstack.agents.ac import visual_blocks

        src = inspect.getsource(visual_blocks.ACAuthorBlock.run)
        self.assertIn("_call_author_with_operator_comments", src)


if __name__ == "__main__":
    unittest.main()
