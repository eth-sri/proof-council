"""Offline tests of file-backed research notes; no provider or model calls."""
import asyncio
import hashlib
import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import active_trace
from proofstack.agents.ac.critic import ACCritic
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext


@pytest.fixture
def runner():
    with asyncio.Runner() as value:
        yield value


@pytest.fixture(params=["explicit", "auto"])
def harness(tmp_path, monkeypatch, request):
    uploads, calls, clients, sdk_options, downloads = [], [], [], [], []
    containers, touches, deleted = [], [], []
    control = SimpleNamespace(upload_error=None, query_error=None, evidence=True, logs=True,
                              artifact=False, artifact_error=None, receipt_override=None,
                              poll_evidence=False, expire_after_poll=False,
                              reply="<research_notes_status>verified</research_notes_status>\n"
                                    "<answer_ready>true</answer_ready>")

    class SDK:
        def __init__(self, **kwargs):
            sdk_options.append(kwargs)
            self.files = self
            self.containers = SimpleNamespace(files=SimpleNamespace(create=self.upload_container_file,
                list=self.list, content=SimpleNamespace(with_streaming_response=self)),
                create=self.create_container, retrieve=self.touch, delete=self.delete)
            self.content = SimpleNamespace(with_streaming_response=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, *, file, purpose, extra_body):
            if control.upload_error:
                raise control.upload_error
            uploads.append({"name": Path(file.name).name, "data": file.read(),
                            "purpose": purpose, **extra_body})
            return SimpleNamespace(id=f"file-test{len(uploads)}")

        def create_container(self, *, name, expires_after, timeout):
            container_id = "container-test" + (str(len(containers) + 1) if containers else "")
            containers.append({"id": container_id, "name": name, "expires_after": expires_after})
            return SimpleNamespace(id=container_id)

        def upload_container_file(self, container_id, *, file, timeout):
            if control.upload_error:
                raise control.upload_error
            uploads.append({"name": file[0], "data": file[1].read(), "container_id": container_id})
            return SimpleNamespace(id=f"file-test{len(uploads)}")

        def touch(self, container_id):
            touches.append(container_id)
            return SimpleNamespace(status="running")

        def delete(self, container_id):
            deleted.append(container_id)

        def list(self, container_id, **kwargs):
            if control.artifact_error:
                raise control.artifact_error
            data = []
            if control.artifact:
                data = [SimpleNamespace(id="cfile-test", container_id=container_id, bytes=None,
                    path=f"/mnt/data/{uploads[-1]['name']}.verification.json")]
            return SimpleNamespace(data=data, has_more=False)

        def retrieve(self, file_id, *, container_id, timeout):
            downloads.append({"file_id": file_id, "container_id": container_id, "timeout": timeout})
            return self

        def iter_bytes(self, chunk_size):
            upload = uploads[-1]
            content = control.receipt_override
            if content is None:
                content = json.dumps({"filename": upload["name"], "bytes": len(upload["data"]),
                                      "sha256": hashlib.sha256(upload["data"]).hexdigest(),
                                      "receipt_sha256": hashlib.sha256(b"receipt:" + upload["data"]).hexdigest()})
            yield content.encode()

    def query(self, messages, **kwargs):
        reasoning = self.kwargs.get("reasoning") or {}
        if reasoning.get("mode") == "pro":
            assert not any(tool.get("type") == "code_interpreter" and isinstance(tool.get("container"), str)
                           for tool in self.tool_descriptions), "Pro rejects explicit container IDs"
        calls.append({"messages": messages, "tools": self.tool_descriptions, "reasoning": reasoning})
        if control.query_error:
            error, control.query_error = control.query_error, None
            raise error
        evidence = []
        if uploads and control.evidence:
            upload = uploads[-1]
            inp = ACCritic.Inputs(problem="", round=int(upload["name"].split("-r")[1].split("-")[0]),
                                  research_notes_tex=upload["data"].decode())
            critic = ACCritic(ctx)
            metadata = critic._notes_metadata(inp)
            evidence = [{"type": "code_interpreter_call", "id": "ci-test", "status": "completed",
                         "container_id": upload.get("container_id", "container-test"), "code": critic._notes_verification_code(inp),
                         "outputs": ([{"type": "logs", "logs": json.dumps({k: metadata[k] for k in ("filename", "bytes", "sha256")})}]
                                     if control.logs else [])}]
        trace = active_trace.get()
        if trace is not None:
            if control.poll_evidence:
                trace.response(self, "synthetic", 0, {"id": f"response-{len(calls)}",
                    "status": "in_progress", "output": evidence})
                if control.expire_after_poll:
                    assert len(downloads) == 1
                    control.artifact_error = RuntimeError("Container expired during later reasoning")
            trace.update(("synthetic", 0), response_id=f"response-{len(calls)}", status="completed", outcome="response",
                         cost=0.25, usage_unavailable=False, code_interpreter_calls=evidence)
            trace._retain_completed_report({"id": f"response-{len(calls)}", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": control.reply}]}]})
        yield (0, messages[0] + [{"role": "assistant", "content": control.reply}],
               {"cost": 0.25, "input_tokens": 100, "output_tokens": 20})

    def factory(config):
        client = APIClient(**config)
        clients.append(client)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("mathagents.api_client.OpenAI", SDK)
    monkeypatch.setattr(APIClient, "run_queries", query)
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=factory,
        component_configs={"ACCritic": {"research_notes_container": "explicit", "model": "models/openai/gpt-6-astra-max"}}
        if request.param == "explicit" else {})
    return SimpleNamespace(ctx=ctx, uploads=uploads, calls=calls, clients=clients,
                           sdk_options=sdk_options, control=control, downloads=downloads,
                           containers=containers, touches=touches, deleted=deleted, mode=request.param)


def packet(**updates):
    return {"problem": "Synthetic question", "answer_tex": "The complete proof",
            "references_bib": "Complete bibliography", "research_notes_tex": "PRIVATE_SCRATCHPAD_CONTENT",
            **updates}


def previous_receipt_checker(critic, inp):
    # Freeze the checker emitted before verification version 3, independently
    # of the current generator so replay tests detect compatibility regressions.
    name = critic._notes_metadata(inp)["filename"]
    receipt = f"/mnt/data/{name}.verification.json"
    return (
        "from pathlib import Path\n"
        f"Path({receipt!r}).unlink(missing_ok=True)\n"
        "import hashlib, json\nfrom pathlib import Path\n"
        f"name = {name!r}\n"
        "paths = [p for p in Path('/mnt/data').rglob('*') if p.is_file() and p.name.endswith(name)]\n"
        "assert len(paths) == 1, 'Current notes file missing or ambiguous'\n"
        "data = paths[0].read_bytes()\n"
        "print(json.dumps({'filename': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))\n"
        f"Path({receipt!r}).write_text("
        "json.dumps({'filename': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}), "
        "encoding='utf-8')"
    )


@pytest.mark.parametrize("mode", [None, "auto", "explicit"])
@pytest.mark.parametrize("model,pro", [
    ("models/openai/gpt-6-astra-pro", True),
    ("models/openai/gpt-6-astra-max", False),
    ({"base": "models/openai/gpt-6-astra-max", "reasoning": {"mode": "pro"}}, True),
])
def test_container_mode_uses_effective_reasoning_config(tmp_path, mode, model, pro):
    config = {"model": model}
    if mode is not None:
        config["research_notes_container"] = mode
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs={"ACCritic": config})
    critic = ACCritic(ctx)
    if mode == "explicit" and pro:
        with pytest.raises(ValueError, match="reasoning.mode=pro; use auto"):
            critic._notes_container_mode()
    else:
        assert critic._notes_container_mode() == (mode or "auto")


def test_pro_explicit_container_fails_before_any_provider_io(harness, runner):
    harness.ctx.component_configs["ACCritic"] = {"research_notes_container": "explicit"}
    with pytest.raises(ValueError, match="reasoning.mode=pro; use auto"):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.uploads and not harness.containers and not harness.calls and not harness.sdk_options


def test_default_pro_auto_container_keeps_pro_reasoning_and_checker_last(harness, runner):
    harness.ctx.component_configs.clear()
    harness.control.logs, harness.control.artifact = False, True
    result = runner.run(ACCritic(harness.ctx)(**packet()))
    assert result.answer_ready and result.research_notes_execution_verified
    assert len(harness.calls) == 1 and harness.calls[0]["reasoning"]["mode"] == "pro"
    assert harness.calls[0]["tools"][0]["container"] == {"type": "auto", "file_ids": ["file-test1"]}
    assert not harness.containers and not harness.touches
    assert "last tool call, immediately before" in json.dumps(harness.calls[0]["messages"])
    assert harness.clients[0].max_hosted_tool_calls == 30


@pytest.mark.parametrize("limit", [2, 30, 48])
def test_hosted_allowance_is_configurable_and_separate_from_local_limit(tmp_path, limit):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            component_configs={"ACCritic": {"max_hosted_tool_calls": limit}})
    critic = ACCritic(ctx)
    assert critic.extra_client_kwargs()["max_hosted_tool_calls"] == limit
    assert critic.MAX_TOOL_CALLS == 12


@pytest.mark.parametrize("limit", [0, 1, -1, True, "30", 2.5, None])
def test_invalid_hosted_allowance_fails_before_provider_io(tmp_path, limit):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            component_configs={"ACCritic": {"max_hosted_tool_calls": limit}})
    with pytest.raises(ValueError, match="max_hosted_tool_calls"):
        ACCritic(ctx).extra_client_kwargs()


@pytest.mark.parametrize("over_budget", [False, True])
def test_explicit_keepalive_spans_silent_reasoning_and_final_download(harness, runner, monkeypatch, over_budget):
    harness.ctx.component_configs["ACCritic"] = {"research_notes_container": "explicit", "model": "models/openai/gpt-6-astra-max"}
    harness.control.logs, harness.control.artifact = False, True
    if over_budget:
        harness.ctx.budgets.root().spec = BudgetSpec(max_usd=.2)
    monkeypatch.setattr(ACCritic, "NOTES_KEEPALIVE_S", .005)
    query_touch, receipt_touch = threading.Event(), threading.Event()
    phase = ["query"]
    original_query, original_read = APIClient.run_queries, APIClient.read_code_interpreter_file

    def touch(self, container_id, *, timeout):
        assert container_id == "container-test"
        harness.touches.append(phase[0])
        (query_touch if phase[0] == "query" else receipt_touch).set()
        return "running"

    def query(self, messages, **kwargs):
        # No intermediate response/tool evidence: only the known container ID
        # can protect notes while a background response silently reasons.
        assert query_touch.wait(2)
        yield from original_query(self, messages, **kwargs)

    def read(self, *args, **kwargs):
        phase[0] = "receipt"
        assert receipt_touch.wait(2)
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(APIClient, "touch_code_interpreter_container", touch)
    monkeypatch.setattr(APIClient, "run_queries", query)
    monkeypatch.setattr(APIClient, "read_code_interpreter_file", read)

    async def exercise():
        if over_budget:
            with pytest.raises(BudgetExhausted) as error:
                await ACCritic(harness.ctx)(**packet())
            result = error.value.completed_output
        else:
            result = await ACCritic(harness.ctx)(**packet())
        assert result.answer_ready and result.research_notes_execution_verified
        count = len(harness.touches)
        await asyncio.sleep(.03)
        assert len(harness.touches) == count
        assert not harness.deleted
    runner.run(exercise())
    assert query_touch.is_set() and receipt_touch.is_set() and len(harness.calls) == 1


def test_keepalive_is_deadline_bounded_and_reports_failures(harness, runner, monkeypatch):
    critic, stop = ACCritic(harness.ctx), asyncio.Event()
    client = SimpleNamespace(terminated=False)
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise RuntimeError("synthetic keepalive failure")
    client.touch_code_interpreter_container = fail
    monkeypatch.setattr(ACCritic, "NOTES_KEEPALIVE_S", .005)
    async def exercise():
        import time
        await critic._keep_notes_container_active(client, "saved-container", stop, time.monotonic() + .05)
        count = len(calls)
        await asyncio.sleep(.02)
        assert count == len(calls) and count > 0
    runner.run(exercise())
    assert all(0 < timeout < .05 for timeout in calls)
    assert "RuntimeError" in (harness.ctx.root_workdir / "events.jsonl").read_text()


@pytest.fixture
def queued_notes_executor(monkeypatch):
    from proofstack.agents.ac import critic as module
    queued, shutdown, options = [], [], []
    submitted = asyncio.Event()

    class QueuedExecutor:
        def __init__(self, **kwargs):
            options.append(kwargs)
        def submit(self, function, *args):
            future = Future()
            assert future.set_running_or_notify_cancel()
            queued.append((function, args, future))
            submitted.set()
            return future
        def shutdown(self, *, wait, cancel_futures):
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(module, "ThreadPoolExecutor", QueuedExecutor)
    return SimpleNamespace(queued=queued, shutdown=shutdown, options=options, submitted=submitted)


def test_keepalive_runs_while_default_executor_is_occupied(tmp_path, runner, monkeypatch, queued_notes_executor):
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))

    async def exercise():
        loop = asyncio.get_running_loop()
        busy = loop.create_future()
        original = loop.run_in_executor
        # Model a saturated default pool without relying on thread scheduling.
        monkeypatch.setattr(loop, "run_in_executor", lambda executor, function, *args:
                            busy if executor is None else original(executor, function, *args))
        calls = []
        client = SimpleNamespace(terminated=False, touch_code_interpreter_container=lambda *a, **k:
                                 calls.append((a, k)) or "running")
        keeper = asyncio.create_task(critic._touch_notes_container(
            client, "saved-container", deadline=float("inf"), timeout=float("inf")))
        try:
            await asyncio.sleep(0)  # Run setup through its first suspension, without a real worker or timer.
            assert queued_notes_executor.queued, "Keepalive must bypass the blocked default pool"
            function, args, future = queued_notes_executor.queued[0]
            future.set_result(function(*args))
            assert await keeper == "running"
            assert len(calls) == 1 and not busy.done()
            assert queued_notes_executor.options == [{"max_workers": 1, "thread_name_prefix": "critic-notes-keepalive"}]
            assert queued_notes_executor.shutdown == [(False, True)]
        finally:
            keeper.cancel()
            await asyncio.gather(keeper, return_exceptions=True)
            busy.cancel()
    runner.run(exercise())


@pytest.mark.parametrize("end", ["timeout", "deadline", "stop", "cancel", "terminated"])
def test_queued_keepalive_is_bounded_and_cannot_touch_after_stop(tmp_path, runner, monkeypatch, end):
    from proofstack.agents.ac import critic as module
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    clock, queued, shutdown = [0.0], [], []
    submitted = asyncio.Event()

    class DelayedExecutor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 1
        def submit(self, function, *args):
            future = Future()
            # A worker has taken the item but has not entered our guard yet.
            assert future.set_running_or_notify_cancel()
            queued.append((function, args, future))
            submitted.set()
            return future
        def shutdown(self, *, wait, cancel_futures):
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(module, "ThreadPoolExecutor", DelayedExecutor)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    client = SimpleNamespace(terminated=False, touch_code_interpreter_container=lambda *a, **k:
                             pytest.fail("An abandoned queued keepalive must not touch the container"))

    async def exercise():
        stop = asyncio.Event()
        task = asyncio.create_task(critic._touch_notes_container(
            client, "saved-container", deadline=10, timeout=.02 if end == "timeout" else 10, stop=stop))
        await submitted.wait()
        if end == "timeout":
            with pytest.raises(TimeoutError, match="keepalive deadline"):
                await asyncio.wait_for(task, timeout=2)
        elif end == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif end == "deadline":
            clock[0] = 11
        elif end == "stop":
            stop.set()
        else:
            client.terminated = True
        function, args, future = queued[0]
        assert function(*args) is None
        future.set_result(None)
        if end not in {"timeout", "cancel"}:
            assert await asyncio.wait_for(task, timeout=2) is None
        assert shutdown == [(False, True)]
    runner.run(exercise())


def test_running_keepalive_respects_deadline_without_joining_worker(tmp_path, runner):
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    release, finished = threading.Event(), threading.Event()

    async def exercise():
        loop, entered = asyncio.get_running_loop(), asyncio.Event()
        def touch(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(5)
                return "running"
            finally:
                finished.set()
        client = SimpleNamespace(terminated=False, touch_code_interpreter_container=touch)
        task = asyncio.create_task(critic._touch_notes_container(
            client, "saved-container", deadline=time.monotonic() + .2))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            with pytest.raises(TimeoutError, match="keepalive deadline"):
                await asyncio.wait_for(task, timeout=2)
            assert not finished.is_set()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
    runner.run(exercise())


def test_cancelled_setup_discards_a_result_arriving_during_cancellation(harness, runner):
    entered, release, discarded = threading.Event(), threading.Event(), threading.Event()
    deleted = []
    def setup(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return "unused-container", "file-test"
    def discard(container_id):
        deleted.append(container_id)
        discarded.set()
    client = SimpleNamespace(terminated=False, create_code_interpreter_container_with_file=setup,
                             discard_code_interpreter_container=discard)
    client.terminate = lambda: setattr(client, "terminated", True)
    async def exercise():
        task = asyncio.create_task(ACCritic(harness.ctx)._attach_notes(client, "unused", explicit=True, timeout=30))
        try:
            assert await asyncio.to_thread(entered.wait, 30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client.terminated and not deleted
        finally:
            release.set()
        assert await asyncio.to_thread(discarded.wait, 30)
    runner.run(exercise())
    assert deleted == ["unused-container"]


@pytest.mark.parametrize("cancel", [False, True])
def test_keepalive_shutdown_does_not_wait_for_worker_or_lose_paid_report(harness, runner, monkeypatch, cancel):
    harness.ctx.component_configs["ACCritic"] = {"research_notes_container": "explicit", "model": "models/openai/gpt-6-astra-max"}
    monkeypatch.setattr(ACCritic, "NOTES_KEEPALIVE_S", .005)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original_query = APIClient.run_queries
    def touch(self, container_id, *, timeout):
        entered.set()
        try:
            assert release.wait(5)
            return "running"
        finally:
            finished.set()
    def query(self, messages, **kwargs):
        assert entered.wait(2)
        yield from original_query(self, messages, **kwargs)
    monkeypatch.setattr(APIClient, "touch_code_interpreter_container", touch)
    monkeypatch.setattr(APIClient, "run_queries", query)
    async def exercise():
        critic = ACCritic(harness.ctx)
        receipt_started = asyncio.Event()
        if cancel:
            async def receipt(*args, **kwargs):
                receipt_started.set()
                await asyncio.Event().wait()
            monkeypatch.setattr(critic, "_retrieve_notes_verification", receipt)
        task = asyncio.create_task(critic(**packet()))
        try:
            if cancel:
                await asyncio.wait_for(receipt_started.wait(), timeout=2)
                assert critic._completed_review is not None
                task.cancel()
                asyncio.get_running_loop().call_soon(task.cancel)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
                assert harness.clients[0].terminated
            else:
                assert (await asyncio.wait_for(task, timeout=2)).answer_ready
            assert not finished.is_set()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
        result = await ACCritic(harness.ctx)(**packet())
        assert result.answer_ready and result.research_notes_execution_verified
    runner.run(exercise())
    assert len(harness.calls) == len(harness.uploads) == 1 and not harness.deleted


def test_default_notes_are_attached_not_inlined_and_cost_is_recorded(harness, runner):
    inp = packet(research_notes_tex="PRIVATE_SCRATCHPAD_CONTENT" * 100_000)
    out = runner.run(ACCritic(harness.ctx)(**inp))
    assert out.answer_ready and out.research_notes_status == "verified"
    assert out.research_notes_execution_verified
    assert len(harness.uploads) == len(harness.calls) == 1
    upload, call = harness.uploads[0], harness.calls[0]
    assert upload["data"] == inp["research_notes_tex"].encode()
    if harness.mode == "explicit":
        assert call["tools"][0]["container"] == "container-test"
        assert harness.containers[0]["expires_after"] == {"anchor": "last_active_at", "minutes": 20}
    else:
        assert upload["purpose"] == "user_data"
        assert upload["expires_after"] == {"anchor": "created_at", "seconds": 172800}
        assert call["tools"][0]["container"] == {"type": "auto", "file_ids": ["file-test1"]}
    prompt = json.dumps(call["messages"])
    assert "PRIVATE_SCRATCHPAD_CONTENT" not in prompt
    assert len(prompt) < 20_000
    for text in (inp["problem"], inp["answer_tex"], inp["references_bib"], upload["name"]):
        assert text in prompt
    assert "input_file" not in prompt
    receipt_path = next(harness.ctx.root_workdir.glob("agents/*/research-notes-attachment.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["sha256"] == hashlib.sha256(upload["data"]).hexdigest()
    assert receipt["bytes"] == len(upload["data"]) and receipt["round"] == 0
    assert receipt["file_id"] == "file-test1"
    assert receipt["container_mode"] == harness.mode
    if harness.mode == "explicit":
        assert receipt["container_id"] == "container-test" and receipt["idle_expiry_seconds"] == 1200
    assert (receipt_path.parent / receipt["filename"]).read_bytes() == upload["data"]
    assert harness.ctx.budgets.root().counters.usd == 0.25
    assert harness.clients[0].max_hosted_tool_calls == 30
    assert harness.clients[0].required_hosted_tool_types == {"code_interpreter"}
    assert "ac.critic.notes.attached" in (harness.ctx.root_workdir / "events.jsonl").read_text()


def test_new_round_attaches_only_its_snapshot_even_with_cached_client(harness, runner):
    critic = ACCritic(harness.ctx)
    first = runner.run(critic(**packet(round=1)))
    second = runner.run(critic(**packet(round=2, mode="stateful", prior_messages=first.messages_after,
                                       research_notes_tex="New revision notes")))
    assert second.answer_ready
    assert harness.uploads[0]["name"] != harness.uploads[1]["name"]
    assert harness.calls[1]["tools"][0]["container"] == (
        "container-test2" if harness.mode == "explicit" else {"type": "auto", "file_ids": ["file-test2"]})
    assert len(harness.clients) == 2
    assert "New revision notes" not in json.dumps(harness.calls[1]["messages"])
    assert "PRIVATE_SCRATCHPAD_CONTENT" not in json.dumps(second.messages_after)


@pytest.mark.parametrize("early_evidence", [False, True])
def test_background_poll_captures_receipt_when_tools_become_visible(harness, runner, monkeypatch, early_evidence):
    harness.control.logs = False
    harness.control.artifact = True
    polls, requests = [], []

    class Response(SimpleNamespace):
        def model_dump(self):
            return json.loads(json.dumps(vars(self), default=vars))

    def response(status, output):
        return Response(id="response-polled", status=status, output=output,
                        usage=SimpleNamespace(input_tokens=100, output_tokens=20)
                        if status == "completed" else None)

    def create(**kwargs):
        requests.append(kwargs)
        return response("queued", [])

    def retrieve(response_id, **kwargs):
        assert response_id == "response-polled"
        assert kwargs["include"] == ["code_interpreter_call.outputs"]
        polls.append(response_id)
        inp = ACCritic.Inputs(**packet())
        checker = SimpleNamespace(type="code_interpreter_call", id="ci-polled", status="completed",
                                  container_id="container-polled", outputs=[],
                                  code=ACCritic(harness.ctx)._notes_verification_code(inp))
        if len(polls) == 1:
            return response("in_progress", [checker] if early_evidence else [])
        assert len(harness.downloads) == int(early_evidence)
        if early_evidence:
            harness.control.artifact_error = RuntimeError("Container expired during later reasoning")
        message = SimpleNamespace(type="message", id="msg-polled", content=[
            SimpleNamespace(type="output_text", text=harness.control.reply)])
        return response("completed", [checker, message])

    def query(self, messages, **kwargs):
        sdk = SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=retrieve))
        result = self._openai_query_responses_api(sdk, 0, messages[0])
        yield (0, result.conversation, {"cost": result.cost_usd,
                                      "input_tokens": result.input_tokens, "output_tokens": result.output_tokens})

    monkeypatch.setattr(APIClient, "run_queries", query)
    monkeypatch.setattr("mathagents.api_client.time.sleep", lambda _: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_request", lambda **_: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_response", lambda **_: None)
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert out.answer_ready and out.research_notes_execution_verified
    assert len(requests) == 1 and len(polls) == 2 and len(harness.downloads) == 1
    assert requests[0]["background"] is True and requests[0]["max_tool_calls"] == 30
    resumed = runner.run(ACCritic(harness.ctx)(**packet()))
    assert resumed.answer_ready and resumed.research_notes_execution_verified
    assert resumed.review_md.strip() == out.review_md.strip()
    assert len(requests) == 1 and len(harness.downloads) == 1


@pytest.mark.parametrize("reply,status", [
    ("<answer_ready>true</answer_ready>", "not_checked"),
    ("<research_notes_status>unavailable</research_notes_status><answer_ready>true</answer_ready>", "unavailable"),
    ("<research_notes_status>verified</research_notes_status>" * 2 + "<answer_ready>true</answer_ready>", "not_checked"),
])
def test_missing_or_failed_file_verification_blocks_acceptance(harness, runner, reply, status):
    harness.control.reply = reply
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert not out.answer_ready
    assert out.research_notes_status == status


def test_upload_failure_is_explicit_and_never_falls_back_to_inline(harness, runner):
    harness.control.upload_error = RuntimeError("synthetic upload failure")
    with pytest.raises(RuntimeError, match="upload failure"):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.calls
    assert harness.ctx.budgets.root().counters.usd == 0
    assert "ac.critic.notes.call_failed" in (harness.ctx.root_workdir / "events.jsonl").read_text()


def test_empty_notes_need_no_upload(harness, runner):
    harness.control.reply = "<answer_ready>true</answer_ready>"
    out = runner.run(ACCritic(harness.ctx)(**packet(research_notes_tex="")))
    assert out.answer_ready and out.research_notes_status == "empty"
    assert not harness.uploads
    assert "no notes file is attached" in json.dumps(harness.calls[0]["messages"])


def test_explicit_inline_compatibility_mode(harness, runner):
    harness.ctx.component_configs["ACCritic"] = {"research_notes_transport": "inline"}
    harness.control.reply = "<answer_ready>true</answer_ready>"
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert out.answer_ready and out.research_notes_status == "inline"
    assert not harness.uploads
    assert "PRIVATE_SCRATCHPAD_CONTENT" in json.dumps(harness.calls[0]["messages"])


def test_context_recovery_reattaches_same_snapshot_and_replays_completed_receipt(harness, runner):
    harness.control.query_error = ValueError("context_length_exceeded")
    inp = packet(mode="stateful", round=2, prior_messages=[{"role": "user", "content": "Old proof"}])
    out = runner.run(ACCritic(harness.ctx)(**inp))
    assert out.answer_ready and out.mode == "fresh"
    assert len(harness.uploads) == 2
    assert all(harness.uploads[0][key] == harness.uploads[1][key] for key in ("name", "data"))
    assert harness.calls[1]["tools"][0]["container"] == (
        "container-test2" if harness.mode == "explicit" else {"type": "auto", "file_ids": ["file-test2"]})
    assert "Old proof" not in json.dumps(harness.calls[1]["messages"])
    assert runner.run(ACCritic(harness.ctx)(**inp)) == out
    assert len(harness.uploads) == len(harness.calls) == 2


def test_legacy_cached_verified_tag_does_not_bypass_execution_gate(harness, runner):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    legacy = critic.Outputs(answer_ready=True, research_notes_status="verified")
    harness.ctx.resume_cache.put(critic._cache_key(inp), legacy.model_dump(mode="json"))
    harness.control.evidence = False
    out = runner.run(critic(**inp.model_dump()))
    assert len(harness.calls) == 1 and not out.answer_ready


@pytest.mark.parametrize("variant", ["formatting", "split_imports", "null_before_valid", "null_only", "forged",
                                      "deep_before_valid", "deep_after_valid", "deep_only"])
def test_file_verification_tolerates_formatting_not_fabricated_evidence(harness, variant):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    code = critic._notes_verification_code(inp)
    metadata = critic._notes_metadata(inp)
    logs = json.dumps({k: metadata[k] for k in ("filename", "bytes", "sha256")})
    if variant == "formatting":
        import ast
        code = ast.unparse(ast.parse(code)) + "\n# Formatting only\n"
    elif variant == "split_imports":
        import ast
        tree = ast.parse(code)
        tree.body = [part for node in tree.body for part in
                     ([ast.Import(names=[name]) for name in node.names] if isinstance(node, ast.Import) else [node])]
        code = ast.unparse(tree)
    elif variant == "forged":
        code = f"print({logs!r})"
    elif variant == "null_only":
        code = None
    evidence = {"status": "completed", "container_id": "container", "code": code,
                "outputs": [{"type": "logs", "logs": logs}]}
    critic._provider_tool_evidence = [evidence]
    if variant == "null_before_valid":
        critic._provider_tool_evidence.insert(0, {**evidence, "code": None})
    if variant.startswith("deep_"):
        deep = {**evidence, "code": "x = " + " + ".join(["1"] * 1200)}
        if variant == "deep_only":
            critic._provider_tool_evidence = [deep]
        else:
            critic._provider_tool_evidence.insert(0 if variant == "deep_before_valid" else 1, deep)
    assert critic._notes_verified(inp) is (variant not in {"null_only", "forged", "deep_only"})


@pytest.mark.parametrize("stage", ["parse", "dump"])
def test_verification_recursion_is_nonmatching_evidence(monkeypatch, stage):
    def too_deep(*args, **kwargs):
        raise RecursionError("synthetic deep tool code")

    monkeypatch.setattr(f"proofstack.agents.ac.critic.ast.{stage}", too_deep)
    assert ACCritic._verification_syntax("x = 1") is None


@pytest.mark.parametrize("surrounding", [
    ("Diagnostic before\n", "\nDiagnostic after"),
    ("Diagnostic before: ", " Diagnostic after"),
    ('Warning {not JSON}\n{"diagnostic": true}\n', '\n{"diagnostic": false}'),
    ("```json\n", "\n```"),
])
@pytest.mark.parametrize("fault", [None, "hash", "bytes", "filename", "code", "status", "container"])
def test_verifier_accepts_diagnostic_logs_without_weakening_evidence(harness, surrounding, fault):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    metadata = critic._notes_metadata(inp)
    if fault in {"hash", "bytes", "filename"}:
        metadata[{"hash": "sha256"}.get(fault, fault)] = "wrong"
    result = json.dumps(metadata, indent=2)
    prefix, suffix = surrounding
    critic._provider_tool_evidence = [{
        "status": "failed" if fault == "status" else "completed",
        "container_id": None if fault == "container" else "container",
        "code": f"print({result!r})" if fault == "code" else critic._notes_verification_code(inp),
        "outputs": [{"type": "logs", "logs": prefix + result + suffix}],
    }]
    assert critic._notes_verified(inp) is (fault is None)


@pytest.mark.parametrize("logs", [None, {}, "{unfinished", '{"diagnostic": true}', "[]"])
def test_unusable_verifier_logs_do_not_raise_or_verify(harness, logs):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    critic._provider_tool_evidence = [{
        "status": "completed", "container_id": "container", "code": critic._notes_verification_code(inp),
        "outputs": [{"type": "logs", "logs": logs}],
    }]
    assert not critic._notes_verified(inp)


def test_invalid_expected_verification_syntax_fails_closed(harness, monkeypatch):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    logs = json.dumps({k: critic._notes_metadata(inp)[k] for k in ("filename", "bytes", "sha256")})
    critic._provider_tool_evidence = [{"status": "completed", "container_id": "container",
                                      "code": None, "outputs": [{"type": "logs", "logs": logs}]}]
    monkeypatch.setattr(critic, "_verification_syntax", lambda code: None)
    assert not critic._notes_verified(inp)


def test_deep_tool_code_does_not_discard_paid_review(harness, runner, monkeypatch):
    from mathagents.provider_trace import ProviderTrace

    original = ProviderTrace.tool_evidence

    def with_deep_code(self):
        return [{"status": "completed", "container_id": "container-test",
                 "code": "x = " + " + ".join(["1"] * 1200), "outputs": []}, *original(self)]

    monkeypatch.setattr(ProviderTrace, "tool_evidence", with_deep_code)
    critic = ACCritic(harness.ctx)
    result = runner.run(critic(**packet()))
    assert result.answer_ready and result.research_notes_execution_verified
    assert critic._completed_review.answer_ready
    assert len(harness.calls) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25


def test_history_read_failure_does_not_discard_completed_review(harness, runner, monkeypatch):
    read_text = Path.read_text
    events_path = harness.ctx.root_workdir / "events.jsonl"

    def unreadable_events(path, *args, **kwargs):
        if path == events_path:
            raise OSError("synthetic history read failure")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable_events)
    critic = ACCritic(harness.ctx)
    result = runner.run(critic(**packet()))
    assert result.answer_ready and result.research_notes_execution_verified
    assert critic._completed_review.answer_ready
    assert len(harness.calls) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25
    assert "accounting.events_unreadable" in read_text(events_path)


def test_unsupported_provider_fails_before_upload_or_model_call(harness, runner):
    harness.ctx.model_overrides["ACCritic"] = {"base": "models/openai/gpt-6-astra-max", "use_openai_responses_api": False}
    with pytest.raises(ValueError, match="Responses"):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.uploads and not harness.calls


def test_budget_exhausted_before_upload(harness, runner):
    harness.ctx.budgets.root().spec = BudgetSpec(max_usd=1)
    harness.ctx.budgets.root().add_usd(1)
    with pytest.raises(BudgetExhausted):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.uploads and not harness.calls


def test_upload_deadline_is_bounded_and_uses_configured_endpoint(harness, monkeypatch, runner):
    harness.ctx.model_overrides["ACCritic"] = {
        "base": "models/openai/gpt-6-astra-max", "base_url": "https://synthetic.invalid/v1",
        "api": "custom", "api_key_env": "SYNTHETIC_API_KEY",
    }
    monkeypatch.setenv("SYNTHETIC_API_KEY", "synthetic-other-key")
    critic = ACCritic(harness.ctx)
    monkeypatch.setattr(critic.tracker, "remaining_wallclock_s", lambda: 5.0)
    runner.run(critic(**packet()))
    assert harness.sdk_options[0]["api_key"] == "synthetic-other-key"
    assert harness.sdk_options[0]["base_url"] == "https://synthetic.invalid/v1"
    assert 0 < harness.sdk_options[0]["timeout"] <= 5.0
    assert harness.sdk_options[0]["max_retries"] == 0


def test_cancelling_upload_terminates_client_without_query(harness, monkeypatch, runner):
    entered, finished = threading.Event(), threading.Event()

    def upload(self, path, *, timeout):
        entered.set()
        try:
            # The actual bounded operation notices terminate() without waiting
            # for a stuck SDK. Keep this fake deterministic and offline.
            while not self.terminated:
                finished.wait(0.001)
        finally:
            finished.set()
        raise RuntimeError("cancelled upload")

    monkeypatch.setattr(APIClient, "create_code_interpreter_container_with_file" if harness.mode == "explicit"
                        else "attach_code_interpreter_file", upload)

    async def run():
        task = asyncio.create_task(ACCritic(harness.ctx)(**packet()))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(finished.wait, 2)

    runner.run(run())
    assert not harness.calls and harness.clients[0].terminated


def test_upload_timeout_does_not_wait_for_stuck_sdk(harness, monkeypatch, runner):
    harness.ctx.component_configs["ACCritic"] = {"research_notes_container": "auto"}
    release, finished = threading.Event(), threading.Event()

    class StuckSDK:
        def __init__(self, **kwargs):
            self.files = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            try:
                assert release.wait(5)
                return SimpleNamespace(id="file-late")
            finally:
                finished.set()

    monkeypatch.setattr("mathagents.api_client.OpenAI", StuckSDK)
    critic = ACCritic(harness.ctx)
    monkeypatch.setattr(critic.tracker, "remaining_wallclock_s", lambda: 0.05)
    try:
        with pytest.raises(Exception, match="wall-clock deadline"):
            runner.run(asyncio.wait_for(critic(**packet()), timeout=2))
        assert not harness.calls
    finally:
        release.set()
        assert finished.wait(2)


def test_cleanup_critic_keeps_its_separate_inline_policy(harness, runner):
    from proofstack.agents.batch3_critic import Batch3CleanupCritic

    harness.control.reply = "<cleanup_verdict>accept</cleanup_verdict>"
    out = runner.run(Batch3CleanupCritic(harness.ctx)(**packet(), baseline_tex="Baseline proof"))
    assert out.answer_ready and not harness.uploads
    prompt = json.dumps(harness.calls[0]["messages"])
    assert "Baseline proof" in prompt
    assert "research_notes_status" not in prompt


def test_self_attested_verification_without_tool_output_is_not_accepted(harness, runner):
    harness.control.evidence = False
    result = runner.run(ACCritic(harness.ctx)(**packet()))
    assert not result.answer_ready and result.research_notes_status == "not_checked"


@pytest.mark.parametrize("corrupt", ["code", "digest", "status", "container"])
def test_verification_requires_matching_executed_code_and_result(harness, runner, monkeypatch, corrupt):
    from mathagents.provider_trace import ProviderTrace

    original = ProviderTrace.tool_evidence

    def evidence(self):
        data = json.loads(json.dumps(original(self)))
        if data:
            if corrupt == "code":
                data[0]["code"] = "print('invented verification')"
            elif corrupt == "digest":
                data[0]["outputs"][0]["logs"] = '{"sha256": "wrong"}'
            elif corrupt == "status":
                data[0]["status"] = "failed"
            else:
                data[0]["container_id"] = None
        return data

    monkeypatch.setattr(ProviderTrace, "tool_evidence", evidence)
    assert not runner.run(ACCritic(harness.ctx)(**packet())).answer_ready


def test_upload_outages_do_not_consume_fresh_model_recovery_attempts(harness, runner, monkeypatch):
    from proofstack.kinds.api_call import APICallAgent

    original = APICallAgent.run
    calls = []

    async def review(self, inp):
        calls.append(inp.mode)
        if inp.mode == "stateful":
            harness.control.upload_error = ConnectionError("upload unavailable")
            raise ValueError("context_length_exceeded")
        return await original(self, inp)

    monkeypatch.setattr(APICallAgent, "run", review)
    inp = packet(mode="stateful", round=2)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            runner.run(ACCritic(harness.ctx)(**inp))
    checkpoint = next(harness.ctx.root_workdir.glob("critic_context/*.json"))
    assert json.loads(checkpoint.read_text())["attempts"] == 0
    harness.control.upload_error = None
    result = runner.run(ACCritic(harness.ctx)(**inp))
    assert result.answer_ready and calls == ["stateful", "fresh"]


def test_empty_stdout_uses_downloaded_receipt_without_another_model_call(harness, runner):
    harness.control.logs = False
    harness.control.artifact = True
    critic = ACCritic(harness.ctx)
    out = runner.run(critic(**packet()))
    assert out.answer_ready and out.research_notes_execution_verified
    assert critic._completed_review == out
    assert len(harness.calls) == len(harness.uploads) == len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25
    saved = next(harness.ctx.root_workdir.glob("agents/*/notes-verification-*.json"))
    receipt = json.loads(saved.read_text())
    assert receipt["call_id"] == "ci-test" and receipt["container_id"] == "container-test"
    assert "PRIVATE_SCRATCHPAD_CONTENT" not in saved.read_text()
    assert "ac.critic.notes.receipt_verified" in (harness.ctx.root_workdir / "events.jsonl").read_text()


@pytest.mark.parametrize("corrupt", ["hash", "bytes", "filename", "malformed", "oversized", "missing", "expired"])
def test_invalid_or_missing_receipt_fails_closed_without_repaying(harness, runner, corrupt):
    harness.control.logs = False
    harness.control.artifact = corrupt != "missing"
    metadata = ACCritic(harness.ctx)._notes_metadata(ACCritic.Inputs(**packet()))
    if corrupt in {"hash", "bytes", "filename"}:
        metadata[{"hash": "sha256"}.get(corrupt, corrupt)] = "stale"
        harness.control.receipt_override = json.dumps(metadata)
    elif corrupt == "malformed":
        harness.control.receipt_override = "not json"
    elif corrupt == "oversized":
        harness.control.receipt_override = " " * 4097
    elif corrupt == "expired":
        class ContainerExpired(Exception):
            status_code = 404
        harness.control.artifact_error = ContainerExpired("Error code: 404 - Container is expired")
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert not out.answer_ready and not out.research_notes_execution_verified
    assert out.research_notes_status == "not_checked"
    assert len(harness.calls) == 1 and harness.ctx.budgets.root().counters.usd == 0.25
    assert out.messages_after[-1]["content"] == harness.control.reply
    assert not ACCritic(harness.ctx).cache_output_is_reusable(out)


def test_receipt_still_requires_exact_completed_checker(harness, runner, monkeypatch):
    from mathagents.provider_trace import ProviderTrace

    harness.control.logs = False
    harness.control.artifact = True
    original = ProviderTrace.tool_evidence

    def forged(self):
        data = original(self)
        for item in data:
            item["code"] = "print('verified')"
        return data

    monkeypatch.setattr(ProviderTrace, "tool_evidence", forged)
    assert not runner.run(ACCritic(harness.ctx)(**packet())).answer_ready
    assert not harness.downloads


def test_file_receipt_recovery_reuses_durable_evidence_after_container_expiry(harness, runner):
    harness.control.logs = False
    harness.control.artifact = True
    harness.control.query_error = ValueError("context_length_exceeded")
    inp = packet(mode="stateful", round=2)
    first = runner.run(ACCritic(harness.ctx)(**inp))
    assert first.answer_ready and len(harness.calls) == 2 and len(harness.downloads) == 1
    # Force recovery from the provider report plus downloaded receipt, before
    # the augmented completed checkpoint was saved.
    checkpoint = next(harness.ctx.root_workdir.glob("critic_context/*.json"))
    saved = json.loads(checkpoint.read_text())
    saved.update(status="interrupted")
    saved.pop("tool_evidence")
    saved.pop("output")
    checkpoint.write_text(json.dumps(saved))
    harness.control.artifact_error = RuntimeError("Container expired")
    second = runner.run(ACCritic(harness.ctx)(**inp))
    assert second == first
    assert len(harness.calls) == 2 and len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25


@pytest.mark.parametrize("fault", ["container_id", "call_id", "path", "file_id"])
def test_receipt_must_belong_to_the_current_executed_checker(harness, runner, fault):
    harness.control.logs = False
    harness.control.artifact = True
    critic = ACCritic(harness.ctx)
    out = runner.run(critic(**packet()))
    assert out.answer_ready
    item = critic._provider_tool_evidence[0]
    item["notes_verification_receipt"][fault] = "" if fault == "file_id" else "other"
    assert not critic._notes_verified(critic.Inputs(**packet()))


def test_receipt_download_respects_remaining_deadline(harness, runner, monkeypatch):
    harness.control.logs = False
    harness.control.artifact = True
    critic = ACCritic(harness.ctx)
    monkeypatch.setattr(critic.tracker, "remaining_wallclock_s", lambda: 2.0)
    assert runner.run(critic(**packet())).answer_ready
    assert 0 < harness.downloads[0]["timeout"] <= 2


def test_legacy_stdout_receipts_remain_valid_without_download(harness):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    critic._provider_tool_evidence = [{"status": "completed", "container_id": "container",
        "code": critic._notes_verification_code(inp, receipt=False),
        "outputs": [{"type": "logs", "logs": json.dumps(critic._notes_metadata(inp))}]}]
    assert critic._notes_verified(inp)


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("source", ["provider_report", "checkpoint"])
@pytest.mark.parametrize("field", [None, "filename", "bytes", "sha256"])
def test_previous_checker_stdout_survives_review_resume(harness, runner, policy, source, field):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet(mode="stateful", round=2))
    assert runner.run(critic(**inp.model_dump())).answer_ready
    checkpoint = critic._recovery_checkpoint(inp)
    saved = json.loads(checkpoint.read_text())
    metadata = {k: critic._notes_metadata(inp)[k] for k in ("filename", "bytes", "sha256")}
    if field:
        metadata[field] = "wrong"
    evidence = [{"id": "ci-old", "status": "completed", "container_id": "container-test",
        "code": previous_receipt_checker(critic, inp),
        "outputs": [{"type": "logs", "logs": json.dumps(metadata)}]}]
    if source == "provider_report":
        root = harness.ctx.root_workdir / saved["attempt_workdir"]
        report_path = next(root.glob("provider-completed-*.json"))
        report = json.loads(report_path.read_text())
        report["tool_evidence"] = evidence
        report_path.write_text(json.dumps(report))
        saved.update(status="interrupted")
        saved.pop("tool_evidence")
        saved.pop("output")
    else:
        saved.pop("attempt_workdir")
        saved["tool_evidence"] = evidence
        saved["output"]["research_notes_verification_version"] = 2
    checkpoint.write_text(json.dumps(saved))
    resumed = runner.run(ACCritic(harness.ctx)(**inp.model_dump()))
    assert resumed.answer_ready is (field is None)
    assert resumed.research_notes_execution_verified is (field is None)
    assert resumed.research_notes_integrity_failed is (field is not None)
    assert resumed.research_notes_status == ("verified" if field is None else "not_checked")
    assert resumed.research_notes_verification_version == (3 if field is None else 0)
    assert not resumed.research_notes_advisory_accepted
    assert runner.run(ACCritic(harness.ctx)(**inp.model_dump())) == resumed
    assert len(harness.calls) == len(harness.uploads) == 1 and not harness.downloads
    assert harness.ctx.budgets.root().counters.usd == .25


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("digest", [False, True])
def test_previous_checker_cannot_verify_from_receipt_only(harness, runner, policy, digest):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    metadata = critic._notes_metadata(inp)
    if not digest:
        metadata.pop("receipt_sha256")
    item = {"id": "ci-old", "status": "completed", "container_id": "container-test",
            "code": previous_receipt_checker(critic, inp), "outputs": []}
    item["notes_verification_receipt"] = {"call_id": item["id"], "container_id": item["container_id"],
        "file_id": "file-old", "path": critic._notes_receipt_file(inp), "content": json.dumps(metadata)}
    critic._provider_tool_evidence = [item]
    runner.run(critic._retrieve_notes_verification(inp))
    out = critic.parse_output(harness.control.reply, inp)
    assert out.research_notes_status == "not_checked" and not out.research_notes_execution_verified
    assert out.research_notes_verification_version == 0
    assert out.answer_ready is (policy == "advisory")
    assert out.research_notes_advisory_accepted is (policy == "advisory")
    assert not harness.calls and not harness.downloads


@pytest.mark.parametrize("fault", ["extra_statement", "other_round", "missing_container", "incomplete"])
def test_previous_checker_stdout_still_requires_exact_current_completed_call(harness, fault):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    checked_input = inp.model_copy(update={"round": inp.round + 1}) if fault == "other_round" else inp
    item = {"id": "ci-old", "status": "completed", "container_id": "container-test",
        "code": previous_receipt_checker(critic, checked_input),
        "outputs": [{"type": "logs", "logs": json.dumps(critic._notes_metadata(inp))}]}
    if fault == "extra_statement":
        item["code"] += "\nprint('extra')"
    elif fault == "missing_container":
        item.pop("container_id")
    elif fault == "incomplete":
        item["status"] = "in_progress"
    critic._provider_tool_evidence = [item]
    assert not critic.parse_output(harness.control.reply, inp).answer_ready
    assert critic._notes_evidence_status(inp) == "not_checked"


@pytest.mark.parametrize("layout", ["valid", "changed", "missing", "ambiguous"])
@pytest.mark.parametrize("stale_receipt", [False, True])
def test_actual_checker_writes_computed_receipt_only_after_reading_unique_file(harness, tmp_path, monkeypatch, layout, stale_receipt):
    import sys

    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    metadata = critic._notes_metadata(inp)
    code = critic._notes_verification_code(inp)
    data_root = tmp_path / "container"
    data_root.mkdir()
    receipt_path = data_root / Path(critic._notes_receipt_file(inp)).name
    if stale_receipt:
        receipt_path.write_text(json.dumps(metadata))
    if layout != "missing":
        (data_root / f"file-platform-{metadata['filename']}").write_text(
            "Different bytes" if layout == "changed" else inp.research_notes_tex)
    if layout == "ambiguous":
        (data_root / f"file-other-{metadata['filename']}").write_text(inp.research_notes_tex)

    def container_path(value):
        return data_root / Path(value).relative_to("/mnt/data")

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pathlib", SimpleNamespace(Path=container_path))
        if layout in {"missing", "ambiguous"}:
            with pytest.raises(AssertionError, match="missing or ambiguous"):
                exec(code, {})
        else:
            exec(code, {})
    receipt_path = data_root / Path(critic._notes_receipt_file(inp)).name
    if layout in {"missing", "ambiguous"}:
        assert not receipt_path.exists()
    else:
        receipt = {"path": critic._notes_receipt_file(inp), "content": receipt_path.read_text(),
                   "file_id": "cfile", "call_id": "call", "container_id": "container"}
        item = {"id": "call", "container_id": "container"}
        assert critic._notes_receipt_matches(inp, item, receipt) is (layout == "valid")


@pytest.mark.parametrize("context_recovery", [False, True])
def test_cancelled_receipt_download_keeps_paid_review_recoverable(harness, runner, monkeypatch, context_recovery):
    harness.control.logs = False
    harness.control.artifact = True
    if context_recovery:
        harness.control.query_error = ValueError("context_length_exceeded")
    original = APIClient.read_code_interpreter_file

    def interrupted(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(APIClient, "read_code_interpreter_file", interrupted)
    inp = packet(mode="stateful", round=2)
    with pytest.raises(asyncio.CancelledError):
        runner.run(ACCritic(harness.ctx)(**inp))
    assert len(harness.calls) == 1 + context_recovery and harness.ctx.budgets.root().counters.usd == 0.25
    monkeypatch.setattr(APIClient, "read_code_interpreter_file", original)
    out = runner.run(ACCritic(harness.ctx)(**inp))
    assert out.answer_ready and out.research_notes_execution_verified
    assert len(harness.calls) == 1 + context_recovery and len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25


def test_poll_captures_receipt_before_container_expires(harness, runner):
    harness.control.logs = False
    harness.control.artifact = True
    harness.control.poll_evidence = True
    harness.control.expire_after_poll = True
    inp = packet(mode="stateful", round=2)
    first = runner.run(ACCritic(harness.ctx)(**inp))
    assert first.answer_ready and first.research_notes_verification_version == 3
    assert len(harness.downloads) == len(harness.calls) == 1
    assert runner.run(ACCritic(harness.ctx)(**inp)) == first
    assert len(harness.downloads) == len(harness.calls) == 1


@pytest.mark.parametrize("mode", ["fresh", "stateful"])
def test_completed_over_budget_review_still_retrieves_and_retains_verification(harness, runner, mode):
    harness.control.logs = False
    harness.control.artifact = True
    harness.ctx.budgets.root().spec = BudgetSpec(max_usd=0.20)
    critic = ACCritic(harness.ctx)
    with pytest.raises(BudgetExhausted) as caught:
        runner.run(critic(**packet(mode=mode)))
    assert caught.value.completed_output.answer_ready
    assert critic._completed_review == caught.value.completed_output
    assert len(harness.calls) == len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == 0.25


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("status", ["not_checked", "unavailable", "inline"])
def test_missing_evidence_is_not_a_reusable_research_verdict(harness, ready, status):
    critic = ACCritic(harness.ctx)
    assert not critic.cache_output_is_reusable(critic.Outputs(answer_ready=ready, research_notes_status=status))


@pytest.mark.parametrize("ready", [False, True])
def test_only_current_verified_research_outputs_are_cacheable(harness, ready):
    critic = ACCritic(harness.ctx)
    legacy = critic.Outputs(answer_ready=ready, research_notes_status="verified", research_notes_execution_verified=True)
    assert not critic.cache_output_is_reusable(legacy)
    assert not critic.cache_output_is_reusable(legacy.model_copy(update={"research_notes_verification_version": 2}))
    assert critic.cache_output_is_reusable(legacy.model_copy(update={"research_notes_verification_version": 3}))


@pytest.mark.parametrize("policy", [None, True, "optional", [], {}])
def test_invalid_verification_policy_fails_before_provider_io(harness, runner, policy):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    with pytest.raises(ValueError, match="research_notes_verification must be"):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.uploads and not harness.calls and not harness.sdk_options


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("failure", ["expired", "missing", "timeout", "no_tool_evidence"])
@pytest.mark.parametrize("context_recovery", [False, True])
def test_receipt_policy_preserves_paid_verdict_on_resume(harness, runner, policy, failure, context_recovery):
    class NotFoundError(Exception):
        status_code = 404

    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    harness.control.logs = False
    if failure == "expired":
        harness.control.artifact_error = NotFoundError("Error code: 404 - Container is expired")
    elif failure == "timeout":
        harness.control.artifact_error = TimeoutError("synthetic timeout")
    elif failure == "no_tool_evidence":
        harness.control.evidence = False
    if context_recovery:
        harness.control.query_error = ValueError("context_length_exceeded")
    inp = packet(mode="stateful", round=2)
    out = runner.run(ACCritic(harness.ctx)(**inp))
    assert out.answer_ready is (policy == "advisory")
    assert out.research_notes_advisory_accepted is (policy == "advisory")
    assert out.research_notes_verification_policy == policy
    assert out.research_notes_status == "not_checked"
    assert not out.research_notes_execution_verified and not out.research_notes_integrity_failed
    assert out.research_notes_verification_version == 0
    assert out.messages_after[-1]["content"] == harness.control.reply
    assert len(harness.calls) == 1 + context_recovery
    assert harness.ctx.budgets.root().counters.usd == .25
    assert runner.run(ACCritic(harness.ctx)(**inp)) == out
    assert len(harness.calls) == 1 + context_recovery
    assert harness.ctx.budgets.root().counters.usd == .25
    events = [json.loads(line) for line in (harness.ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    decisions = [e["payload"] for e in events if e["kind"] == "ac.critic.notes.verification"]
    assert decisions and decisions[-1]["policy"] == policy
    assert decisions[-1]["advisory_accepted"] is (policy == "advisory")
    assert not decisions[-1]["execution_verified"]
    if policy == "advisory":
        assert "not verified access" in out.review_md
        assert not harness.clients[-1].required_hosted_tool_types
        assert harness.clients[-1].required_hosted_tool_validator is None


@pytest.mark.parametrize("mode", ["fresh", "stateful"])
def test_advisory_prompt_keeps_proof_and_integrity_requirements(harness, mode):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    inp = ACCritic.Inputs(**packet(mode=mode))
    text = ACCritic(harness.ctx).render_messages(inp)[-1]["content"]
    assert "Verification is advisory" in text
    assert "missing essential proof steps or computational certificates" in text
    assert "LaTeX-contract violations still require answer_ready=false" in text
    assert "<research_notes_status>mismatch</research_notes_status> and set answer_ready=false" in text
    assert "Notes-only arguments never count as proof" in text
    assert "last tool call, immediately before" in text
    assert "If unavailable, explain the access/verification failure and set answer_ready=false" not in text


@pytest.mark.parametrize("reply,ready,status", [
    ("<answer_ready>true</answer_ready>", True, "not_checked"),
    ("<research_notes_status>unavailable</research_notes_status><answer_ready>true</answer_ready>", True, "unavailable"),
    ("<research_notes_status>verified</research_notes_status>" * 2 + "<answer_ready>true</answer_ready>", True, "not_checked"),
    ("Missing a proof certificate. <answer_ready>false</answer_ready>", False, "not_checked"),
    ("There is a mathematical gap. <answer_ready>false</answer_ready>", False, "not_checked"),
    ("No parseable verdict", False, "not_checked"),
])
def test_advisory_never_invents_verification_or_promotes_rejection(harness, runner, reply, ready, status):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.logs, harness.control.reply = False, reply
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert out.answer_ready is ready and out.research_notes_advisory_accepted is ready
    assert out.research_notes_status == status
    assert not out.research_notes_execution_verified


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("field", ["filename", "bytes", "sha256"])
@pytest.mark.parametrize("poll", [False, True])
def test_confirmed_receipt_mismatch_blocks_both_policies_and_survives_resume(harness, runner, policy, field, poll):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    harness.control.logs, harness.control.artifact, harness.control.poll_evidence = False, True, poll
    critic = ACCritic(harness.ctx)
    metadata = critic._notes_metadata(critic.Inputs(**packet()))
    metadata[field] = "wrong"
    harness.control.receipt_override = json.dumps(metadata)
    first = runner.run(critic(**packet()))
    assert not first.answer_ready and not first.research_notes_advisory_accepted
    assert first.research_notes_integrity_failed
    assert first.research_notes_status == "not_checked" and not first.research_notes_execution_verified
    assert "integrity mismatch" in first.review_md
    assert not critic.cache_output_is_reusable(first)
    harness.control.artifact_error = RuntimeError("container expired after mismatch")
    second = runner.run(ACCritic(harness.ctx)(**packet()))
    assert second == first
    assert len(harness.calls) == len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == .25


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("field", ["filename", "bytes", "sha256"])
@pytest.mark.parametrize("reverse", [False, True])
def test_stdout_mismatch_is_not_overridden_by_another_matching_checker(harness, policy, field, reverse):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    good = critic._notes_metadata(inp)
    bad = {**good, field: "wrong"}
    critic._provider_tool_evidence = [{"status": "completed", "container_id": "container", "id": f"ci-{index}",
        "code": critic._notes_verification_code(inp), "outputs": [{"type": "logs", "logs": json.dumps(metadata)}]}
        for index, metadata in enumerate([bad, good] if reverse else [good, bad])]
    out = critic.parse_output(harness.control.reply, inp)
    assert not out.answer_ready and not critic._notes_verified(inp)
    assert out.research_notes_integrity_failed and not out.research_notes_execution_verified


@pytest.mark.parametrize("status", ["mismatch", "MISMATCH"])
def test_model_reported_mismatch_cannot_be_waived_by_advisory_policy(harness, runner, status):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.reply = f"<research_notes_status>{status}</research_notes_status><answer_ready>true</answer_ready>"
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert not out.answer_ready and out.research_notes_integrity_failed
    assert not out.research_notes_advisory_accepted


def test_advisory_upload_failure_still_fails_before_model(harness, runner):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.upload_error = ValueError("synthetic invalid upload")
    with pytest.raises(ValueError, match="synthetic invalid upload"):
        runner.run(ACCritic(harness.ctx)(**packet()))
    assert not harness.calls


@pytest.mark.parametrize("policy", ["required", "advisory"])
def test_advisory_cache_is_policy_scoped_and_legacy_safe(harness, policy):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    legacy = critic.Outputs(answer_ready=True, research_notes_status="not_checked")
    assert not critic.cache_output_is_reusable(legacy)
    advisory = legacy.model_copy(update={"research_notes_verification_policy": "advisory",
                                         "research_notes_advisory_accepted": True})
    assert critic.cache_output_is_reusable(advisory) is (policy == "advisory")
    assert not critic.cache_output_is_reusable(advisory.model_copy(update={"research_notes_integrity_failed": True}))
    assert not critic.cache_output_is_reusable(advisory.model_copy(update={"parse_failed": True}))
    assert not critic.cache_output_is_reusable(advisory.model_copy(update={"research_notes_execution_verified": True}))


def test_advisory_verified_review_still_has_real_verification(harness, runner):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.logs, harness.control.artifact = False, True
    critic = ACCritic(harness.ctx)
    out = runner.run(critic(**packet()))
    assert out.answer_ready and out.research_notes_execution_verified
    assert out.research_notes_status == "verified" and not out.research_notes_advisory_accepted
    assert out.research_notes_verification_version == 3 and critic.cache_output_is_reusable(out)


def test_over_budget_advisory_review_retains_verdict_and_usage(harness, runner):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.logs = False
    harness.ctx.budgets.root().spec = BudgetSpec(max_usd=.2)
    critic = ACCritic(harness.ctx)
    with pytest.raises(BudgetExhausted) as caught:
        runner.run(critic(**packet()))
    out = caught.value.completed_output
    assert out.answer_ready and out.research_notes_advisory_accepted
    assert not out.research_notes_execution_verified and critic._completed_review == out
    assert len(harness.calls) == 1 and harness.ctx.budgets.root().counters.usd == .25


@pytest.mark.parametrize("context_recovery", [False, True])
def test_cancelled_advisory_download_reuses_report_after_expiry(harness, runner, monkeypatch, context_recovery):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    harness.control.logs, harness.control.artifact = False, True
    if context_recovery:
        harness.control.query_error = ValueError("context_length_exceeded")
    original = APIClient.read_code_interpreter_file

    def interrupted(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(APIClient, "read_code_interpreter_file", interrupted)
    inp = packet(mode="stateful", round=2)
    with pytest.raises(asyncio.CancelledError):
        runner.run(ACCritic(harness.ctx)(**inp))
    monkeypatch.setattr(APIClient, "read_code_interpreter_file", original)
    harness.control.artifact_error = RuntimeError("Container expired")
    out = runner.run(ACCritic(harness.ctx)(**inp))
    assert out.answer_ready and out.research_notes_advisory_accepted
    assert not out.research_notes_execution_verified
    assert len(harness.calls) == 1 + context_recovery and harness.ctx.budgets.root().counters.usd == .25


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("raw_ready", [None, False, True])
def test_saved_output_recovery_uses_raw_verdict_not_old_acceptance_flag(harness, runner, policy, raw_ready):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    output = critic.Outputs(answer_ready=raw_ready is not True, research_notes_status="not_checked")
    if raw_ready is not None:
        output.messages_after = [{"role": "assistant", "content":
            f"<answer_ready>{str(raw_ready).lower()}</answer_ready>"}]
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": "completed", "kind": "ordinary",
        "review_input": inp.model_dump(mode="json"), "output": output.model_dump(mode="json")}))
    out = runner.run(critic(**packet()))
    assert out.answer_ready is (policy == "advisory" and raw_ready is True)
    assert out.research_notes_advisory_accepted is out.answer_ready
    assert out.research_notes_verification_policy == policy
    assert not out.research_notes_execution_verified
    assert not harness.calls and not harness.uploads


@pytest.mark.parametrize("ready", [False, True])
def test_verified_legacy_checkpoint_without_raw_report_keeps_verdict(harness, runner, ready):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    output = critic.Outputs(answer_ready=ready, research_notes_status="verified",
                            research_notes_execution_verified=True, research_notes_verification_version=2)
    evidence = [{"id": "ci-legacy", "status": "completed", "container_id": "container",
        "code": critic._notes_verification_code(inp),
        "outputs": [{"type": "logs", "logs": json.dumps(critic._notes_metadata(inp))}]}]
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": "completed", "kind": "ordinary", "tool_evidence": evidence,
        "review_input": inp.model_dump(mode="json"), "output": output.model_dump(mode="json")}))
    out = runner.run(critic(**packet()))
    assert out == output
    assert not harness.calls and not harness.uploads


def test_verification_policy_is_part_of_cache_and_recovery_identity(harness):
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    old_cache, old_checkpoint = critic._cache_key(inp), critic._recovery_checkpoint(inp)
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = "advisory"
    advisory = ACCritic(harness.ctx)
    assert advisory._cache_key(inp) != old_cache
    assert advisory._recovery_checkpoint(inp) != old_checkpoint


@pytest.mark.parametrize("preset", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_competition_presets_use_advisory_verification(preset, tmp_path):
    from proofstack.registry import load_preset

    config = load_preset(preset).component_configs
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs=config)
    critic = ACCritic(ctx, name="ACCritic")
    assert critic._notes_transport() == "file"
    assert critic._notes_verification_policy() == "advisory"
    inp = critic.Inputs(**packet())
    out = critic.parse_output("<answer_ready>true</answer_ready>", inp)
    assert out.answer_ready and out.research_notes_advisory_accepted
    assert not out.research_notes_execution_verified
    mismatch = critic.parse_output(
        "<answer_ready>true</answer_ready><research_notes_status>mismatch</research_notes_status>", inp)
    assert not mismatch.answer_ready and mismatch.research_notes_integrity_failed


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("first", ["cached", "in_memory", "missing"])
def test_provider_report_replay_restores_later_cached_mismatch(harness, runner, policy, first):
    from copy import deepcopy

    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    harness.control.logs, harness.control.artifact = False, True
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet(mode="stateful", round=2))
    assert runner.run(critic(**inp.model_dump())).answer_ready
    checkpoint = critic._recovery_checkpoint(inp)
    saved = json.loads(checkpoint.read_text())
    root = harness.ctx.root_workdir / saved["attempt_workdir"]
    report_path = next(root.glob("provider-completed-*.json"))
    report = json.loads(report_path.read_text())
    item = critic._provider_tool_evidence[0]
    items = []
    for index in range(2):
        current = deepcopy(item)
        current["id"] = f"checker-{index}"
        receipt = current.pop("notes_verification_receipt")
        receipt["call_id"] = current["id"]
        if index == 1:
            metadata = json.loads(receipt["content"])
            metadata["sha256"] = "wrong"
            receipt["content"] = json.dumps(metadata)
        if index == 1 or first == "cached":
            critic._notes_receipt_cache_path(inp, current, root).write_text(json.dumps(receipt))
        elif first == "in_memory":
            current["notes_verification_receipt"] = receipt
        items.append(current)
    report["tool_evidence"] = items
    report_path.write_text(json.dumps(report))
    saved.update(status="interrupted")
    saved.pop("output")
    saved.pop("tool_evidence")
    checkpoint.write_text(json.dumps(saved))
    # Replay must load the veto even when the deadline/remote container is gone.
    harness.control.artifact_error = RuntimeError("expired")
    resumed = runner.run(ACCritic(harness.ctx)(**inp.model_dump()))
    assert not resumed.answer_ready and resumed.research_notes_integrity_failed
    assert not resumed.research_notes_execution_verified and not resumed.research_notes_advisory_accepted
    assert len(harness.calls) == len(harness.downloads) == 1
    assert harness.ctx.budgets.root().counters.usd == .25
    assert runner.run(ACCritic(harness.ctx)(**inp.model_dump())) == resumed


@pytest.mark.parametrize("policy", ["required", "advisory"])
def test_receipt_download_does_not_stop_before_later_mismatch(harness, runner, policy):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    critic._provider_tool_evidence = [{"id": f"ci-{n}", "container_id": f"container-{n}",
        "status": "completed", "code": critic._notes_verification_code(inp), "outputs": []} for n in range(2)]
    calls = []
    def download(container_id, path, **kwargs):
        calls.append(container_id)
        metadata = critic._notes_metadata(inp)
        if container_id == "container-1":
            metadata["receipt_sha256"] = "wrong"
        return {"container_id": container_id, "file_id": "file", "path": path, "content": json.dumps(metadata)}
    runner.run(critic._retrieve_notes_verification(inp, client=SimpleNamespace(read_code_interpreter_file=download)))
    out = critic.parse_output(harness.control.reply, inp)
    assert calls == ["container-0", "container-1"]
    assert not out.answer_ready and out.research_notes_integrity_failed


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("proof", ["missing", "copied_sha256", "computed"])
def test_receipt_requires_a_digest_not_disclosed_in_prompt(harness, policy, proof):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    metadata = critic._notes_metadata(inp)
    prompt = critic._notes_context(inp)
    assert metadata["receipt_sha256"] not in prompt and metadata["sha256"] in prompt
    content = {k: metadata[k] for k in ("filename", "bytes", "sha256")}
    if proof != "missing":
        content["receipt_sha256"] = metadata["sha256"] if proof == "copied_sha256" else metadata["receipt_sha256"]
    item = {"id": "ci", "container_id": "container", "status": "completed",
            "code": critic._notes_verification_code(inp), "outputs": []}
    item["notes_verification_receipt"] = {"file_id": "file", "call_id": "ci", "container_id": "container",
        "path": critic._notes_receipt_file(inp), "content": json.dumps(content)}
    critic._provider_tool_evidence = [item]
    out = critic.parse_output(harness.control.reply, inp)
    assert out.research_notes_execution_verified is (proof == "computed")
    assert out.research_notes_integrity_failed is (proof == "copied_sha256")
    assert out.answer_ready is (proof == "computed" or (proof == "missing" and policy == "advisory"))
    assert out.research_notes_advisory_accepted is (proof == "missing" and policy == "advisory")


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("end", ["cancel", "timeout", "deadline", "terminated"])
def test_queued_upload_is_bounded_and_never_starts_after_abandonment(
        tmp_path, monkeypatch, runner, queued_notes_executor, explicit, end):
    from proofstack.agents.ac import critic as module
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    def forbidden(*a, **k):
        pytest.fail("Abandoned upload reached the provider")
    client = SimpleNamespace(terminated=False, attach_code_interpreter_file=forbidden,
        create_code_interpreter_container_with_file=forbidden)
    client.terminate = lambda: setattr(client, "terminated", True)

    async def exercise():
        task = asyncio.create_task(critic._attach_notes(client, "unused", explicit=explicit,
                                                      timeout=.01 if end == "timeout" else 10))
        await queued_notes_executor.submitted.wait()
        function, args, future = queued_notes_executor.queued[0]
        if end == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif end == "timeout":
            with pytest.raises(TimeoutError, match="upload wall-clock deadline"):
                await task
        else:
            if end == "deadline":
                clock[0] = 11
            else:
                client.terminated = True
        try:
            function(*args)
        except TimeoutError as exc:
            future.set_exception(exc)
        else:
            pytest.fail("Queued upload should have failed its entry guard")
        if end in {"deadline", "terminated"}:
            with pytest.raises(TimeoutError):
                await task
        assert client.terminated
        assert queued_notes_executor.shutdown == [(False, True)]
        assert queued_notes_executor.options == [{"max_workers": 1, "thread_name_prefix": "critic-notes-upload"}]
    runner.run(exercise())


def test_upload_result_racing_with_cancellation_is_discarded_once(tmp_path, runner, queued_notes_executor):
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    deleted = []
    client = SimpleNamespace(terminated=False, discard_code_interpreter_container=deleted.append)
    client.terminate = lambda: setattr(client, "terminated", True)

    async def exercise():
        def upload(*args, **kwargs):
            task.cancel()
            return "unused", "file"
        client.create_code_interpreter_container_with_file = upload
        task = asyncio.create_task(critic._attach_notes(client, "unused", explicit=True, timeout=float("inf")))
        await queued_notes_executor.submitted.wait()
        function, args, future = queued_notes_executor.queued[0]
        future.set_result(function(*args))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.terminated and deleted == ["unused"]
        assert queued_notes_executor.shutdown == [(False, True)]
    runner.run(exercise())


@pytest.mark.parametrize("policy", ["required", "advisory"])
@pytest.mark.parametrize("raw", [False, True])
def test_saved_integrity_failure_cannot_become_unconfirmed_acceptance(harness, runner, policy, raw):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    output = critic.Outputs(answer_ready=False, research_notes_status="not_checked",
                            research_notes_integrity_failed=True, research_notes_verification_policy=policy)
    if raw:
        output.messages_after = [{"role": "assistant", "content": harness.control.reply}]
    path = critic._recovery_checkpoint(inp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "completed", "kind": "ordinary",
        "review_input": inp.model_dump(mode="json"), "output": output.model_dump(mode="json")}))
    out = runner.run(critic(**packet()))
    assert not out.answer_ready and out.research_notes_integrity_failed
    assert not out.research_notes_execution_verified and not out.research_notes_advisory_accepted
    assert not harness.calls and not harness.uploads


@pytest.mark.parametrize("policy", ["required", "advisory"])
def test_provider_report_replay_preserves_saved_integrity_veto(harness, runner, policy):
    harness.ctx.component_configs.setdefault("ACCritic", {})["research_notes_verification"] = policy
    critic = ACCritic(harness.ctx)
    inp = critic.Inputs(**packet())
    assert runner.run(critic(**packet())).answer_ready
    path = critic._recovery_checkpoint(inp)
    saved = json.loads(path.read_text())
    saved["output"].update(answer_ready=False, research_notes_integrity_failed=True)
    path.write_text(json.dumps(saved))
    # The provider report has matching evidence but predates the durable veto.
    out = runner.run(ACCritic(harness.ctx)(**packet()))
    assert not out.answer_ready and out.research_notes_integrity_failed
    assert not out.research_notes_execution_verified and not out.research_notes_advisory_accepted
    assert len(harness.calls) == 1 and harness.ctx.budgets.root().counters.usd == .25
