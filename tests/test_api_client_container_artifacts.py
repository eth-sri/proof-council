"""Bounded reads of existing hosted artifacts, without model calls."""
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import threading
import tempfile
from pathlib import Path

import pytest

from mathagents.api_client import APIClient


@pytest.fixture
def storage(monkeypatch):
    row = SimpleNamespace(id="cfile-test", container_id="container-test", bytes=2,
                          path="/mnt/data/check.json")
    state = SimpleNamespace(pages=[SimpleNamespace(data=[row], has_more=False)],
                            chunks=[b"{}"], listed=[], retrieved=[], closed=0,
                            options=[], listing_hook=None, row=row)

    class SDK:
        def __init__(self, **kwargs):
            state.options.append(kwargs)
            self.containers = SimpleNamespace(files=self)
            self.content = SimpleNamespace(with_streaming_response=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed += 1

        def list(self, container_id, **kwargs):
            state.listed.append((container_id, kwargs))
            if state.listing_hook:
                state.listing_hook()
            return state.pages[min(len(state.listed) - 1, len(state.pages) - 1)]

        def retrieve(self, file_id, **kwargs):
            state.retrieved.append((file_id, kwargs))
            return self

        def iter_bytes(self, chunk_size):
            assert chunk_size == 4097
            yield from state.chunks

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("mathagents.api_client.OpenAI", SDK)
    state.client = APIClient(model="synthetic", use_openai_responses_api=True,
                             base_url="https://synthetic.invalid/v1")
    return state


def download(storage, **kwargs):
    return storage.client.read_code_interpreter_file("container-test", "/mnt/data/check.json", **kwargs)


@pytest.mark.parametrize("declared_bytes", [2, None])
def test_artifact_download_uses_same_endpoint_and_streams_exact_file(storage, declared_bytes):
    storage.row.bytes = declared_bytes
    got = download(storage, timeout=2)
    assert got == {"container_id": "container-test", "file_id": "cfile-test",
                   "path": "/mnt/data/check.json", "content": "{}"}
    assert storage.options[0]["base_url"] == "https://synthetic.invalid/v1"
    assert storage.options[0]["api_key"] == "synthetic-test-key"
    assert storage.options[0]["max_retries"] == 0
    assert 0 < storage.retrieved[0][1]["timeout"] <= 2
    assert storage.closed == 2


def test_listing_paginates_before_deciding_uniqueness(storage):
    first = SimpleNamespace(id="other", path="/mnt/data/other.txt")
    storage.pages.insert(0, SimpleNamespace(data=[first], has_more=True))
    assert download(storage)["content"] == "{}"
    assert storage.listed[1][1]["after"] == "other"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "other_container", "oversize_metadata",
                                  "empty_page", "unknown_pagination", "repeated_cursor", "page_limit"])
def test_unusable_listing_never_downloads(storage, fault):
    if fault == "missing":
        storage.pages[0].data = []
    elif fault == "duplicate":
        storage.pages.append(SimpleNamespace(data=[storage.row], has_more=False))
        storage.pages[0].has_more = True
    elif fault == "other_container":
        storage.row.container_id = "old-container"
    elif fault == "oversize_metadata":
        storage.row.bytes = 4097
    elif fault == "empty_page":
        storage.pages[0] = SimpleNamespace(data=[], has_more=True)
    elif fault == "unknown_pagination":
        storage.pages[0].has_more = None
    elif fault == "repeated_cursor":
        storage.pages[0].has_more = True
    elif fault == "page_limit":
        storage.pages = [SimpleNamespace(data=[SimpleNamespace(id=str(i), path="other")], has_more=True)
                         for i in range(11)]
    with pytest.raises(ValueError):
        download(storage)
    assert not storage.retrieved and len(storage.listed) <= 10


@pytest.mark.parametrize("declared_bytes", [2, None])
@pytest.mark.parametrize("chunks", [[b"a" * 4097], [b"a" * 4096, b"b"], [b"\xff"]])
def test_content_is_bounded_and_strict_utf8(storage, chunks, declared_bytes):
    storage.row.bytes = declared_bytes
    storage.chunks = chunks
    with pytest.raises(ValueError):
        download(storage)
    assert storage.closed == 2


def test_hung_storage_call_has_an_overall_deadline(storage):
    release = threading.Event()
    storage.listing_hook = lambda: release.wait(3)
    try:
        with pytest.raises(TimeoutError):
            download(storage, timeout=0.05)
        assert not storage.retrieved
    finally:
        release.set()


def test_wrong_provider_is_rejected_before_storage_io(storage):
    storage.client.use_openai_responses_api = False
    with pytest.raises(ValueError, match="Responses"):
        download(storage)
    assert not storage.options


@pytest.mark.parametrize("stage", ["create", "upload"])
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_abandoned_explicit_setup_discards_late_container(tmp_path, monkeypatch, stage, stop):
    entered, release, deleted = threading.Event(), threading.Event(), threading.Event()
    actions = []
    class SDK:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.containers = SimpleNamespace(create=self.create, files=SimpleNamespace(create=self.upload),
                                              delete=self.delete)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            actions.append("create")
            if stage == "create":
                entered.set()
                assert release.wait(3)
            return SimpleNamespace(id="unused-container")
        def upload(self, container_id, **kwargs):
            actions.append("upload")
            entered.set()
            assert release.wait(3)
            return SimpleNamespace(id="unused-file")
        def delete(self, container_id):
            assert container_id == "unused-container"
            actions.append("delete")
            deleted.set()
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("mathagents.api_client.OpenAI", SDK)
    client = APIClient(model="synthetic", use_openai_responses_api=True,
                      tools=[(None, {"type": "code_interpreter", "container": {"type": "auto"}})])
    snapshot = tmp_path / "notes.tex"
    snapshot.write_text("Synthetic notes")
    errors = []
    def setup():
        try:
            client.create_code_interpreter_container_with_file(snapshot, timeout=.1 if stop == "timeout" else 2)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=setup)
    worker.start()
    try:
        assert entered.wait(2)
        if stop == "cancel":
            client.terminate()
        worker.join(1)
        assert not worker.is_alive() and len(errors) == 1
        assert client.tool_descriptions[0]["container"] == {"type": "auto"}
    finally:
        release.set()
        worker.join(3)
    assert deleted.wait(2)
    assert actions.count("delete") == 1
    if stage == "create":
        assert "upload" not in actions


def test_real_sdk_container_listing_and_streamed_content():
    # Other modules install global SDK stubs during collection.
    result = subprocess.run([sys.executable, "-c",
        "import runpy, sys, pytest\n"
        "with pytest.MonkeyPatch.context() as patch:\n"
        "    runpy.run_path(sys.argv[1])['_check_real_sdk'](patch)", __file__],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _check_real_sdk(monkeypatch):
    from openai import OpenAI, _base_client

    http = vars(_base_client).get("httpx2") or vars(_base_client)["httpx"]
    requests, deleted = [], threading.Event()

    def handle(request):
        requests.append(request)
        if request.url.path == "/v1/containers":
            assert request.method == "POST"
            assert json.loads(request.content) == {"name": "critic-notes",
                "expires_after": {"anchor": "last_active_at", "minutes": 20}}
            return http.Response(200, json={"id": "container-test", "object": "container",
                "name": "critic-notes", "created_at": 1, "status": "running"})
        if request.url.path == "/v1/containers/container-test":
            if request.method == "DELETE":
                deleted.set()
                return http.Response(200, json={"id": "container-test", "object": "container.deleted", "deleted": True})
            assert request.method == "GET"
            return http.Response(200, json={"id": "container-test", "object": "container",
                "name": "critic-notes", "created_at": 1, "status": "running"})
        if request.url.path == "/v1/containers/container-test/files" and request.method == "POST":
            assert "multipart/form-data" in request.headers["content-type"]
            assert b"Synthetic notes" in request.content and b'filename="notes.tex"' in request.content
            return http.Response(200, json={"id": "uploaded-file", "container_id": "container-test",
                "object": "container.file", "bytes": 15, "path": "/mnt/data/notes.tex", "created_at": 1,
                "source": "user"})
        assert request.method == "GET"
        if request.url.path == "/v1/containers/container-test/files":
            return http.Response(200, json={"object": "list", "has_more": False, "data": [{
                "id": "cfile-test", "container_id": "container-test", "bytes": None,
                "path": "/mnt/data/check.json", "object": "container.file", "created_at": 1,
                "source": "assistant"}]})
        assert request.url.path == "/v1/containers/container-test/files/cfile-test/content"
        return http.Response(200, content=b"{}", headers={"content-type": "application/binary"})

    def sdk(**kwargs):
        return OpenAI(**kwargs, http_client=http.Client(transport=http.MockTransport(handle)))

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("mathagents.api_client.OpenAI", sdk)
    client = APIClient(model="synthetic", use_openai_responses_api=True,
                       base_url="https://synthetic.invalid/v1",
                       tools=[(None, {"type": "code_interpreter", "container": {"type": "auto"}})])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "notes.tex"
        path.write_text("Synthetic notes")
        assert client.create_code_interpreter_container_with_file(path, timeout=2) == ("container-test", "uploaded-file")
    assert client.tool_descriptions[0]["container"] == "container-test"
    assert client.touch_code_interpreter_container("container-test", timeout=2) == "running"
    got = client.read_code_interpreter_file("container-test", "/mnt/data/check.json")
    assert got["content"] == "{}" and len(requests) == 5
    client.terminate()
    client.discard_code_interpreter_container("container-test")
    assert deleted.wait(2) and len(requests) == 6
