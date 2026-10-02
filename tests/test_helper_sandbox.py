import asyncio
from contextlib import contextmanager
import json
import threading
import time
from types import SimpleNamespace

import pytest

from proofstack.agents.ac.delegation_context import DelegationContext
from proofstack.agents.ac.helper_sandbox import HelperSandbox, CHECKPOINT_ROOT


def messages(cid="first", call="ci1"):
    return [{"type": "code_interpreter_call", "id": call, "container_id": cid}]


class FilesAPI:
    def __init__(self):
        self.entries = {}
        self.bodies = {}
        self.uploads = []
        self.deleted = []
        self.downloads = []
        self.before_create = lambda: None
        self.list_error = None
        self.content = SimpleNamespace(with_streaming_response=SimpleNamespace(retrieve=self.retrieve))
        self.containers = SimpleNamespace(files=self)
        self.files = SimpleNamespace(create=self.create, delete=self.deleted.append)

    def add(self, cid, name, body, *, size=None, path=None):
        body = body.encode() if isinstance(body, str) else body
        fid = f"{cid}-{name}"
        self.entries.setdefault(cid, []).append(SimpleNamespace(
            id=fid, path=path or f"{CHECKPOINT_ROOT}/{name}", bytes=len(body) if size is None else size))
        self.bodies[cid, fid] = body

    def list(self, cid, **kwargs):
        if self.list_error:
            raise self.list_error
        return iter(self.entries.get(cid, []))

    @contextmanager
    def retrieve(self, *, container_id, file_id):
        self.downloads.append((container_id, file_id))
        data = self.bodies[container_id, file_id]
        yield SimpleNamespace(iter_bytes=lambda **kw: iter([data]))

    def create(self, *, file, purpose):
        self.before_create()
        self.uploads.append(file)
        return SimpleNamespace(id=f"file-{len(self.uploads)}")


def state(tmp_path, api=None, publisher="helpers/wave1-prover1", store=None):
    api = api or FilesAPI()
    store = store or DelegationContext(tmp_path / "context", round=1)
    view = store.view([], publisher=publisher)
    workdir = tmp_path / publisher
    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = HelperSandbox(view, workdir)
    descriptor = {"type": "auto", "file_ids": []}
    sandbox.bind(descriptor, lambda **kw: api)
    return sandbox, api, descriptor


def test_direct_publication_downloads_bytes_and_cannot_take_content_or_container(tmp_path):
    s, api, descriptor = state(tmp_path)
    api.add("first", "proof.tex", "actual sandbox proof")
    out = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", messages=messages()))
    assert out["status"] == "published"
    assert s.view.store.files[out["path"]] == "actual sandbox proof"
    assert out["provenance"]["container_id"] == "first"
    assert out["provenance"]["file_id"] == "first-proof.tex"
    assert out["provenance"]["call_id"] == "ci1"
    assert descriptor["file_ids"] == ["file-1"]
    assert s.view.store.metadata[out["path"]]["provenance"] == out["provenance"]
    with pytest.raises(TypeError):
        s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", content="invented", container_id="other")


def test_context_and_text_tools_checkpoint_and_reuse_capture(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "code.py", "print(1)")
    a = json.loads(s.read(messages=messages()))
    b = json.loads(s.publish("report.md", "claim", messages=messages()))
    assert a["sandbox_checkpoint"] == b["sandbox_checkpoint"]
    assert len(api.downloads) == 1 and len(api.uploads) == 1
    assert "provenance" not in b  # A model-authored report is not a sandbox receipt.


def test_reset_retains_attachments_and_requires_current_file_for_publication(tmp_path):
    s, api, descriptor = state(tmp_path)
    api.add("first", "marker.txt", "original")
    first = json.loads(s.read(messages=messages()))["sandbox_checkpoint"]
    restored = first["files"][0]
    assert restored["attachment_path"] == "/mnt/data/file-1-marker.txt"
    second = json.loads(s.read(messages=messages("second", "ci2")))["sandbox_checkpoint"]
    assert second["files"] == first["files"]
    assert descriptor["file_ids"] == ["file-1"]
    assert "error" in json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/marker.txt", messages=messages("second", "ci2")))
    api.add("second", "marker.txt", "original")
    published = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/marker.txt", messages=messages("second", "ci3")))
    assert published["status"] == "published"
    assert published["provenance"]["container_id"] == "second"
    assert descriptor["file_ids"] == ["file-1"]  # Reuse bytes, but record the fresh capture's provenance.


@pytest.mark.parametrize("path", ["/etc/passwd", "/mnt/data/checkpoints/../secret.txt",
    "/mnt/data/checkpoints/.key.txt", "/mnt/data/checkpoints/auth.json",
    "/mnt/data/checkpoints/nested/proof.txt", "/mnt/data/checkpoints/a.zip", "/mnt/data/checkpoints//a.txt"])
def test_rejects_unsafe_paths(tmp_path, path):
    s, api, _ = state(tmp_path)
    api.add("first", "bad", "secret", path=path)
    out = json.loads(s.publish_file(path, messages=messages()))
    assert "error" in out
    assert not api.downloads and not api.uploads


@pytest.mark.parametrize("case", ["oversize", "actual_oversize", "incomplete", "credentials", "non_utf8", "list_error"])
def test_failures_are_explicit_and_never_attach_bad_data(tmp_path, monkeypatch, case):
    s, api, descriptor = state(tmp_path)
    monkeypatch.setattr("proofstack.agents.ac.helper_sandbox.MAX_FILE_BYTES", 32)
    if case == "oversize":
        api.add("first", "big.txt", "x", size=33)
    elif case == "actual_oversize":
        api.add("first", "big.txt", "x" * 33, size=1)
    elif case == "incomplete":
        api.add("first", "partial.txt", "x", size=10)
    elif case == "credentials":
        api.add("first", "key.txt", "sk-" + "a" * 25)
    elif case == "non_utf8":
        api.add("first", "bad.txt", b"\xff")
    else:
        api.list_error = RuntimeError("do not expose sk-private")
    out = json.loads(s.read(messages=messages()))
    assert out["sandbox_checkpoint"]["status"] == "failed"
    assert s.failures
    assert not api.uploads and not descriptor["file_ids"]
    assert "sk-private" not in json.dumps(out)


def test_deadline_does_not_do_io(tmp_path):
    s, api, _ = state(tmp_path)
    out = json.loads(s.read(messages=messages(), call_deadline_monotonic_s=0))
    assert out["sandbox_checkpoint"]["status"] == "failed"
    assert not api.downloads


def test_scan_file_total_and_version_limits(tmp_path, monkeypatch):
    for variable, value, count in [("MAX_SCAN_ENTRIES", 1, 2), ("MAX_CHECKPOINT_FILES", 1, 2),
                                   ("MAX_CHECKPOINT_BYTES", 1, 1), ("MAX_CHECKPOINT_VERSIONS", 0, 1)]:
        with monkeypatch.context() as m:
            m.setattr("proofstack.agents.ac.helper_sandbox." + variable, value)
            s, api, _ = state(tmp_path / variable)
            for n in range(count):
                api.add("first", f"{n}.txt", "xx")
            assert json.loads(s.read(messages=messages()))["sandbox_checkpoint"]["status"] == "failed"


def test_identical_paths_in_concurrent_helpers_do_not_cross(tmp_path):
    store = DelegationContext(tmp_path / "context", round=1)
    a, api_a, desc_a = state(tmp_path, publisher="helpers/one", store=store)
    b, api_b, desc_b = state(tmp_path, publisher="helpers/two", store=store)
    api_a.add("a", "proof.txt", "proof A")
    api_b.add("b", "proof.txt", "proof B")
    async def run():
        return await asyncio.gather(
            asyncio.to_thread(a.publish_file, f"{CHECKPOINT_ROOT}/proof.txt", messages=messages("a")),
            asyncio.to_thread(b.publish_file, f"{CHECKPOINT_ROOT}/proof.txt", messages=messages("b")),
        )
    x, y = map(json.loads, asyncio.run(run()))
    assert store.files[x["path"]] == "proof A" and store.files[y["path"]] == "proof B"
    assert "error" in json.loads(a.view.read(y["path"]))
    assert "error" in json.loads(b.view.read(x["path"]))
    assert api_a.downloads == [("a", "a-proof.txt")]
    assert api_b.downloads == [("b", "b-proof.txt")]


def test_late_upload_after_close_is_deleted_and_never_attached(tmp_path):
    s, api, descriptor = state(tmp_path)
    api.add("first", "proof.txt", "proof")
    entered, release = threading.Event(), threading.Event()
    def slow():
        entered.set()
        assert release.wait(3)
    api.before_create = slow
    result = []
    worker = threading.Thread(target=lambda: result.append(s.read(messages=messages())))
    worker.start()
    try:
        assert entered.wait(1)
        assert s.close() == []
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert descriptor["file_ids"] == []
        assert api.deleted == ["file-1"]
        assert json.loads(result[0])["sandbox_checkpoint"]["status"] == "failed"
    finally:
        release.set()
        worker.join(3)


def test_same_name_revision_is_immutable_and_cleanup_deletes_all_versions(tmp_path):
    s, api, descriptor = state(tmp_path)
    api.add("first", "proof.txt", "one")
    first = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.txt", messages=messages()))
    api.entries["first"] = []
    api.add("first", "proof.txt", "two")
    second = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.txt", messages=messages(call="ci2")))
    assert "error" in second  # Caller must choose a new public filename for revisions.
    assert s.view.store.files[first["path"]] == "one"
    assert descriptor["file_ids"] == ["file-2"]
    s._delete(s.close())
    assert set(api.deleted) == {"file-1", "file-2"}


def test_no_hosted_execution_does_not_claim_verified_checkpoints(tmp_path):
    s, api, _ = state(tmp_path)
    assert json.loads(s.read(messages=[]))["sandbox_checkpoint"]["status"] == "not_started"
    assert "error" in json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.txt", messages=[]))
    assert not api.uploads


def test_new_execution_refreshes_receipt_without_reuploading_same_bytes(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "marker.txt", "original")
    a = json.loads(s.read(messages=messages()))["sandbox_checkpoint"]["files"][0]
    b = json.loads(s.read(messages=messages(call="ci2")))["sandbox_checkpoint"]["files"][0]
    assert a["provenance"]["call_id"] == "ci1" and b["provenance"]["call_id"] == "ci2"
    assert a["artifact_path"] == b["artifact_path"]
    assert len(api.downloads) == 2 and len(api.uploads) == 1


def test_smoke_verifier_accepts_real_deduplicated_capture_receipts(tmp_path):
    from proofstack.agents.ac.helper_smoke_check import assess_recovery

    s, api, _ = state(tmp_path)
    nonce = "a" * 32
    api.add("first", "smoke-marker.txt", nonce)
    api.add("first", "checkpoint.json", json.dumps({"nonce": nonce}))
    cp = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/checkpoint.json", messages=messages()))
    api.add("second", "smoke-marker.txt", nonce)
    api.add("second", "result.json", json.dumps({"nonce": nonce, "restored_nonce": nonce}))
    result = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/result.json", messages=messages("second", "ci2")))
    store = s.view.store
    args = (store.files, store.metadata, s.view.publisher, cp["path"], result["path"])
    assert assess_recovery(*args)["status"] == "unverified"
    receipts = [{"publisher": s.view.publisher, "checkpoint": reply["sandbox_checkpoint"]}
                for reply in (cp, result)]
    assert assess_recovery(*args, capture_receipts=receipts) == {"status": "passed", "errors": []}
    assert api.downloads == [("first", "first-smoke-marker.txt"), ("first", "first-checkpoint.json"),
                             ("second", "second-smoke-marker.txt"), ("second", "second-result.json")]
    assert len(api.uploads) == 3  # The new receipt does not require another copy of the marker.


def test_failed_upload_leaves_durable_capture_but_not_a_successful_attachment(tmp_path):
    s, api, desc = state(tmp_path)
    api.add("first", "proof.txt", "proof")
    def fail():
        raise RuntimeError("upload failed")
    api.before_create = fail
    result = json.loads(s.read(messages=messages()))
    assert result["sandbox_checkpoint"]["status"] == "failed"
    assert not desc["file_ids"]
    assert any(meta.get("provenance", {}).get("kind") == "sandbox_file" for meta in s.view.store.metadata.values())


def test_existing_text_artifact_cannot_be_promoted_to_downloaded_provenance(tmp_path):
    s, api, _ = state(tmp_path)
    s.view.publish("proof.txt", "same bytes")
    api.add("first", "proof.txt", "same bytes")
    result = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.txt", messages=messages()))
    assert "error" in result
    assert "provenance" not in s.view.store.metadata["helpers/wave1-prover1/proof.txt"]


def test_deleted_file_is_not_published_as_a_fresh_capture(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "proof.txt", "earlier proof")
    s.read(messages=messages())
    api.entries["first"] = []
    out = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.txt", messages=messages(call="ci2")))
    assert "error" in out
    assert "helpers/wave1-prover1/proof.txt" not in s.view.store.files


@pytest.mark.parametrize("aux", ["scan.cpp", "__pycache__/scan.cpython-312.pyc", "scratch.zip"])
@pytest.mark.parametrize("bad_first", [True, False])
def test_auxiliary_files_do_not_block_a_valid_publication(tmp_path, aux, bad_first):
    s, api, _ = state(tmp_path)
    entries = [(aux, "auxiliary"), ("proof.tex", "valid proof")]
    for name, body in entries if bad_first else reversed(entries):
        api.add("first", name, body)
    out = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", messages=messages()))
    assert out["status"] == "published"
    assert out["sandbox_checkpoint"]["status"] == "passed"
    assert s.view.store.files[out["path"]] == "valid proof"


def test_partial_download_failure_does_not_block_fresh_valid_file(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "broken.txt", "short", size=10)
    api.add("first", "proof.tex", "valid")
    out = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", messages=messages()))
    assert out["status"] == "published"
    assert out["sandbox_checkpoint"]["status"] == "failed"
    assert s.failures


def test_unchanged_files_do_not_consume_versions_or_context_capacity(tmp_path):
    s, api, _ = state(tmp_path)
    for n in range(8):
        api.add("first", f"{n}.txt", "unchanged")
    for boundary in range(12):
        out = json.loads(s.read(messages=messages(call=f"ci{boundary}")))
        assert out["sandbox_checkpoint"]["status"] == "passed"
        assert all(r["provenance"]["call_id"] == f"ci{boundary}" for r in out["sandbox_checkpoint"]["captured"])
    assert len(s._versions) == len(s.view.store.files) == len(api.uploads) == 8


def test_duplicate_requested_path_is_never_published(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "proof.tex", "one")
    api.add("first", "proof.tex", "two")
    out = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", messages=messages()))
    assert "error" in out
    assert not api.downloads


def test_failed_upload_can_retry_without_poisoning_immutable_capture(tmp_path):
    s, api, _ = state(tmp_path)
    api.add("first", "proof.tex", "valid")
    def fail():
        raise RuntimeError("secret provider diagnostic")
    api.before_create = fail
    first = json.loads(s.read(messages=messages()))
    assert "upload_error" in first["sandbox_checkpoint"]["errors"]
    api.before_create = lambda: None
    second = json.loads(s.publish_file(f"{CHECKPOINT_ROOT}/proof.tex", messages=messages(call="ci2")))
    assert second["status"] == "published"
    assert second["provenance"]["call_id"] == "ci2"
