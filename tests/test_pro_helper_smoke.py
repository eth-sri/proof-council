import importlib.util
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from proofstack.agents.ac.delegation_context import ContextView, DelegationContext
from proofstack.agents.ac.helper_sandbox import HelperSandbox


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MATHAGENTS_REQUEST_LOG_DIR", str(tmp_path / "request-logs"))
    spec = importlib.util.spec_from_file_location(
        "_pro_helper_smoke_test", Path(__file__).parents[1] / "scripts/pro_helper_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for cls, name in ((ContextView, "read"), (ContextView, "_publish"), (HelperSandbox, "checkpoint")):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    return module


def test_smoke_records_every_checkpoint_boundary_without_mutating_it(smoke, monkeypatch):
    receipts = [
        {"status": "passed", "errors": [], "source_container_id": "original", "captured": []},
        {"status": "passed", "errors": [], "source_container_id": "recovered", "captured": []},
        {"status": "failed", "errors": ["download_error"], "source_container_id": "later", "captured": []},
    ]
    seen = []

    def checkpoint(self, messages, call_deadline_monotonic_s=None):
        seen.append((self, messages, call_deadline_monotonic_s))
        return receipts[len(seen) - 1]

    monkeypatch.setattr(HelperSandbox, "checkpoint", checkpoint)
    smoke.instrument_context()
    sandbox = SimpleNamespace(view=SimpleNamespace(publisher="helpers/wave1-prover1"))
    for index, expected in enumerate(receipts):
        assert HelperSandbox.checkpoint(sandbox, [index], 123) is expected
    audit = [json.loads(line) for line in (smoke.ROOT / "tool-audit.jsonl").read_text().splitlines()]
    assert [row["checkpoint"] for row in audit] == receipts
    assert all(row["actor"] == sandbox.view.publisher and row["kind"] == "sandbox_checkpoint" for row in audit)
    assert [row["ok"] for row in audit] == [True, True, False]
    assert seen == [(sandbox, [index], 123) for index in range(3)]


def test_smoke_instrumentation_preserves_checkpoint_exception(smoke, monkeypatch):
    def checkpoint(*args, **kwargs):
        raise TimeoutError("closed")

    monkeypatch.setattr(HelperSandbox, "checkpoint", checkpoint)
    smoke.instrument_context()
    with pytest.raises(TimeoutError, match="closed"):
        HelperSandbox.checkpoint(SimpleNamespace(), [], 0)
    assert not (smoke.ROOT / "tool-audit.jsonl").exists()


def test_instrumented_context_read_preserves_revision_and_sandbox_reads(smoke, tmp_path):
    store = DelegationContext(tmp_path / "context", round=1)
    store.put("round/critic.md", "critic evidence", source="test")
    view = store.view(publisher="helpers/helper1")
    original_read = ContextView.read
    smoke.instrument_context()
    first = json.loads(view.read("manifest.json", 0, 30, None))
    view.publish("result.txt", "new publication between pages")
    expected = original_read(view, "manifest.json", 30, 12000, first["revision"])
    assert view.read("manifest.json", 30, 12000, revision=first["revision"]) == expected
    manifest = json.loads(first["content"] + json.loads(expected)["content"])
    assert [entry["path"] for entry in manifest["files"]] == ["round/critic.md"]

    sandbox = SimpleNamespace(view=view, _boundary=lambda *args: {"status": "passed"})
    result = json.loads(HelperSandbox.read(sandbox, "round/critic.md", revision=first["revision"]))
    assert result["content"] == "critic evidence"
    audit = [json.loads(line) for line in (smoke.ROOT / "tool-audit.jsonl").read_text().splitlines()]
    reads = [row for row in audit if row["kind"] == "read_context"]
    assert len(reads) == 3 and all(row["ok"] for row in reads)
    assert reads[1]["revision"] == first["revision"]


def test_instrumented_context_read_preserves_exceptions(smoke, monkeypatch):
    def read(*args, **kwargs):
        raise OSError("test read failed")
    monkeypatch.setattr(ContextView, "read", read)
    smoke.instrument_context()
    with pytest.raises(OSError, match="test read failed"):
        ContextView.read(SimpleNamespace(), "manifest.json", 0, 10, "revision")
    assert not (smoke.ROOT / "tool-audit.jsonl").exists()


def test_smoke_verify_passes_harness_receipts_to_recovery_check(smoke, monkeypatch):
    files, records, receipts, nonces = {}, [], [], []
    for n in (12, 13, 14, 15):
        agent_id = f"prover{n}"
        publisher = "helpers/wave1-" + agent_id
        cp_path, result_path = publisher + "/checkpoint.json", publisher + "/result.json"
        nonce, script = f"{n:032x}", "assert True"
        nonces.append(nonce)
        values = {"n": n, "sum": math.comb(2*n, n), "closed_form": math.comb(2*n, n),
                  "nonce": nonce, "critic_marker": "context-marker"}
        files[cp_path] = json.dumps({**values, "script": script})
        files[result_path] = json.dumps({**values, "restored_nonce": nonce, "recomputed": True,
                                        "marker_survived": False,
                                        "script_sha256": hashlib.sha256(script.encode()).hexdigest()})
        records.append({"agent_id": agent_id, "error": None, "report": "done", "usd": 0})
        checkpoint = {"source_container_id": agent_id, "status": "passed", "captured": []}
        receipts.append({"publisher": publisher, "checkpoint": checkpoint})
        smoke.record("sandbox_checkpoint", actor=publisher, checkpoint=checkpoint, ok=True)
        for actor, path in ((publisher, "round/critic.md"), (publisher, cp_path),
                            ("lead", cp_path), ("lead", result_path)):
            smoke.record("read_context", actor=actor, path=path, offset=0, next_offset=None, ok=True)
    checked = []

    def assess(files, metadata, publisher, cp_path, result_path, *, capture_receipts):
        assert capture_receipts == receipts
        checked.append(publisher)
        return {"status": "passed", "errors": []}

    monkeypatch.setattr(smoke, "assess_recovery", assess)
    author = SimpleNamespace(_delegation_log=records, _waves_done=1,
                             _context=SimpleNamespace(files=files, metadata={}))
    out = SimpleNamespace(research_notes_tex="LEAD_PREWAVE_MARKER " + " ".join(nonces),
                          via="container_files", artifact_status="changed", files_changed=["answer.tex"],
                          container_id="lead", compute_instructions="", council_question="")
    smoke.verify(author, out, "context-marker")
    assert checked == [receipt["publisher"] for receipt in receipts]
    assert smoke.REPORT["checks"]["recovery"]["status"] == "passed"
