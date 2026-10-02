import hashlib
import json
from copy import deepcopy

import pytest

from proofstack.agents.ac.helper_smoke_check import assess_recovery


PUBLISHER = "helpers/wave1-prover1"
NONCE = "a" * 32


def evidence():
    files, metadata = {}, {}
    def add(name, body, container, call, sandbox_name):
        path = PUBLISHER + "/" + name
        files[path] = body
        metadata[path] = {"source": PUBLISHER, "sha256": hashlib.sha256(body.encode()).hexdigest(),
                          "provenance": {"kind": "sandbox_file", "container_id": container,
                                         "call_id": call, "file_id": "cf-" + name,
                                         "sandbox_path": "/mnt/data/checkpoints/" + sandbox_name}}
        return path
    cp = add("checkpoint.json", json.dumps({"nonce": NONCE}), "old", "ci1", "checkpoint.json")
    result = add("result.json", json.dumps({"nonce": NONCE, "restored_nonce": NONCE}), "new", "ci2", "result.json")
    add("checkpoints/old/marker.txt", NONCE, "old", "ci1", "smoke-marker.txt")
    add("checkpoints/new/marker.txt", NONCE, "new", "ci2", "smoke-marker.txt")
    return files, metadata, cp, result


def assess(data, *, capture_receipts=()):
    files, meta, cp, result = data
    return assess_recovery(files, meta, PUBLISHER, cp, result, capture_receipts=capture_receipts)


def test_requires_downloads_from_original_and_fresh_sandbox():
    assert assess(evidence()) == {"status": "passed", "errors": []}


@pytest.mark.parametrize("field,value", [("original_nonce_verified", False), ("protocol_failure", "missing original UUID")])
@pytest.mark.parametrize("artifact", ["checkpoint", "result"])
def test_explicit_failure_is_not_ignored_even_when_values_agree(field, value, artifact):
    data = evidence()
    files, meta, checkpoint, result = data
    path = checkpoint if artifact == "checkpoint" else result
    files[path] = json.dumps({"nonce": NONCE, "restored_nonce": NONCE, field: value})
    meta[path]["sha256"] = hashlib.sha256(files[path].encode()).hexdigest()
    assert assess(data)["status"] == "failed"


def test_transcribed_matching_artifacts_are_not_execution_evidence():
    data = evidence()
    for meta in data[1].values():
        meta.pop("provenance")
    assert assess(data)["status"] == "unverified"


@pytest.mark.parametrize("which", ["old", "new"])
def test_missing_independent_marker_is_unverified(which):
    data = evidence()
    data[1].pop(PUBLISHER + f"/checkpoints/{which}/marker.txt")
    assert assess(data)["status"] == "unverified"


def test_identical_container_does_not_test_reset():
    data = evidence()
    for meta in data[1].values():
        meta["provenance"]["container_id"] = "old"
    assert assess(data)["status"] == "unverified"


def test_changed_nonce_is_failure_even_if_helper_claims_success():
    data = evidence()
    files, meta, _, _ = data
    path = PUBLISHER + "/checkpoints/new/marker.txt"
    files[path] = "b" * 32
    meta[path]["sha256"] = hashlib.sha256(files[path].encode()).hexdigest()
    assert assess(data)["status"] == "failed"


def test_hash_mismatch_is_failure():
    data = evidence()
    data[0][data[2]] += " "
    assert assess(data)["status"] == "failed"


def test_other_helpers_receipt_does_not_supply_missing_original():
    data = evidence()
    data[1][PUBLISHER + "/checkpoints/old/marker.txt"]["source"] = "helpers/another"
    assert assess(data)["status"] == "unverified"


@pytest.mark.parametrize("replacement", [None, "", "not json", "[]"])
def test_absent_or_malformed_checkpoint_never_passes(replacement):
    data = evidence()
    if replacement is None:
        del data[0][data[2]]
    else:
        data[0][data[2]] = replacement
    assert assess(data)["status"] != "passed"


def deduplicated_evidence():
    data = evidence()
    files, meta, _, _ = data
    fresh_path = PUBLISHER + "/checkpoints/new/marker.txt"
    original_path = PUBLISHER + "/checkpoints/old/marker.txt"
    fresh_provenance = meta.pop(fresh_path)["provenance"]
    del files[fresh_path]
    entry = {"artifact_path": original_path, "sandbox_path": fresh_provenance["sandbox_path"],
             "provenance": fresh_provenance, "sha256": meta[original_path]["sha256"],
             "bytes": len(files[original_path].encode())}
    receipt = {"publisher": PUBLISHER, "checkpoint": {
        "status": "passed", "errors": [], "source_container_id": "new", "source_call_id": "ci2",
        "captured": [entry], "files": [deepcopy(entry)],
    }}
    return data, receipt


def test_deduplicated_bytes_require_fresh_download_receipt():
    data, receipt = deduplicated_evidence()
    before = deepcopy(data)
    assert assess(data)["status"] == "unverified"
    assert assess(data, capture_receipts=[receipt]) == {"status": "passed", "errors": []}
    assert data == before  # Do not replace the immutable original provenance.


@pytest.mark.parametrize("field,value", [
    ("publisher", "helpers/wave1-other"), ("source_container_id", "old"),
    ("source_call_id", "ci1"), ("status", "failed"), ("errors", ["download failed"]),
    ("captured", []), ("captured", None), ("captured", [None]),
])
def test_missing_stale_or_failed_boundary_cannot_verify_recovery(field, value):
    data, receipt = deduplicated_evidence()
    (receipt if field == "publisher" else receipt["checkpoint"])[field] = value
    assert assess(data, capture_receipts=[receipt])["status"] != "passed"


@pytest.mark.parametrize("field,value", [
    ("container_id", "old"), ("call_id", "ci1"), ("file_id", ""),
    ("file_id", 7), ("kind", "model_report"), ("sandbox_path", "/tmp/smoke-marker.txt"),
])
def test_fresh_marker_receipt_requires_complete_matching_provenance(field, value):
    data, receipt = deduplicated_evidence()
    receipt["checkpoint"]["captured"][0]["provenance"][field] = value
    assert assess(data, capture_receipts=[receipt])["status"] != "passed"


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("bytes", 31), ("bytes", "32"), ("bytes", True),
])
def test_capture_hash_and_byte_count_must_match_downloaded_content(field, value):
    data, receipt = deduplicated_evidence()
    receipt["checkpoint"]["captured"][0][field] = value
    assert assess(data, capture_receipts=[receipt])["status"] == "failed"


@pytest.mark.parametrize("case", ["other_owner", "missing", "hash", "changed_nonce", "transcribed"])
def test_capture_does_not_bypass_content_or_ownership_checks(case):
    data, receipt = deduplicated_evidence()
    entry = receipt["checkpoint"]["captured"][0]
    path = entry["artifact_path"]
    if case == "other_owner":
        data[1][path]["source"] = "helpers/other"
    elif case == "missing":
        del data[0][path]
    elif case == "hash":
        data[1][path]["sha256"] = "0" * 64
    elif case == "changed_nonce":
        data[0][path] = "b" * 32
        entry["sha256"] = data[1][path]["sha256"] = hashlib.sha256(data[0][path].encode()).hexdigest()
    else:
        data[1][path].pop("provenance")
    assert assess(data, capture_receipts=[receipt])["status"] != "passed"


@pytest.mark.parametrize("path", [
    "helpers/other/checkpoints/marker.txt", PUBLISHER + "/result.json",
    PUBLISHER + "/checkpoints/../marker.txt", PUBLISHER + "/checkpoints//marker.txt",
    PUBLISHER + "/checkpoints/\\marker.txt", None,
])
def test_capture_artifact_path_must_be_in_helpers_checkpoint_namespace(path):
    data, receipt = deduplicated_evidence()
    receipt["checkpoint"]["captured"][0]["artifact_path"] = path
    assert assess(data, capture_receipts=[receipt])["status"] != "passed"


def test_retained_files_are_not_fresh_capture_evidence():
    data, receipt = deduplicated_evidence()
    del receipt["checkpoint"]["captured"]
    assert receipt["checkpoint"]["files"]
    assert assess(data, capture_receipts=[receipt])["status"] == "unverified"


def test_model_supplied_receipt_is_not_used():
    data, receipt = deduplicated_evidence()
    path = data[3]
    report = json.loads(data[0][path])
    report["capture_receipts"] = [receipt]
    data[0][path] = json.dumps(report)
    data[1][path]["sha256"] = hashlib.sha256(data[0][path].encode()).hexdigest()
    assert assess(data)["status"] == "unverified"


def test_later_boundary_does_not_hide_matching_receipt():
    data, receipt = deduplicated_evidence()
    later = deepcopy(receipt)
    later["checkpoint"].update(source_container_id="later", source_call_id="ci3", captured=[])
    assert assess(data, capture_receipts=[receipt, later])["status"] == "passed"


@pytest.mark.parametrize("receipt", [None, "model claim", {"publisher": PUBLISHER, "checkpoint": None}])
def test_malformed_capture_receipts_are_unverified(receipt):
    data, _ = deduplicated_evidence()
    assert assess(data, capture_receipts=[receipt])["status"] == "unverified"


def test_explicit_failed_boundary_is_not_overridden_by_another_receipt():
    data, receipt = deduplicated_evidence()
    failed = deepcopy(receipt)
    failed["checkpoint"]["status"] = "failed"
    assert assess(data, capture_receipts=[failed, receipt])["status"] == "failed"
