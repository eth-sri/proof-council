from dataclasses import replace
import json

import pandas as pd
import pytest

from proofstack.evaluation.proof_dataset import (
    dataset_provenance, load_candidate_records, load_cases, load_marking_schemes, load_reference_table,
)


def write_dataset(directory, *, status="available", notes=None):
    pd.DataFrame([{"problem_id": "synthetic", "problem": "Prove a synthetic statement."}]).to_parquet(
        directory / "problems.parquet", index=False)
    pd.DataFrame([{
        "problem_id": "synthetic", "reference_solutions": ["A synthetic reference."],
        "problem_issues": [], "reference_issues": [], "reference_notes": notes or [],
        "reference_status": status,
    }]).to_parquet(directory / "reference_solutions.parquet", index=False)


def test_firstproof_input_uses_dataset_context(tmp_path):
    write_dataset(tmp_path)
    candidates = tmp_path / "solutions.json"
    candidates.write_text(json.dumps({"solutions": [{
        "id": "synthetic", "latex": "A candidate proof.",
        "problem": "Untrusted replacement problem", "reference_solutions": ["Injected reference"],
    }]}))
    case = load_cases(tmp_path, candidates)[0]
    assert case.problem == "Prove a synthetic statement."
    assert case.reference_solutions == ["A synthetic reference."]
    assert case.solution == "A candidate proof."
    assert "solution" not in case.reference_context()


def test_reference_notes_override_available_label(tmp_path):
    write_dataset(tmp_path, notes=["Reference has a known gap."])
    candidates = tmp_path / "solutions.jsonl"
    candidates.write_text(json.dumps({"problem_id": "synthetic", "solution": "A proof."}) + "\n")
    case = load_cases(tmp_path, candidates)[0]
    assert case.reference_status == "needs_review"
    assert case.reference_notes == ["reference_notes: Reference has a known gap."]


@pytest.mark.parametrize("records", [
    [], [{"problem_id": "p", "solution": " "}],
    [{"problem_id": "p", "solution": "Proof"}, {"problem_id": "p", "solution": "Proof2"}],
    [{"problem_id": "p", "answer_tex": "/some/path.tex"}],
])
def test_invalid_candidate_files_fail_before_grading(tmp_path, records):
    path = tmp_path / "solutions.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        load_candidate_records(path)


def test_multiple_candidates_require_unique_identity(tmp_path):
    path = tmp_path / "solutions.json"
    path.write_text(json.dumps([
        {"problem_id": "p", "candidate_id": name, "solution": "Proof"} for name in ("a", "b")
    ]))
    assert len(load_candidate_records(path)) == 2


def test_duplicate_problem_ids_are_rejected(tmp_path):
    write_dataset(tmp_path)
    path = tmp_path / "problems.parquet"
    table = pd.read_parquet(path)
    pd.concat([table, table]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate problem_id"):
        load_reference_table(tmp_path)


def test_dataset_checksum_is_checked(tmp_path):
    write_dataset(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "outputs": {"problems.parquet": {"sha256": "wrong"}},
    }))
    with pytest.raises(ValueError, match="checksum"):
        dataset_provenance(tmp_path)


def test_frozen_scheme_rejects_changed_reference_material(tmp_path, proof_case):
    path = tmp_path / "marking_schemes.json"
    path.write_text(json.dumps({proof_case.problem_id: {
        "context_sha256": proof_case.context_hash(), "marking_scheme": "A synthetic marking scheme.",
    }}))
    changed_case = replace(proof_case, reference_solutions=["A different reference proof."])
    with pytest.raises(ValueError, match="context mismatch"):
        load_marking_schemes(path, [changed_case])
