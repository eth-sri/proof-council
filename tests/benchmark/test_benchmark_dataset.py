import json

import pandas as pd
import pytest

from proofstack.evaluation.benchmark_dataset import select_problems


def write_dataset(directory, *, reverse=False):
    records = [
        ("a", "Statement A", "available", [], ["Secret reference A"]),
        ("b", "Statement B", "available", [], ["Secret reference B"]),
        ("c", "Statement A", "available", [], ["Secret reference C"]),
        ("d", "Statement D", "needs_review", [], ["Secret reference D"]),
        ("e", "Statement E", "available", ["Secret issue"], ["Secret reference E"]),
        ("f", "Statement F", "available", [], ["Secret reference F1", "Secret reference F2"]),
    ]
    if reverse:
        records.reverse()
    pd.DataFrame([{"problem_id": key, "problem": problem} for key, problem, *_ in records]).to_parquet(
        directory / "problems.parquet", index=False)
    pd.DataFrame([{
        "problem_id": key, "reference_status": status, "reference_solutions": references,
        "problem_issues": notes, "reference_issues": [], "reference_notes": [],
    } for key, _, status, notes, references in records]).to_parquet(
        directory / "reference_solutions.parquet", index=False)


def test_selection_excludes_flags_and_deduplicates_statements(tmp_path):
    write_dataset(tmp_path)
    selection = select_problems(tmp_path, limit=None)
    assert selection.records == [
        {"problem_id": "a", "problem": "Statement A"},
        {"problem_id": "b", "problem": "Statement B"},
    ]
    assert selection.report["excluded_flagged_count"] == 3
    assert selection.report["duplicate_statement_count"] == 1
    assert selection.report["eligible_problem_count"] == 3
    assert selection.report["eligible_unique_statement_count"] == 2
    assert "Secret" not in json.dumps(selection.records)
    assert "Secret" not in json.dumps(selection.report)
    assert set(selection.provenance["files"]) == {"problems.parquet", "reference_solutions.parquet"}


def test_sampling_is_reproducible_and_independent_of_input_order(tmp_path):
    write_dataset(tmp_path)
    original = select_problems(tmp_path, limit=1, seed=13)
    write_dataset(tmp_path, reverse=True)
    reordered = select_problems(tmp_path, limit=1, seed=13)
    assert original.records == reordered.records
    assert original.report == reordered.report
    assert len(original.records) == 1


def test_explicit_selection_uses_all_requested_ids_and_preserves_noncanonical_id(tmp_path):
    write_dataset(tmp_path)
    selection = select_problems(tmp_path, limit=1, problem_ids=["c", "b"])
    assert selection.report["selected_problem_ids"] == ["b", "c"]
    assert selection.report["mode"] == "explicit"
    assert selection.report["limit"] is None


@pytest.mark.parametrize("problem_ids, message", [
    (["a", "a"], "Duplicate problem IDs"),
    (["unknown"], "Unknown problem IDs"),
    (["d"], "Flagged references"),
    (["e"], "Flagged references"),
    (["f"], "Flagged references"),
    (["a", "c"], "duplicate problem statements"),
    ([], "nonempty list"),
    ("a", "nonempty list"),
    ([" "], "nonempty string"),
])
def test_invalid_explicit_selection_fails(tmp_path, problem_ids, message):
    write_dataset(tmp_path)
    with pytest.raises(ValueError, match=message):
        select_problems(tmp_path, problem_ids=problem_ids)


def test_dataset_with_only_flagged_references_has_no_eligible_problems(tmp_path):
    write_dataset(tmp_path)
    for name in ("problems.parquet", "reference_solutions.parquet"):
        path = tmp_path / name
        table = pd.read_parquet(path)
        table[table.problem_id.isin(["d", "e", "f"])].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="No eligible benchmark problems"):
        select_problems(tmp_path, limit=None)


@pytest.mark.parametrize("options", [
    {"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1.5},
    {"seed": True}, {"seed": "1"}, {"seed": None}, {"limit": 3},
])
def test_invalid_sampling_fails(tmp_path, options):
    write_dataset(tmp_path)
    with pytest.raises(ValueError):
        select_problems(tmp_path, **options)


def test_duplicate_dataset_ids_are_rejected(tmp_path):
    write_dataset(tmp_path)
    path = tmp_path / "problems.parquet"
    table = pd.read_parquet(path)
    pd.concat([table, table.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate problem_id"):
        select_problems(tmp_path)


def test_missing_references_are_rejected(tmp_path):
    write_dataset(tmp_path)
    path = tmp_path / "reference_solutions.parquet"
    table = pd.read_parquet(path)
    table.at[0, "reference_solutions"] = []
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="must contain a reference"):
        select_problems(tmp_path)


def test_selection_verifies_manifest_checksums(tmp_path):
    write_dataset(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "outputs": {"problems.parquet": {"sha256": "wrong"}},
    }))
    with pytest.raises(ValueError, match="checksum"):
        select_problems(tmp_path)
