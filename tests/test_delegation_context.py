import asyncio
import io
import json
import stat
import zipfile
from datetime import datetime

import pytest

from proofstack.agents.ac.author import Author
from proofstack.agents.ac import delegation_context
from proofstack.agents.ac.delegation_context import DelegationContext
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext


def read_all(read, path):
    pieces, offset, revision = [], 0, None
    while True:
        reply = json.loads(read(path, offset=offset, max_chars=1000, revision=revision))
        revision = reply.get("revision")
        pieces.append(reply["content"])
        offset = reply["next_offset"]
        if offset is None:
            return "".join(pieces)


def lead_at(tmp_path, **config):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            run_budget=BudgetSpec(max_wallclock_s=10000),
                            component_configs={"Author": {"delegation": {"enabled": True, **config}}})
    lead = MultiAuthor(ctx, name="Author")
    lead._begin_turn(Author.Inputs(problem="P", round=3, n_rounds=5, answer_tex="DRAFT",
                                  prev_critique="GAP", prev_council="COUNCIL",
                                  prev_compute_response="COMPUTATION", workflow_feedback="FEEDBACK"))
    return lead


def test_files_are_immutable_chunked_and_restricted(tmp_path):
    store = DelegationContext(tmp_path, round=1)
    text = "A detailed proof. " * 4000
    store.put("round/proof.md", text, source="Author")
    view = store.view(set(), publisher="helpers/wave1-prover1")
    assert "error" in json.loads(view.read("round/proof.md"))
    assert read_all(store.view().read, "round/proof.md") == text
    published = json.loads(view.publish("lemma.tex", "Proof."))
    assert read_all(view.read, published["path"]) == "Proof."
    assert "immutable" in json.loads(view.publish("lemma.tex", "Different"))["error"]
    assert "reserved" in json.loads(view.publish("final-response.md", "spoof"))["error"]
    view.close()
    assert "closed" in json.loads(view.publish("late.md", "Late"))["error"]
    assert not (tmp_path / "helpers/wave1-prover1/late.md").exists()
    assert "error" in json.loads(view.read("/etc/passwd"))


@pytest.mark.parametrize("name", ["../outside.md", "/tmp/out.md", "x/../out.md", "x/.ssh/key.txt",
                                      "x\\out.md", "auth.json", "double//x.md", "binary.zip"])
def test_unsafe_artifact_names_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        DelegationContext(tmp_path, round=1).put(name, "bad", source="helper")


def test_compute_archive_import_is_text_only_bounded_and_secret_filtered(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "private-credential-12345")
    archive = tmp_path / "compute.zip"
    link = zipfile.ZipInfo("link.txt")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("code/check.py", "assert 2 + 2 == 4")
        zf.writestr("../escape.md", "no")
        zf.writestr("auth.json", "no")
        zf.writestr("token.txt", "private-credential-12345")
        zf.writestr("binary.txt", b"\xff")
        zf.writestr(link, "/etc/passwd")
    store = DelegationContext(tmp_path / "bundle", round=1)
    store.add_compute(archive)
    assert set(store.files) == {"compute/code/check.py"}
    assert store.omissions
    assert not (tmp_path / "bundle/escape.md").exists()
    monkeypatch.setattr("proofstack.agents.ac.delegation_context.MAX_TOTAL_BYTES", 25)
    with pytest.raises(ValueError, match="capacity"):
        store.put("large.txt", "x" * 25, source="test")


@pytest.mark.parametrize("size", [10, 11, 29])
def test_compute_scan_budget_never_transfers_partial_members(tmp_path, monkeypatch, size):
    monkeypatch.setattr(delegation_context, "MAX_TOTAL_BYTES", 120)
    archive = tmp_path / "compute.zip"
    content = "x" * size
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("invalid.txt", b"\xff" * 110)
        zf.writestr("proof.txt", content)
        zf.writestr("tail.txt", "ok")
    store = DelegationContext(tmp_path / "bundle", round=1)
    store.add_compute(archive)
    if size == 10:
        assert store.files["compute/proof.txt"] == content
    else:
        assert "compute/proof.txt" not in store.files
        assert not (store.root / "compute/proof.txt").exists()
        assert store.files["compute/tail.txt"] == "ok"
        assert any("scan byte limit" in reason for reason in store.omissions)


def test_compute_short_read_is_not_published(tmp_path, monkeypatch):
    archive = tmp_path / "compute.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("proof.txt", "complete proof")
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *args, **kwargs: io.BytesIO(b"incomplete"))
    store = DelegationContext(tmp_path / "bundle", round=1)
    store.add_compute(archive)
    assert not store.files
    assert store.omissions


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_compute_quota_preserves_capacity_for_helper_handoffs(tmp_path, monkeypatch, limit):
    if limit == "files":
        monkeypatch.setattr(delegation_context, "MAX_FILES", 8)
    else:
        monkeypatch.setattr(delegation_context, "MAX_TOTAL_BYTES", 160)
    archive = tmp_path / "compute.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(8):
            zf.writestr(f"certificate-{i}.txt", "v" * 20)
    store = DelegationContext(tmp_path / "bundle", round=1)
    store.put("round/problem.txt", "P", source="Author")
    store.add_compute(archive)
    store.add_compute(archive)
    assert len([p for p in store.files if p.startswith("compute/")]) == 3
    with pytest.raises(ValueError, match="Compute context capacity"):
        store.put("compute/extra.txt", "v" * 20, source="Compute")
    first = store.view(set(store.files), publisher="helpers/wave1-prover1")
    assert json.loads(first.publish("check.py", "assert True"))["status"] == "published"
    report = first.retain_report("Verified the lemma.")
    second = store.view(set(first.published), publisher="helpers/wave2-checker1")
    assert read_all(second.read, report) == "Verified the lemma."
    assert second.retain_report("Checked the proof.") in store.files
    assert store.omissions


@pytest.mark.parametrize("large_compute_handoff", [False, True])
def test_two_waves_deliver_full_reports_artifacts_and_blind_dependencies(tmp_path, monkeypatch, large_compute_handoff):
    views = []
    full = ("## Findings\n" + "Long exact proof. " * 2000).strip()

    async def helper(self, inp):
        views.append(self.context_view)
        manifest = json.loads(read_all(self.context_view.read, "manifest.json"))
        paths = {f["path"] for f in manifest["files"]}
        if inp.role == "prover":
            assert {"round/critic.md", "round/council.md", "round/compute_response.md"} <= paths
            assert json.loads(self.context_view.publish("check.py", "assert True"))["status"] == "published"
            return self.parse_output(full, inp)
        assert "round/critic.md" not in paths and inp.answer_tex == ""
        report = "helpers/wave1-prover1/final-response.md"
        assert read_all(self.context_view.read, report) == full
        assert read_all(self.context_view.read, "helpers/wave1-prover1/check.py") == "assert True"
        return self.parse_output("Checked the entire proof", inp)

    async def scenario():
        lead = lead_at(tmp_path, max_report_chars=100)
        if large_compute_handoff:
            archive = tmp_path / "compute.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for i in range(512):
                    zf.writestr(f"certificate-{i}.txt", "valid evidence")
            lead._begin_turn(lead._current_inp.model_copy(update={"compute_zip_path": archive}))
        first = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove L"}])
        assert "read_context" in first and "final-response.md" in first
        assert read_all(lead._read_context, "helpers/wave1-prover1/final-response.md") == full
        second = await asyncio.to_thread(lead._delegate, [{"role": "checker", "task": "Check prover1",
                                                         "include_workspace": False, "depends_on": ["prover1"]}])
        assert "Checked the entire proof" in second
        assert all(v.closed for v in views)

    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())


def test_missing_dependency_does_not_launch_or_consume_wave(tmp_path, monkeypatch):
    async def scenario():
        lead = lead_at(tmp_path)
        result = lead._delegate([{"role": "checker", "task": "Check L", "depends_on": ["prover1"]}])
        assert "Error:" in result and lead._waves_done == 0
    asyncio.run(scenario())


def test_blind_tasks_cannot_read_same_wave_publications(tmp_path, monkeypatch):
    published = asyncio.Event()

    async def helper(self, inp):
        if inp.role == "prover":
            self.context_view.publish("secret.md", "same-wave proof")
            published.set()
            return self.parse_output("Proof", inp)
        await published.wait()
        assert "error" in json.loads(self.context_view.read("helpers/wave1-prover1/secret.md"))
        return self.parse_output("Blind check", inp)

    async def scenario():
        lead = lead_at(tmp_path)
        result = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove"},
            {"role": "checker", "task": "Independent attempt", "include_workspace": False}])
        assert "Blind check" in result and "error:" not in result
    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())


def test_untransferred_report_is_not_an_available_dependency(tmp_path, monkeypatch):
    async def helper(self, inp):
        return self.parse_output("x" * 2_000_001, inp)

    async def scenario():
        lead = lead_at(tmp_path)
        first = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove"}])
        assert "Report transfer failed" in first
        assert "read_context(path='helpers/wave1-prover1/final-response.md')" not in first
        second = lead._delegate([{"role": "checker", "task": "Check", "depends_on": ["prover1"]}])
        assert "not transferred" in second and lead._waves_done == 1
    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())


def test_shared_followup_cannot_claim_to_be_blind(tmp_path, monkeypatch):
    async def helper(self, inp):
        return self.parse_output("Done", inp)

    async def scenario():
        lead = lead_at(tmp_path)
        await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove"}])
        result = lead._delegate([{"role": "prover", "task": "Forget", "agent_id": "prover1", "include_workspace": False}])
        assert "cannot forget" in result and lead._waves_done == 1
    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())


def test_cached_lead_reader_uses_new_turn_without_old_artifacts(tmp_path):
    lead = lead_at(tmp_path)
    reader, _ = lead._context_tools()[0]
    lead._context.put("helpers/old.md", "old", source="old turn")
    lead._begin_turn(Author.Inputs(problem="NEW", round=4, n_rounds=5))
    assert read_all(reader, "round/problem.txt") == "NEW"
    assert "error" in json.loads(reader("helpers/old.md"))


def test_budget_boundary_keeps_helper_report_and_blocks_more_delegation(tmp_path, monkeypatch):
    async def helper(self, inp):
        exc = BudgetExhausted("run", "usd", 1, 2)
        exc.completed_output = self.parse_output("Useful completed proof", inp)
        raise exc

    async def scenario():
        lead = lead_at(tmp_path)
        result = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove"}])
        assert "Useful completed proof" in result
        assert "Useful completed proof" in read_all(lead._read_context, "helpers/wave1-prover1/final-response.md")
        assert "budget is exhausted" in lead._delegate([{"role": "prover", "task": "More"}])

    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())


def test_queued_seats_get_actual_start_and_remaining_wave_time(tmp_path, monkeypatch):
    inputs = []
    async def helper(self, inp):
        inputs.append(inp)
        await asyncio.sleep(.03)
        return self.parse_output("Done", inp)

    async def scenario():
        lead = lead_at(tmp_path, max_threads=1)
        tasks = [{"role": "prover", "task": "Prove", "agent_id": None, "include_workspace": True}] * 2
        await lead._run_wave(1, tasks, "", 10, None)
    monkeypatch.setattr(SubAuthorSeat, "run", helper)
    asyncio.run(scenario())
    assert inputs[1].remaining_seconds < inputs[0].remaining_seconds <= 10
    assert inputs[0].deadline_utc == inputs[1].deadline_utc
    assert datetime.fromisoformat(inputs[1].started_at_utc) > datetime.fromisoformat(inputs[0].started_at_utc)


@pytest.mark.parametrize("cap", [None, 0, 7])
def test_optional_hosted_tool_cap_does_not_limit_context_tools(tmp_path, cap):
    lead = lead_at(tmp_path, seat_max_tool_calls=cap)
    seat = SubAuthorSeat(lead.ctx)
    seat.MAX_TOOL_CALLS = lead._delegation_cfg()["seat_max_tool_calls"]
    cfg = seat.extra_client_kwargs()
    assert cfg["max_hosted_tool_calls"] == cap
    assert cfg["max_tool_calls"]["read_context"] > 0


@pytest.mark.parametrize("cap", [-1, True, "20", 2.5])
def test_invalid_hosted_tool_cap(tmp_path, cap):
    with pytest.raises(ValueError, match="seat_max_tool_calls"):
        lead_at(tmp_path, seat_max_tool_calls=cap)
