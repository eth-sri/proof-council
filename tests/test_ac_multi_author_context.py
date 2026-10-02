from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.registry import load_preset


def _lead(tmp_path, *, delegation=None, model=None):
    config = {"delegation": {"enabled": True, **(delegation or {})}}
    if model is not None:
        config["model"] = model
    ctx = RunContext.create(
        run_id="multi-context", root_workdir=tmp_path, flat=True,
        run_budget=BudgetSpec(max_wallclock_s=80000),
        component_configs={"Author": config},
    )
    lead = MultiAuthor(ctx, name="Author")
    lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=2), "")
    return lead


def test_followup_includes_current_wave_briefing(tmp_path):
    seat = SubAuthorSeat(RunContext.create(root_workdir=tmp_path, flat=True))
    first = SubAuthorSeat.Inputs(
        role="prover", task="Prove L", problem="P",
        briefing="Assume characteristic zero.",
    )
    prior = seat.parse_output("Here is the first proof.", first).messages_after
    followup = seat.render_messages(SubAuthorSeat.Inputs(
        role="prover", task="Check the proof with the revised assumptions.", problem="P",
        briefing="Correction: characteristic two.", prior_messages=prior,
    ))
    assert followup[:-1] == prior
    assert "Correction: characteristic two." in followup[-1]["content"]
    assert "Check the proof" in followup[-1]["content"]


@pytest.mark.parametrize("role", ["explorer", "prover", "checker", "drafter"])
@pytest.mark.parametrize("continued", [False, True])
def test_helper_sandbox_safeguard_is_in_every_invocation(tmp_path, role, continued):
    seat = SubAuthorSeat(RunContext.create(root_workdir=tmp_path, flat=True))
    prior = [
        {"role": "user", "content": "An earlier task without the sandbox warning."},
        {"role": "assistant", "content": "Published proof at wave1/prover1/proof.md."},
    ] if continued else []
    messages = seat.render_messages(SubAuthorSeat.Inputs(
        role=role, task="Check the lemma.", problem="P", prior_messages=prior,
        started_at_utc="2026-09-19T12:00:00Z", deadline_utc="2026-09-19T12:50:00Z",
        remaining_seconds=3000,
    ))
    prompt = messages[-1]["content"]
    assert "read_context or publish_artifact can replace your sandbox" in prompt
    assert "designated checkpoints are automatically" in prompt
    assert "/mnt/data/checkpoints BEFORE" in prompt
    assert "persists only the UTF-8 content you supply" in " ".join(prompt.split())
    assert "Confirm publication succeeded" in prompt
    assert "copy listed\nread-only attachment_path files back to their sandbox_path" in prompt
    assert "missing, not verified" in prompt
    assert "hard wave deadline 2026-09-19T12:50:00Z" in prompt
    if continued:
        assert messages[:-1] == prior
        assert "Restore useful work from that report or published artifacts" in prompt


@pytest.mark.parametrize("location", ["lead", "subagent", "role"])
def test_inline_model_spec_reaches_factory(tmp_path, location):
    spec = {"base": "models/openai/gpt-6-astra-max", "background": False}
    delegation = {}
    model = None
    if location == "lead":
        model = spec
    elif location == "subagent":
        delegation["subagent_model"] = spec
    else:
        delegation["role_models"] = {"prover": spec}
    captured = []

    async def run_seat(self, inp):
        await self._get_client()
        return self.parse_output("Proved L.", inp)

    async def scenario():
        lead = _lead(tmp_path, delegation=delegation, model=model)
        lead.ctx.api_client_factory = lambda cfg: captured.append(cfg) or SimpleNamespace(model=cfg["model"])
        with patch.object(SubAuthorSeat, "run", run_seat):
            result = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "Prove L"}])
        assert "Proved L." in result
        assert "error:" not in result

    asyncio.run(scenario())
    assert len(captured) == 1
    assert captured[0]["background"] is False
    assert captured[0]["model"] == "gpt-6-astra--max"
    assert captured[0]["max_hosted_tool_calls"] is None


@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
@pytest.mark.parametrize("role", ["explorer", "prover", "checker"])
def test_batch3_helpers_request_astra_pro(tmp_path, workflow, role):
    component = load_preset(workflow).component_configs["Author"]
    captured = []

    async def run_seat(self, inp):
        await self._get_client()
        return self.parse_output("Completed the delegated task.", inp)

    async def scenario():
        lead = _lead(tmp_path, delegation=component["delegation"], model=component["model"])
        lead.ctx.api_client_factory = lambda cfg: captured.append(cfg) or SimpleNamespace(model=cfg["model"])
        with patch.object(SubAuthorSeat, "run", run_seat):
            result = await asyncio.to_thread(lead._delegate, [{"role": role, "task": "Investigate L"}])
        assert "Completed the delegated task." in result
        assert "error:" not in result

    asyncio.run(scenario())
    assert len(captured) == 1
    assert captured[0]["model"] == "gpt-6-astra--max"
    assert captured[0]["reasoning"]["mode"] == "pro"
    assert captured[0]["openai_pro_fallback_after_failures"] == 2


def test_short_lead_call_refuses_wave_despite_long_workflow_budget(tmp_path):
    async def scenario():
        lead = _lead(tmp_path)
        with patch.object(lead, "_run_wave") as run_wave:
            result = lead._delegate(
                [{"role": "prover", "task": "Prove L"}],
                call_deadline_monotonic_s=time.monotonic() + 600,
            )
        assert "Not enough wallclock" in result
        assert lead._waves_done == 0
        run_wave.assert_not_called()

    asyncio.run(scenario())


def test_wave_is_clamped_to_lead_time_less_synthesis_reserve(tmp_path):
    deadlines = []

    async def run_wave(wave, tasks, briefing, deadline_s, container_id):
        deadlines.append(deadline_s)
        return "reports"

    async def scenario():
        lead = _lead(tmp_path, delegation={"synthesis_reserve_s": 100})
        with patch.object(lead, "_run_wave", run_wave):
            result = await asyncio.to_thread(
                lead._delegate, [{"role": "prover", "task": "Prove L"}],
                call_deadline_monotonic_s=time.monotonic() + 1000,
            )
        assert result == "reports"
        assert 899 < deadlines[0] <= 900

    asyncio.run(scenario())


def test_non_responses_seats_do_not_receive_unsupported_hosted_limit(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    seat = SubAuthorSeat(ctx, model_ref={"api": "anthropic", "model": "claude-test"})
    assert "max_hosted_tool_calls" not in seat.extra_client_kwargs()
