"""Lossless cleanup packets, including migration from pre-compaction checkpoints."""
import asyncio
import json

import pytest

from proofstack.agents.batch3_critic import Batch3CleanupCritic, CleanupContextTooLarge
from proofstack.context import RunContext
from proofstack.kinds.api_call import APICallAgent


@pytest.fixture
def critic(tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("preflight must not contact a provider")

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=no_network)
    return Batch3CleanupCritic(ctx)


@pytest.mark.parametrize("preset", [None, "firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_cleanup_critic_explicitly_requests_pro_max_and_logs_it(critic, monkeypatch, preset):
    from proofstack.registry import load_preset

    if preset:
        critic.ctx.component_configs = load_preset(preset).component_configs

    async def reply(self, inp):
        return self.parse_output("<cleanup_verdict>accept</cleanup_verdict>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    asyncio.run(critic(problem="p", answer_tex="proof"))
    events = [json.loads(line) for line in (critic.ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    check = next(e["payload"] for e in events if e["kind"] == "cleanup.context.preflight")
    assert check["requested_model"] == "gpt-6-astra"
    assert check["requested_reasoning_mode"] == "pro"
    assert check["requested_reasoning_effort"] == "max"


def test_critic_model_metadata_honors_explicit_override(critic):
    critic.ctx.model_overrides["Batch3CleanupCritic"] = "models/openai/gpt-6-astra-max"
    check = critic.context_preflight(critic.Inputs(problem="p", answer_tex="proof"))
    assert check["requested_reasoning_mode"] is None
    assert check["requested_reasoning_effort"] == "max"


@pytest.mark.parametrize("verdict", ["accept", "repair", "restore", "research"])
def test_cleanup_review_uses_its_own_cache_contract(critic, monkeypatch, verdict):
    calls = []

    async def reply(self, inp):
        calls.append(inp)
        return self.parse_output(f"<cleanup_verdict>{verdict}</cleanup_verdict>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    async def exercise():
        first = await critic(problem="p", answer_tex="proof")
        second = await Batch3CleanupCritic(critic.ctx)(problem="p", answer_tex="proof")
        assert first == second

    asyncio.run(exercise())
    assert len(calls) == 1


def test_large_legacy_history_keeps_all_findings_but_only_one_pair_of_manuscripts(critic):
    baseline = "BASELINE_START\n" + "b" * 220_000 + "\nBASELINE_END"
    candidate = "CANDIDATE_START\n" + "c" * 268_595 + "\nCANDIDATE_END"
    history = []
    for n in range(7):
        history.extend([
            {"role": "user", "content": f"OLD_DRAFT_{n}" + "x" * 400_000},
            {"role": "assistant", "content": f"Unresolved finding {n}\n" + "r" * 25_000},
        ])
    inp = critic.Inputs(problem="Original problem", baseline_tex=baseline, answer_tex=candidate,
                        prior_messages=history, editor_response="Repaired finding 6; please check it.")
    messages = critic.render_messages(inp)
    assert messages[:-1] == history[1::2]
    text = json.dumps(messages)
    assert "OLD_DRAFT_" not in text
    assert text.count("BASELINE_START") == text.count("CANDIDATE_START") == 1
    assert baseline in messages[-1]["content"] and candidate in messages[-1]["content"]
    assert inp.editor_response in messages[-1]["content"]
    report = critic.context_preflight(inp)
    assert report["input_token_upper_bound"] < report["input_token_budget"]
    assert report["prior_review_count"] == 7 and report["removed_user_packets"] == 7


def test_multiple_repairs_and_resumes_keep_earliest_and_latest_objections(critic):
    history = []
    for n in range(5):
        inp = critic.Inputs(problem="Problem", answer_tex=f"draft-{n}", prior_messages=history,
                            baseline_tex="original baseline" if n == 0 else "")
        # JSON roundtrip models saving/loading a checkpoint into a new instance.
        history = json.loads(critic.parse_output(
            f"Objection {n}\n<cleanup_verdict>repair</cleanup_verdict>", inp,
        ).model_dump_json())["messages_after"]
    assert sum(m["role"] == "user" for m in history) == 1
    assert len([m for m in history if m["role"] == "assistant"]) == 5
    for n in range(5):
        assert any(f"Objection {n}" in m["content"] for m in history)
    current = history[-2]["content"]
    assert "original baseline" in current and "draft-4" in current
    assert all(f"draft-{n}" not in current for n in range(4))


def test_explicit_baseline_wins_over_old_episode(critic):
    old = critic.parse_output("Old finding", critic.Inputs(
        problem="p", baseline_tex="old-baseline", answer_tex="old-candidate"))
    inp = critic.Inputs(problem="p", baseline_tex="new-baseline", answer_tex="new-candidate",
                        prior_messages=old.messages_after)
    text = critic.render_messages(inp)[-1]["content"]
    assert "old-baseline" not in text and "old-candidate" not in text
    assert "new-baseline" in text and "new-candidate" in text


@pytest.mark.parametrize("oversize", ["manuscript", "findings", "unicode"])
def test_oversized_essential_packet_stops_before_provider_without_truncation(critic, oversize):
    candidate = "Proof"
    history = []
    if oversize == "findings":
        history = [{"role": "assistant", "content": "Unresolved " + "x" * 1_000_000}]
    else:
        candidate = ("x" * 1_000_000 if oversize == "manuscript" else "\N{GREEK SMALL LETTER ALPHA}" * 500_000)
    inp = critic.Inputs(problem="p", baseline_tex="baseline", answer_tex=candidate, prior_messages=history)
    assert candidate in critic.render_messages(inp)[-1]["content"]
    with pytest.raises(CleanupContextTooLarge, match="do not retry"):
        asyncio.run(critic(**inp.model_dump()))
    events = [json.loads(line) for line in (critic.ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    check = next(e for e in events if e["kind"] == "cleanup.context.preflight")
    assert not check["payload"]["fits"]
    assert not any(e["kind"] == "model.call.start" for e in events)


def test_configured_output_and_tool_capacity_are_reserved(critic):
    critic.ctx.component_configs["Batch3CleanupCritic"] = {
        "context_window_tokens": 300_000, "context_tool_reserve_tokens": 32_000,
    }
    critic.ctx.model_overrides["Batch3CleanupCritic"] = {
        "base": "models/openai/gpt-6-astra-pro", "max_tokens": 64_000,
    }
    preflight = critic.context_preflight(critic.Inputs(problem="p", answer_tex="proof"))
    assert preflight["input_token_budget"] == 204_000


def test_provider_context_error_after_preflight_is_operator_action_not_research(critic, monkeypatch):
    async def fail(self, inp):
        raise ValueError("OpenAI response.status=failed: context_length_exceeded")

    monkeypatch.setattr(APICallAgent, "run", fail)
    with pytest.raises(CleanupContextTooLarge, match="Provider rejected"):
        asyncio.run(critic(problem="p", answer_tex="proof", baseline_tex="baseline"))


@pytest.mark.parametrize("window", [None, "invalid", 0, -1])
def test_invalid_preflight_configuration_does_not_trigger_paid_recovery(critic, window):
    critic.ctx.component_configs["Batch3CleanupCritic"] = {"context_window_tokens": window}
    with pytest.raises(CleanupContextTooLarge, match="Invalid cleanup critic context limits"):
        asyncio.run(critic(problem="p", answer_tex="proof", baseline_tex="baseline"))
