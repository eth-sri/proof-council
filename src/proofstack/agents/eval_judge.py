"""EvalJudge — score an agent output against a free-text success criterion.

Used by the agent regression-eval framework (SPEC §13). The judge sees:
  - the qualname of the agent under test (context only),
  - the inputs that agent saw (so the criterion can reference them),
  - the success_criteria string from the test case YAML,
  - the agent's actual output (Pydantic-dumped JSON).

It returns a structured verdict (``pass`` / ``fail`` / ``inconclusive``)
with a short rationale and a confidence score.

The judge is itself a ``proofstack.Agent`` so it gets event-logged,
cost-tracked, and dev-UI-introspectable like any other agent.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.context import ModelSpec
from proofstack.kinds.api_call import APICallAgent


JUDGE_SYSTEM_PROMPT = """\
You are an evaluator of AI agent outputs. You will receive:

  - the name of the agent being tested (for context only),
  - the inputs the agent received,
  - a SUCCESS CRITERIA string written by a developer,
  - the agent's actual OUTPUT.

Your task is to decide whether the OUTPUT satisfies the SUCCESS
CRITERIA. You are NOT being asked whether the underlying mathematics
is correct in general — only whether the output meets the criterion
the developer recorded.

Decision rules:

  - "pass":         output clearly satisfies all stated criteria.
  - "fail":         output clearly violates at least one stated criterion,
                    OR the output is empty/malformed in a way the
                    criteria say should not happen.
  - "inconclusive": the criteria are ambiguous, the output is
                    partially-relevant in a way the criteria do not
                    cover, or you genuinely cannot tell. This is the
                    correct verdict whenever you would otherwise be
                    guessing — better to flag for human review than to
                    silently green-flag a regression.

Output format — emit EXACTLY one JSON object inside <verdict>...</verdict>
tags, with these fields:

  {
    "verdict": "pass" | "fail" | "inconclusive",
    "confidence": 0.0..1.0,
    "rationale": "1-3 sentences explaining your decision",
    "flagged_aspects": ["specific issue 1", "specific issue 2"]
  }

Do not include any prose outside the tags.
"""


Verdict = Literal["pass", "fail", "inconclusive"]


class EvalJudge(APICallAgent):
    """Judge an agent's output against a free-text success criterion."""

    description: ClassVar[str] = (
        "Score an agent output against a developer-written success criterion."
    )
    execution_mode: ClassVar[str] = "agent"

    SYSTEM_PROMPT: ClassVar[str] = JUDGE_SYSTEM_PROMPT
    USER_PROMPT: ClassVar[str] = (
        "Agent under test: {agent_under_test}\n\n"
        "Inputs the agent received:\n```json\n{case_inputs_json}\n```\n\n"
        "SUCCESS CRITERIA:\n{success_criteria}\n\n"
        "Agent OUTPUT:\n```json\n{agent_output_json}\n```\n\n"
        "Emit your verdict in the required <verdict>...</verdict> tags."
    )
    # Cheap default; per-case override available via ``judge_model:`` in YAML.
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-54-mini"

    class Inputs(BaseModel):
        agent_under_test: str = Field(
            description="Qualname of the agent the output came from (context only)."
        )
        case_inputs: dict[str, Any] = Field(
            default_factory=dict,
            description="The inputs the agent saw, JSON-serializable.",
        )
        success_criteria: str = Field(
            description="Developer-written description of what counts as success."
        )
        agent_output: dict[str, Any] = Field(
            default_factory=dict,
            description="The agent's Outputs model, JSON-serialized.",
        )

    class Outputs(BaseModel):
        verdict: Verdict = "inconclusive"
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)
        rationale: str = ""
        flagged_aspects: list[str] = Field(default_factory=list)

    def render_messages(self, inp):
        msgs = []
        if self.SYSTEM_PROMPT:
            msgs.append({"role": "developer", "content": self.SYSTEM_PROMPT})
        msgs.append(
            {
                "role": "user",
                "content": self.USER_PROMPT.format(
                    agent_under_test=inp.agent_under_test,
                    case_inputs_json=json.dumps(inp.case_inputs, indent=2, default=str),
                    success_criteria=inp.success_criteria.strip(),
                    agent_output_json=json.dumps(inp.agent_output, indent=2, default=str),
                ),
            }
        )
        return msgs

    def parse_output(self, raw_text: str, inp):
        m = re.search(r"<verdict>(.*?)</verdict>", raw_text, re.DOTALL)
        body = m.group(1).strip() if m else raw_text.strip()
        body = re.sub(r"^```(?:json)?", "", body).strip()
        body = re.sub(r"```$", "", body).strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self.Outputs(
                verdict="inconclusive",
                rationale="judge produced unparseable output",
                flagged_aspects=[f"raw_response_excerpt: {raw_text[:200]}"],
            )
        if not isinstance(data, dict):
            return self.Outputs(
                verdict="inconclusive",
                rationale="judge produced non-object output",
            )
        verdict = data.get("verdict", "inconclusive")
        if verdict not in ("pass", "fail", "inconclusive"):
            verdict = "inconclusive"
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        flagged = data.get("flagged_aspects") or []
        if not isinstance(flagged, list):
            flagged = [str(flagged)]
        return self.Outputs(
            verdict=verdict,
            confidence=confidence,
            rationale=str(data.get("rationale", "")),
            flagged_aspects=[str(x) for x in flagged],
        )


__all__ = ["EvalJudge", "JUDGE_SYSTEM_PROMPT"]
