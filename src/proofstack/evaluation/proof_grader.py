"""ProofGrader-style marking schemes and independent 0–7 grading calls."""

import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError

from proofstack.kinds.api_call import APICallAgent


class ReferenceContext(BaseModel):
    problem: str = Field(min_length=1)
    reference_solutions: list[str] = Field(min_length=1)
    reference_notes: list[str] = Field(default_factory=list)


class MarkingScheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    usable: StrictBool
    marking_scheme: str = Field(min_length=1)


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: StrictInt = Field(ge=0, le=7)
    rationale: str = Field(min_length=1)
    errors: list[str]


class SchemeResponse(BaseModel):
    scheme: MarkingScheme | None = None
    raw_response: str
    error: str | None = None


class GradeResponse(BaseModel):
    grade: Grade | None = None
    raw_response: str
    error: str | None = None


def parse_json_response(text: str, schema: type[BaseModel]) -> BaseModel:
    body = text.strip()
    if body.startswith("```json\n") and body.endswith("```"):
        body = body[len("```json\n"):-3].strip()
    result = schema.model_validate(json.loads(body), strict=True)
    for field in ("marking_scheme", "rationale"):
        value = getattr(result, field, None)
        if value is not None and not value.strip():
            raise ValueError(f"{field} must not be whitespace-only")
    return result


class MarkingSchemeAgent(APICallAgent):
    Inputs = ReferenceContext
    Outputs = SchemeResponse
    SYSTEM_PROMPT: ClassVar[str] = ""

    def render_messages(self, inp):
        return [
            {"role": "developer", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": inp.model_dump_json()},
        ]

    def extra_client_kwargs(self):
        return {"tools": [], "max_tool_calls": 0}

    def parse_output(self, raw_text, inp):
        try:
            scheme = parse_json_response(raw_text, MarkingScheme)
            return SchemeResponse(scheme=scheme, raw_response=raw_text)
        except (ValueError, ValidationError) as error:
            return SchemeResponse(raw_response=raw_text, error=str(error))

    def cache_output_is_reusable(self, out):
        return out.scheme is not None and out.scheme.usable


class ProofGrader(APICallAgent):
    class Inputs(ReferenceContext):
        marking_scheme: str = Field(min_length=1)
        solution: str = Field(min_length=1)

    Outputs = GradeResponse
    SYSTEM_PROMPT: ClassVar[str] = ""
    # Identical prompts must still produce separate ensemble samples.
    cache_enabled: ClassVar[bool] = False

    def render_messages(self, inp):
        return [
            {"role": "developer", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": inp.model_dump_json()},
        ]

    def extra_client_kwargs(self):
        return {"tools": [], "max_tool_calls": 0}

    def parse_output(self, raw_text, inp):
        try:
            grade = parse_json_response(raw_text, Grade)
            return GradeResponse(grade=grade, raw_response=raw_text)
        except (ValueError, ValidationError) as error:
            return GradeResponse(raw_response=raw_text, error=str(error))
