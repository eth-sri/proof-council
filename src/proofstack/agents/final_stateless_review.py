"""An ordinary fresh critic, isolated from research-review receipts and caches."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field

from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.dag_workflow import _deep_merge
from proofstack.context import component_config_for_class


class FinalStatelessCritic(ACCritic):
    cache_enabled = False

    def __init__(self, ctx, **kwargs):
        defaults = component_config_for_class(ctx.component_configs, ACCritic, "ACCritic")
        configured = _deep_merge({"FinalStatelessCritic": defaults}, ctx.component_configs)
        overrides = dict(ctx.model_overrides)
        if "ACCritic" in overrides:
            overrides.setdefault("FinalStatelessCritic", overrides["ACCritic"])
        super().__init__(replace(ctx, component_configs=configured, model_overrides=overrides), **kwargs)

    def _recovery_checkpoint(self, inp):
        path = super()._recovery_checkpoint(inp)
        return self.ctx.root_workdir / "final_stateless_review" / "critic_receipts" / path.name


class FinalReviewBlocked(RuntimeError):
    requires_reconciliation = True


class FinalReviewState(BaseModel):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_input: ACCritic.Inputs
    status: Literal["pending", "completed"] = "pending"
    review: ACCritic.Outputs | None = None
    max_usd: float = Field(gt=0, allow_inf_nan=False)
    initial_spend: float = Field(ge=0, allow_inf_nan=False)
    deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)
