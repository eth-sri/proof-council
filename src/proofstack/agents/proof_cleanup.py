"""File-backed cleanup workflows with non-destructive failure handling."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from proofstack.agents.dag_workflow import DAGWorkflow
from proofstack.budget import BudgetExhausted
from proofstack.latex_contract import DEFAULT_FIRSTPROOF_PAGE_LIMIT


class ProofCleanupWorkflow(DAGWorkflow):
    # The paths can stay the same while their contents change. Only the
    # content-based editor calls, not the file-backed workflow, are reusable.
    cache_enabled: ClassVar[bool] = False

    class Inputs(DAGWorkflow.Inputs):
        # AC restart flags and round counters are not editor task inputs.
        model_config = ConfigDict(extra="ignore")

        answer_tex: str = ""
        references_bib: str = ""
        answer_tex_path: str | Path = ""
        references_bib_path: str | Path = ""
        page_limit: int = DEFAULT_FIRSTPROOF_PAGE_LIMIT

    async def _last_gasp(self, inp, state, error):
        await self.events.emit(
            "workflow.last_gasp", {"type": type(error).__name__, "msg": str(error)}
        )
        # The generic DAG fallback writes solutions/<problem_id>.tex, which
        # may already contain the accepted Author/Critic proof.
        outputs = self._build_outputs(self.dag.get("outputs") or {}, state)
        outputs.update(
            status="budget_exhausted" if isinstance(error, BudgetExhausted) else "error",
            error=f"{type(error).__name__}: {error}",
            last_gasp=True,
        )
        if "final_critic_answer_ready" in outputs:
            outputs["final_critic_answer_ready"] = False
        return self.Outputs(**outputs)


class AuthorCriticCleanupWorkflow(ProofCleanupWorkflow):
    class Inputs(DAGWorkflow.Inputs):
        # Declared explicitly so --restart-from enables AC checkpoint recovery.
        # Leave other AC defaults to the referenced author_critic preset.
        resume_run: bool = False
