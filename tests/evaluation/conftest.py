import json

import pytest

from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.evaluation.proof_dataset import ProofCase
from proofstack.evaluation.proof_grader_runner import ProofGraderRunner


class FakeClient:
    def __init__(self, model, replies, usage):
        self.model = model
        self.replies = iter(replies)
        self.usage = usage
        self.messages = []
        self.config = None

    def run_queries(self, queries, no_tqdm=False):
        self.messages.extend(queries)
        reply = next(self.replies)
        yield 0, [{"role": "assistant", "content": reply}], self.usage


@pytest.fixture
def proof_case():
    return ProofCase("synthetic", "a", "Prove a synthetic statement.", "A candidate proof.",
                     ["A synthetic reference."], [], "available")


@pytest.fixture
def make_runner(tmp_path):
    def create(replies, *, samples=5, budget=1, cost=0.01, scheme_usable=True,
               usage=None, model_options=None):
        usage = usage if usage is not None else {
            "cost": cost, "input_tokens": 10, "output_tokens": 10, "reasoning_tokens": 0,
        }
        scheme_reply = json.dumps({"usable": scheme_usable,
                                   "marking_scheme": "A synthetic seven-point rubric."})
        clients = {
            "offline-scheme": FakeClient("offline-scheme", [scheme_reply], usage),
            "offline-grader": FakeClient("offline-grader", replies, usage),
        }

        def client_factory(config):
            client = clients[config["model"]]
            client.config = config
            return client

        components = {
            agent: {"model": {"model": model, **(model_options or {})},
                    "system_prompt": "Synthetic evaluator instructions."}
            for agent, model in (("MarkingSchemeAgent", "offline-scheme"),
                                 ("ProofGrader", "offline-grader"))
        }
        context = RunContext.create(
            root_workdir=tmp_path / "run", flat=True,
            run_budget=BudgetSpec(max_usd=budget, max_wallclock_s=60),
            component_configs=components, api_client_factory=client_factory,
        )
        return (ProofGraderRunner(context, samples=samples),
                clients["offline-scheme"], clients["offline-grader"])

    return create
