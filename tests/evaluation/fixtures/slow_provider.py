"""Offline CLI fixture whose provider waits until cancellation."""

import json
from pathlib import Path
import sys
import threading

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]

from scripts import evaluate_proofs
from proofstack.evaluation import proof_evaluation

repository, candidates, output = map(Path, sys.argv[1:])


class SlowClient:
    model = "offline"

    def __init__(self):
        self.stopped = threading.Event()

    def run_queries(self, queries, no_tqdm=False):
        (output / "provider_started").touch()
        self.stopped.wait(timeout=30)
        reply = json.dumps({"usable": True, "marking_scheme": "Synthetic scheme"})
        yield 0, [{"role": "assistant", "content": reply}], {"cost": 0}

    def terminate(self):
        (output / "provider_terminated").touch()
        self.stopped.set()


def create_output_directory(repository_root, category):
    output.mkdir(parents=True)
    return output


evaluate_proofs.REPOSITORY_ROOT = repository
proof_evaluation.create_run_directory = create_output_directory
proof_evaluation.resolve_model = lambda reference: {"model": "fake"}
create_context = proof_evaluation.RunContext.create
proof_evaluation.RunContext.create = staticmethod(
    lambda **kwargs: create_context(**kwargs, api_client_factory=lambda config: SlowClient()))
raise SystemExit(evaluate_proofs.main([
    "--solutions", str(candidates),
    "--model", "fake", "--budget-usd", "1", "--samples", "1",
]))
