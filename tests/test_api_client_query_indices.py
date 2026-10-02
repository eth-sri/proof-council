"""Custom query indices must survive all result and cancellation paths."""
import pytest

from mathagents import APIClient


@pytest.mark.parametrize("backend", ["standard", "batch", "vllm"])
def test_custom_indices_are_not_used_as_query_list_offsets(backend):
    client = object.__new__(APIClient)
    client.api = "vllm" if backend == "vllm" else "openai"
    client.batch_processing = backend == "batch"
    client.concurrent_requests = 2
    client._validate_and_prepare_query = lambda q: q
    client._get_cost = lambda *args: 0
    # A stopped provider returns None. Custom indices here are deliberately
    # outside the current batch's positional range.
    client._run_query_with_retry = lambda *args: None
    client._openai_batch_processing = lambda queries, indices: [None] * len(queries)
    client._run_vllm_queries = lambda queries: (
        (i, q + [{"role": "assistant", "content": ""}], {"cost": 0})
        for i, q in enumerate(queries)
    )
    queries = [[{"role": "user", "content": f"passage-{i}"}] for i in (4, 5)]
    results = list(client.run_queries(queries, no_tqdm=True, custom_indices=[4, 5]))
    assert sorted((idx, convo[0]["content"]) for idx, convo, _ in results) == [
        (4, "passage-4"), (5, "passage-5")]

