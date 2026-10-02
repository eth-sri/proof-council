import json
import pytest

from proofstack.agents import linear_read as mod

TEX = r"""\documentclass{article}
\newcommand{\gap}{\operatorname{gap}}
\begin{document}
\section{Introduction}
We study the mediant.

Short.

\begin{lemma}
The first line.

The second line, still inside the environment.
""" + "Padding so the lemma passes the merge threshold. " * 8 + r"""
\end{lemma}

""" + "A long final paragraph. " * 20 + r"""
\end{document}
"""


def test_split_keeps_environments_together_and_merges_short_blocks():
    preamble, blocks = mod.split_document(TEX)
    assert preamble.startswith(r"\documentclass") and r"\gap" in preamble
    assert not any(r"\end{document}" in b for b in blocks)
    lemma = next(b for b in blocks if r"\begin{lemma}" in b)
    assert "The second line" in lemma and r"\end{lemma}" in lemma
    # Blocks under 300 characters are merged into the following one, cumulatively.
    assert blocks[0].startswith(r"\section{Introduction}") and "Short." in blocks[0] and blocks[0] == lemma
    assert len(blocks) == 2


def test_parse_reply_is_lenient_about_surrounding_prose():
    parsed = mod.parse_reply('Here you go:\n{"undefined": [{"item": "$c^\\\\pm$", "quote": "q", "why": "w"}]}')
    assert parsed["undefined"][0]["item"] == "$c^\\pm$" and parsed["unclear"] == []
    assert "parse_error" in mod.parse_reply("no json here")


def test_linear_read_queries_each_prefix_and_totals_cost(monkeypatch):
    seen = []

    class FakeClient:
        def __init__(self, **cfg):
            self.cfg = cfg
            self.model = cfg["model"]

        def run_queries(self, queries, no_tqdm, custom_indices):
            for i, query in zip(custom_indices, queries):
                seen.append(query[1]["content"])
                reply = json.dumps({"undefined": [{"item": "mediant", "quote": "the mediant", "why": "never defined"}]
                                    if i == 0 else []})
                yield i, [*query, {"role": "assistant", "type": "cot", "content": "hmm"},
                          {"role": "assistant", "content": reply}], {"cost": 0.25}

    import mathagents
    monkeypatch.setattr(mathagents, "APIClient", FakeClient)
    monkeypatch.setattr(mathagents, "load_solver_config", lambda ref: {"model": "fake", "__source": ref, "read_cost": 1})

    result = mod.linear_read(TEX, "models/fake")
    assert result["passages"] == 2 and result["cost_usd"] == 0.5 and result["model"] == "fake"
    assert "(nothing yet: this is the first passage)" in seen[0]
    assert "TEXT READ SO FAR:\n" + result["blocks"][0] in seen[1]
    assert [f["item"] for f in result["findings"]] == ["mediant"]
    assert "1 undefined-item findings" in result["report"] and "**mediant**" in result["report"]


def test_linear_read_checks_admission_between_bounded_batches():
    batches, charges = [], []
    blocks = [f"Passage {i}. " + "x" * 400 for i in range(9)]

    class Client:
        model = "fake"
        terminated = False

        def run_queries(self, queries, no_tqdm, custom_indices):
            batches.append(len(queries))
            for i, query in zip(custom_indices, queries):
                yield i, [*query, {"role": "assistant", "content": "{}"}], {"cost": 0.25}

    def admit():
        if charges:
            raise RuntimeError("allowance reached")

    import pytest
    with pytest.raises(RuntimeError, match="allowance reached"):
        mod.linear_read("\n\n".join(blocks), "fake", client=Client(),
                        before_batch=admit, on_result=charges.append)
    assert batches == [4] and len(charges) == 4


def test_linear_read_does_not_call_provider_after_cancellation():
    from types import SimpleNamespace
    import pytest

    with pytest.raises(RuntimeError, match="interrupted"):
        mod.linear_read(TEX, "fake", client=SimpleNamespace(model="fake", terminated=True))


@pytest.mark.parametrize("passages", [5, 8, 9])
def test_multibatch_receipts_and_out_of_order_results_are_not_double_counted(passages):
    from mathagents.provider_trace import COUNTS, ProviderTrace

    trace = ProviderTrace()
    charges, indices = [], []

    class Client:
        model = "fake"
        terminated = False

        def run_queries(self, queries, no_tqdm, custom_indices):
            # APIClient returns cumulative ProviderTrace totals for each query
            # index, including earlier batches if the caller reuses indices.
            for idx, query in reversed(list(zip(custom_indices, queries))):
                indices.append(idx)
                trace.update((len(indices), idx), cost=1.0, usage_unavailable=False,
                             **dict.fromkeys(COUNTS, 10))
                reply = json.dumps({"undefined": [{"item": f"item-{idx}"}]})
                yield idx, [*query, {"role": "assistant", "content": reply}], trace.totals(idx)

    blocks = [f"Passage {i}. " + "x" * 400 for i in range(passages)]
    result = mod.linear_read("\n\n".join(blocks), "fake", client=Client(), on_result=charges.append)
    assert result["passages"] == passages
    assert sorted(indices) == list(range(passages))
    assert result["cost_usd"] == sum(c["cost"] for c in charges) == trace.totals()["cost"] == passages
    for field in COUNTS:
        assert sum(c[field] for c in charges) == trace.totals()[field] == 10 * passages
    assert [(f["passage"], f["item"]) for f in result["findings"]] == [
        (i + 1, f"item-{i}") for i in range(passages)]
