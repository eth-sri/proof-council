"""Independent judge for attn_exp_* runs.

Feeds (ground truth + candidate best_tex) to a fresh `claude -p --model haiku`
and asks for a fixed-schema JSON verdict. Same prompt for every variant so
scores are comparable.

Usage:
  uv run python scripts/judge_attn.py exp-baseline exp-specialists ...
If no run ids are given, judges every outputs/exp-* run found.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

GROUND_TRUTH = r"""
GROUND TRUTH (answer key) for M_e(g) = number of isomorphism classes of stable
graphs of genus g with 0 legs and exactly e edges, for fixed e as a function of g:

1. STRUCTURE RESULT: For each fixed e, M_e(g) is EVENTUALLY (for g large enough,
   depending on e) a QUASI-POLYNOMIAL in g.

2. e = 3 SPECIFICS: M_3(2) = 2, and for g >= 3, M_3(g) is a degree-3
   quasi-polynomial in g with PERIOD 6, with leading terms
   (1/9) g^3 + (7/8) g^2 + ... . The six residue classes mod 6 give six affine
   correction terms (exact constants below):
     g = 0 mod 6:  (1/9)g^3 + (7/8)g^2 + (5/12)g - 2
     g = 1 mod 6:  (1/9)g^3 + (7/8)g^2 + (1/6)g - 155/72
     g = 2 mod 6:  (1/9)g^3 + (7/8)g^2 + (5/12)g - 20/9
     g = 3 mod 6:  (1/9)g^3 + (7/8)g^2 + (1/6)g - 19/8
     g = 4 mod 6:  (1/9)g^3 + (7/8)g^2 + (5/12)g - 16/9
     g = 5 mod 6:  (1/9)g^3 + (7/8)g^2 + (1/6)g - 187/72

3. A correct answer ALSO gives complete descriptions for e = 0, 1, 2.
""".strip()

JUDGE_TEMPLATE = """You are an impartial mathematics grader. Below is an ANSWER KEY and a
CANDIDATE solution. Grade the CANDIDATE strictly against the ANSWER KEY. Judge
ONLY the candidate's final text; do not give credit for things it does not
state. Be skeptical of hand-waving.

==================== ANSWER KEY ====================
{ground_truth}

==================== CANDIDATE SOLUTION ====================
{candidate}
==================== END CANDIDATE ====================

Return your verdict as a SINGLE JSON object on one line, no markdown, with EXACTLY
these keys:
- "structure_correct": boolean - does the candidate correctly identify M_e(g) as
  eventually quasi-polynomial in g for fixed e?
- "m3_correct": boolean - does it get M_3(2)=2 AND the genus-3 quasi-polynomial
  right, at least the leading term (1/9)g^3 (or equivalent) and period 6?
- "covers_e012": boolean - does it give descriptions for e = 0, 1, AND 2?
- "rigor": integer 0-10 - is the argument actually rigorous, or hand-wavy /
  contradictory? 0 = no real argument, 10 = fully rigorous and correct.
- "justification": string - one line explaining the scores.

Output ONLY the JSON object."""


def candidate_tex(run_id: str) -> str | None:
    meta = OUT / run_id / "run-metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            tex = (data.get("outputs") or {}).get("best_tex")
            if tex:
                return tex
        except Exception:
            pass
    # fallback: newest answer.tex in workspaces
    cands = sorted(
        (OUT / run_id).glob("ac_workspaces/**/answer.tex"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0].read_text()
    return None


def judge(run_id: str) -> dict:
    tex = candidate_tex(run_id)
    if not tex:
        return {"run_id": run_id, "error": "no best_tex found"}
    prompt = JUDGE_TEMPLATE.format(ground_truth=GROUND_TRUTH, candidate=tex)
    proc = subprocess.run(
        ["claude", "-p", "--model", "haiku"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )
    raw = proc.stdout.strip()
    verdict: dict
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        verdict = json.loads(raw[start:end])
    except Exception:
        verdict = {"parse_error": True, "raw": raw[:2000], "stderr": proc.stderr[:500]}
    verdict["run_id"] = run_id
    verdict["tex_chars"] = len(tex)
    return verdict


def main() -> None:
    run_ids = sys.argv[1:]
    if not run_ids:
        run_ids = sorted(p.name for p in OUT.glob("exp-*") if p.is_dir())
    results = []
    for rid in run_ids:
        v = judge(rid)
        results.append(v)
        print(json.dumps(v))
    (OUT / "attn_judge_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
