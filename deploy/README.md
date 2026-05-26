# ProofStack Workflow Docker Image

This image packages the current config-first ProofStack runner.

The container entrypoint is:

```bash
python scripts/run_workflow.py
```

Run arguments are the same as the local runner.

## Example

```bash
docker build -f deploy/Dockerfile -t proofstack-workflow .
docker run --rm \
  -v $(pwd)/problems/example.txt:/data/input/problem.txt:ro \
  -v $(pwd)/local-output:/data/output \
  -e OPENAI_API_KEY \
  proofstack-workflow \
  --workflow nimble_proof \
  --problem /data/input/problem.txt \
  --output /data/output \
  --run-id demo
```

Equivalent local command:

```bash
uv run python scripts/run_workflow.py \
  --workflow nimble_proof \
  --problem problems/example.txt \
  --output local-output \
  --run-id demo
```

## Outputs

`scripts/run_workflow.py` writes a run directory under the chosen
`--output` root, for example `/data/output/demo/`. That directory
contains `events.jsonl`, `run-metadata.json`, `resume_cache/`, and
per-agent work directories.

## Code Pointers

- `scripts/run_workflow.py` is the supported CLI entrypoint.
- `configs/workflows/*.yaml` define workflow presets.
- `src/proofstack/` contains the workflow runtime.
- `src/mathagents/api_client.py` handles provider calls, retries, and
  cost accounting.
