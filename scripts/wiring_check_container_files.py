"""Smoke-test the upload --> edit --> download roundtrip for an OpenAI
code_interpreter container.

If this works end-to-end, we can let the AC Author edit the canonical
workspace files in place inside its sandbox rather than re-emitting
their full contents in every assistant turn.

The test:

  1. Write a small text file locally.
  2. Upload it to the OpenAI files endpoint with purpose=user_data.
  3. Issue a single Responses API call against gpt-5.4-mini (cheap)
     with code_interpreter and the uploaded file_id attached via
     ``container.file_ids``.
  4. Ask the model to read /mnt/data/<file>, edit it (append a
     marker line), save it back, and report the final contents.
  5. Pull the container_id out of the response output, list the
     container's files, locate the edited file by path, and download
     its current content.
  6. Diff against the upload to confirm the edit landed.

Uses gpt-5.4-mini to keep the test cheap (~$0.001 expected). If you
want to validate Pro behaves the same, swap MODEL below.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402

load_dotenv_file(REPO_ROOT / ".env")

from openai import OpenAI  # noqa: E402


MODEL = "gpt-5.4-mini"  # use "gpt-5.5-pro" + reasoning to validate Pro
INITIAL_CONTENTS = "round 0 contents\nline two\n"
EDIT_INSTRUCTION = (
    "There is exactly one read-only .txt file at /mnt/data/ (with a "
    "platform-id prefix in its basename). Your task:\n"
    "1. Read its current contents.\n"
    "2. Write the *edited* contents to a NEW writable file at "
    "   /mnt/data/source.txt (no prefix — exactly that path). The new "
    "   contents should be: the original two lines unchanged, followed "
    "   by a new third line `EDITED_BY_MODEL`.\n"
    "3. Verify by `print(open('/mnt/data/source.txt').read())`.\n"
    "4. In your response prose, tell me in one sentence that you saved "
    "   the edited file; do NOT paste the file contents inline."
)


def _wait_for_completion(client: OpenAI, response, timeout_s: int = 600):
    start = time.time()
    while response.status in {"queued", "in_progress"}:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Response {response.id} stuck in {response.status}")
        time.sleep(10)
        response = client.responses.retrieve(response.id)
    return response


def main() -> int:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 1. local file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(INITIAL_CONTENTS)
        local_path = Path(f.name)
    print(f"local file: {local_path}")
    print(f"  contents: {INITIAL_CONTENTS!r}")

    # 2. upload
    with open(local_path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="user_data")
    print(f"uploaded: file_id={uploaded.id} bytes={uploaded.bytes}")

    # 3. responses call with code_interpreter + file attached
    print(f"\ncalling {MODEL} with code_interpreter + file_ids=[{uploaded.id}]...")
    start = time.monotonic()
    response = client.responses.create(
        model=MODEL,
        tools=[
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": [uploaded.id]},
            }
        ],
        input=[{"role": "user", "content": EDIT_INSTRUCTION}],
        background=True,
    )
    response = _wait_for_completion(client, response, timeout_s=900)
    elapsed = time.monotonic() - start
    print(f"  completed in {elapsed:.1f}s; status={response.status}")
    if response.status != "completed":
        print(f"  error: {getattr(response, 'error', None)}")
        print(f"  incomplete_details: {getattr(response, 'incomplete_details', None)}")

    # Surface anything noteworthy in response.output
    print(f"  response.output: {len(response.output)} item(s)")
    container_id = None
    final_text = ""
    for out in response.output:
        kind = out.type
        if kind == "message":
            for c in out.content:
                if getattr(c, "type", None) == "output_text":
                    final_text += c.text + "\n"
            print(f"    message: {len(final_text)} chars")
        elif kind == "code_interpreter_call":
            container_id = out.container_id
            code_head = (out.code or "").strip().splitlines()[:2]
            print(f"    code_interpreter_call: container_id={container_id} code_head={code_head}")
        else:
            print(f"    {kind}")

    if not container_id:
        print("ERROR: no container_id surfaced in response.output")
        return 1
    print(f"\nmodel's final assistant text:\n----\n{final_text.strip()}\n----")

    # 4. list + download from container
    print(f"\nlisting files in container {container_id}...")
    container_files = list(client.containers.files.list(container_id))
    for cf in container_files:
        print(f"  container file id={cf.id} path={getattr(cf, 'path', '?')} bytes={getattr(cf, 'bytes', '?')}")

    # Look for the freshly-written /mnt/data/source.txt first (no
    # platform-id prefix); fall back to any .txt as before.
    target = next(
        (cf for cf in container_files if str(getattr(cf, "path", "")) == "/mnt/data/source.txt"),
        None,
    )
    if target is None:
        target = next(
            (cf for cf in container_files if str(getattr(cf, "path", "")).endswith(".txt")),
            None,
        )
    if target is None:
        print("ERROR: could not locate the edited file in the container")
        return 1
    print(f"  selected target: id={target.id} path={getattr(target,'path','?')} bytes={getattr(target,'bytes','?')}")

    print(f"\ndownloading container file id={target.id} path={getattr(target,'path','?')}")
    response_obj = client.containers.files.content.retrieve(target.id, container_id=container_id)
    # ``content.retrieve`` returns an httpx-like response; read the raw bytes.
    if hasattr(response_obj, "read"):
        body = response_obj.read()
    elif hasattr(response_obj, "content"):
        body = response_obj.content
    else:
        body = bytes(response_obj)
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError:
            body = body.decode("utf-8", errors="replace")
    print(f"  downloaded contents:\n----\n{body}\n----")

    # 5. verdict
    if "EDITED_BY_MODEL" in body and "round 0 contents" in body:
        print("\nPASS: roundtrip works end-to-end.")
        return 0
    print("\nFAIL: download does not contain both the original line and the appended marker.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
