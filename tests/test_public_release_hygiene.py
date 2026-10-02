"""Release packaging guardrails, using only synthetic local files."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_FILES = (
    "problems/local-input.md",
    "problems/nested/draft.tex",
    "archive/research/answer.tex",
    "lean/Research.lean",
    "PRIVATE_DEVELOPMENT.md",
    "ProofCouncil.pdf",
    "private_problems/input.md",
    "test_run_problems/input.md",
    "local-artifacts/trace.json",
    "output/manuscript.tex",
    "outputs/run/events.jsonl",
    ".env",
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_add_keeps_private_inputs_out(tmp_path):
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    public_files = ("problems/example.txt", "src/example.py")
    for name in (*PRIVATE_FILES, *public_files):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", ".")):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=tmp_path,
    ).decode().strip("\0").split("\0")
    assert set(tracked) == {".gitignore", *public_files}


def test_docker_context_excludes_private_roots():
    rules = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    for path in PRIVATE_FILES:
        root = path.split("/", 1)[0]
        if "/" in path:
            assert root + "/" in rules
        elif path.endswith(".pdf"):
            assert "*.pdf" in rules
        else:
            assert path in rules
