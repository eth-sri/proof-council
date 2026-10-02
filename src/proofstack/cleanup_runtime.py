"""Unpaid CLI compatibility checks shared by image builds and cleanup admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


# Fable 5.1 rejects older Claude Code releases at the provider boundary.
MIN_CLAUDE_VERSION = (2, 1, 251)
CLAUDE_FLAGS = (
    "--print", "--setting-sources", "--disable-slash-commands", "--settings",
    "--output-format", "--verbose", "--effort",
    "--max-budget-usd", "--strict-mcp-config", "--mcp-config", "--tools",
    "--allowedTools", "--disallowedTools", "--agents", "--session-id", "--resume",
)
CODEX_FLAGS = (
    "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--json",
    "--sandbox", "--output-last-message",
)


def check_cleanup_runtime(*, claude_version=None, codex_version=None, require_codex=True):
    """Inspect version/help with no credentials, user settings or model calls."""
    with tempfile.TemporaryDirectory(prefix="cleanup-preflight-") as home:
        env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL") if name in os.environ}
        env.update(HOME=home, CLAUDE_CONFIG_DIR=str(Path(home) / ".claude"),
                   CODEX_HOME=str(Path(home) / ".codex"), DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")

        def probe(command):
            result = subprocess.run(command, env=env, cwd=home, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=20, check=True)
            return result.stdout + result.stderr

        def check(binary, help_args, flags, expected=None, minimum=None):
            version_text = probe([binary, "--version"])
            match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version_text)
            if not match:
                raise RuntimeError(f"cannot determine {binary} version")
            version = match.group(0)
            if expected and version != expected:
                raise RuntimeError(f"expected {binary} {expected}, found {version}")
            if minimum and tuple(map(int, match.groups())) < minimum:
                raise RuntimeError(f"cleanup requires {binary} >= {'.'.join(map(str, minimum))}")
            help_text = probe([binary, *help_args])
            missing = [flag for flag in flags if not re.search(re.escape(flag) + r"(?=[\s=,]|$)", help_text)]
            if missing:
                raise RuntimeError(f"{binary} is missing required cleanup options: {', '.join(missing)}")
            return {"version": version, "checked_flags": list(flags)}

        result = {"claude": check("claude", ["--help"], CLAUDE_FLAGS,
                                  claude_version, MIN_CLAUDE_VERSION)}
        if require_codex:
            result["codex"] = check("codex", ["exec", "--help"], CODEX_FLAGS, codex_version)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-version")
    parser.add_argument("--codex-version")
    args = parser.parse_args()
    print(json.dumps(check_cleanup_runtime(**vars(args)), indent=2))
