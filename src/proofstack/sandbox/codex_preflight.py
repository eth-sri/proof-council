"""Offline native code-mode handshake, run inside the actual worker sandbox.

This file is also sent verbatim to ``python3 -c`` in container sandboxes.
It uses no credentials, provider requests, or third-party Python packages.
"""
import json
import os
from pathlib import Path
import platform
import select
import shutil
import struct
import subprocess
import time


def find_host():
    cli = shutil.which("codex")
    if not cli:
        raise RuntimeError("codex is not installed")
    binary = Path(cli).resolve()
    candidates = [binary.parent / "codex-code-mode-host"]
    package = binary.parent.parent
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}.get(platform.machine().lower())
    system = platform.system().lower()
    if binary.suffix == ".js":
        # A Rosetta Python and the Node that launches Codex need not have the
        # same architecture. Follow the CLI launcher, not this interpreter.
        node = json.loads(subprocess.check_output(
            ["node", "-p", "JSON.stringify({platform:process.platform,arch:process.arch})"],
            timeout=5, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"},
        ))
        system = node["platform"]
        arch = {"arm64": "aarch64", "x64": "x86_64"}.get(node["arch"])
    if arch and system in {"linux", "darwin"}:
        target = f"{arch}-" + ("unknown-linux-musl" if system == "linux" else "apple-darwin")
        suffix = "arm64" if arch == "aarch64" else "x64"
        platform_package = f"codex-{system}-{suffix}"
        # Match both nested global npm installs and hoisted local dependencies.
        for root in (package / "node_modules/@openai" / platform_package,
                     package.parent / platform_package, package):
            candidates.append(root / "vendor" / target / "bin/codex-code-mode-host")
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError("native codex-code-mode-host was not found; refusing paid Compute")


def probe(host):
    # No inherited API keys or Codex credentials. Limits are inherited from
    # Sandbox.run_command, exactly as by the subsequent model-driven CLI.
    proc = subprocess.Popen([host], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0,
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"})
    deadline = time.monotonic() + 10

    def read_exact(n):
        data = b""
        while len(data) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("code-mode startup probe timed out")
            part = os.read(proc.stdout.fileno(), n - len(data))
            if not part:
                raise RuntimeError("code-mode host closed its stdout during preflight")
            data += part
        return data

    def frame():
        size = struct.unpack("<I", read_exact(4))[0]
        if size > 65536:
            raise RuntimeError("oversized code-mode preflight response")
        value = json.loads(read_exact(size))
        if value.get("error") or "error" in str(value.get("type", "")).lower():
            raise RuntimeError("code-mode preflight returned an error")
        return value

    messages = [
        {"type": "connection/hello", "supportedVersions": [1], "requiredCapabilities": [], "optionalCapabilities": []},
        {"type": "operation/request", "id": 1, "request": {"method": "session/open", "sessionId": "offline-preflight"}},
        {"type": "operation/request", "id": 2, "request": {"method": "session/execute", "sessionId": "offline-preflight", "request": {
            "tool_call_id": "offline-preflight", "enabled_tools": [], "source": "text(1);", "max_output_tokens": 100}}},
    ]
    frames = []
    try:
        for message in messages:
            data = json.dumps(message).encode()
            proc.stdin.write(struct.pack("<I", len(data)) + data)
            proc.stdin.flush()
            frames.append(frame())
        frames.append(frame())
        result = frames[-1].get("result", {}).get("value", {}).get("Result", {})
        if result.get("error_text") or not any(
            item.get("text") == "1" for item in result.get("content_items", [])
        ):
            raise RuntimeError("native code-mode did not execute text(1) successfully")
        return frames
    finally:
        proc.kill()
        proc.communicate(timeout=2)


if __name__ == "__main__":
    probe(find_host())
    print("code-mode preflight passed")
