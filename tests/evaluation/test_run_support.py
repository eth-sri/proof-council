"""Shared process cleanup and accounting behavior."""

import json
import signal
import subprocess

import pytest

from proofstack.evaluation import run_support
from proofstack.evaluation.run_support import run_process, usage_report


def test_sigterm_handler_is_restored_after_interruption():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with run_support.interrupt_on_sigterm():
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_subprocess_launch_failure_is_reported(tmp_path):
    result = run_process(["/missing/synthetic-command"], tmp_path / "process.log",
                         cwd=tmp_path, timeout_seconds=1)
    assert result["returncode"] is None
    assert "Cannot launch" in result["error"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), subprocess.TimeoutExpired("fixture", 1)])
def test_process_group_is_stopped_on_timeout_or_interruption(tmp_path, monkeypatch, interruption):
    signals = []

    class Child:
        pid = 1234
        returncode = -signal.SIGKILL
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise interruption
            if self.waits == 2:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return self.returncode

    child = Child()

    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(run_support.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    if isinstance(interruption, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_process(["synthetic"], tmp_path / "process.log", cwd=tmp_path, timeout_seconds=1)
    else:
        result = run_process(["synthetic"], tmp_path / "process.log", cwd=tmp_path, timeout_seconds=1)
        assert result["timed_out"] is True
    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    assert child.waits == 3


def test_missing_or_damaged_usage_is_not_reported_complete(tmp_path):
    assert usage_report(tmp_path, completed=True)["cost_usd"] is None
    (tmp_path / "events.jsonl").write_text("damaged\n")
    result = usage_report(tmp_path, completed=True)
    assert result["accounting_complete"] is False
    assert result["damaged_usage_lines"] == [1]


def test_unavailable_provider_usage_is_retained(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps({
        "kind": "model.call", "payload": {"cost_usd": 0.2, "usage_unavailable": True}}))
    result = usage_report(tmp_path, completed=True)
    assert result["cost_usd"] == 0.2
    assert result["usage_unavailable"] is True
    assert result["accounting_complete"] is False


def test_truncated_utf8_keeps_recorded_usage(tmp_path):
    valid_event = json.dumps({"kind": "model.call", "payload": {"cost_usd": 0.25, "in_tokens": 12}})
    (tmp_path / "events.jsonl").write_bytes(valid_event.encode() + b'\n{"partial": "\xe2')
    result = usage_report(tmp_path, completed=False)
    assert result["cost_usd"] == 0.25 and result["tokens"] == 12
    assert result["damaged_usage_lines"] == [2]
    assert result["accounting_complete"] is False
