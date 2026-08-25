"""Shared fixtures and helpers for gorun tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest


class StubAdapter:
    """Minimal :class:`SoftwareAdapter` that records calls.

    Return values come from *responses*, falling back to per-method
    defaults suitable for a happy-path pipeline run.
    """

    name = "stub"

    _DEFAULT_RESPONSES: dict[str, object] = {
        "validate_environment": None,
        "is_valid_input": True,
        "is_converged": False,
        "has_previous_output": False,
        "prepare_inputs": None,
        "setup_commands": "",
        "generate_run_commands": "echo stub",
        "backup_excludes": [],
    }

    def __init__(self, responses: dict[str, object] | None = None) -> None:
        self.responses = {**self._DEFAULT_RESPONSES, **(responses or {})}
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs) -> object:
        self.calls.append((method, args, kwargs))
        return self.responses[method]

    def validate_environment(self) -> None:
        self._record("validate_environment")

    def is_valid_input(self, path) -> bool:
        return self._record("is_valid_input", path)

    def is_converged(self, path) -> bool:
        return self._record("is_converged", path)

    def has_previous_output(self, path) -> bool:
        return self._record("has_previous_output", path)

    def prepare_inputs(self, path, *, keep=None, force=False, **kwargs) -> None:
        self._record("prepare_inputs", path, keep=keep, force=force, **kwargs)

    def setup_commands(self, server_config) -> str:
        return self._record("setup_commands", server_config)

    def generate_run_commands(self, path, server_config=None) -> str:
        return self._record("generate_run_commands", path, server_config)

    def backup_excludes(self) -> list[str]:
        return self._record("backup_excludes")


@pytest.fixture
def fake_config() -> dict:
    """Synthetic server config used across sbatch/pipeline/adapter tests."""
    return {
        "cluster": {"names": {"testserver": ["testhost"]}},
        "testserver": {
            "shebang": "#!/usr/bin/env bash",
            "queues": ["cpu", "gpu"],
            "defaults": {"sbatch": {"account": "default"}},
            "vasp": {"defaults": {"sbatch": {"nodes": "1"}}},
            "cpu": {"sbatch": {"time": "01:00:00"}},
            "gpu": {"sbatch": {"time": "00:30:00", "gres": "gpu:1"}},
        },
    }


@pytest.fixture
def stub_adapter() -> StubAdapter:
    """A recording :class:`StubAdapter` with happy-path defaults."""
    return StubAdapter()


def run_subprocess(commands: dict[str, object]) -> Callable:
    """Build a fake ``subprocess.check_output``.

    *commands* maps a command prefix to the result.  The prefix is
    matched against the start of the invoked command string (or the
    space-joined argv list).  Values may be:

    - a ``str``: returned as UTF-8 bytes (matching ``check_output``);
    - ``bytes``: returned as-is;
    - an ``Exception``: raised to simulate a failed command.

    Commands not present in *commands* raise ``AssertionError``.
    """

    def fake(cmd, *args, **kwargs):
        key = cmd if isinstance(cmd, str) else " ".join(cmd)
        for prefix, result in commands.items():
            if key.startswith(prefix):
                if isinstance(result, BaseException):
                    raise result
                if isinstance(result, str):
                    return result.encode("utf-8")
                return result
        raise AssertionError(f"Unexpected command: {cmd!r}")

    return fake
