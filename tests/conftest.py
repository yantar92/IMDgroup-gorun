"""Shared fixtures and helpers for gorun tests."""

from __future__ import annotations

import gzip
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def fake_cluster(monkeypatch, tmp_path):
    """Build a fake cluster boundary for end-to-end integration tests.

    Creates a ``fakebin/`` directory of executable shims (=sbatch=,
    =squeue=, =uname=, =date=, =rsync=) and a fake =VASP_PATH= tree,
    prepends =fakebin= to =PATH=, and sets the environment the gorun
    entry points expect.  Every fake binary appends its invocation to
    the log file exposed as ``ns.log``.

    This is used by the =integration= tests in =test_integration.py= to
    run the real gorun pipeline against a fake cluster without a Slurm
    scheduler, VASP binary, or pseudopotential library.
    """
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    vasp_path = tmp_path / "fakevasp"
    (vasp_path / "bin").mkdir(parents=True)
    (vasp_path / "pp").mkdir()
    log = tmp_path / "fake.log"

    def _chmod_x(path: Path) -> None:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def write_bin(name: str, body: str) -> None:
        p = fakebin / name
        p.write_text(body)
        _chmod_x(p)

    write_bin(
        "sbatch",
        '#!/bin/sh\n'
        'echo "sbatch $*" >> "$GORUN_FAKE_LOG"\n'
        'case "$*" in\n'
        '  *--test-only*)\n'
        '    echo "sbatch: Job 123 to start at 2026-08-25T14:00:00 using 8 processors on nodes n001 in partition cpu"\n'
        '    ;;\n'
        '  *)\n'
        '    exit 0\n'
        '    ;;\n'
        'esac\n',
    )
    write_bin(
        "squeue",
        '#!/bin/sh\n'
        'echo "squeue $*" >> "$GORUN_FAKE_LOG"\n'
        'exit 0\n',
    )
    write_bin("uname", '#!/bin/sh\necho "testhost"\n')
    write_bin(
        "date",
        '#!/bin/sh\n'
        'echo "date $*" >> "$GORUN_FAKE_LOG"\n'
        'echo "2026-08-25T13:00:00"\n',
    )
    write_bin(
        "rsync",
        '#!/bin/sh\n'
        'echo "rsync $*" >> "$GORUN_FAKE_LOG"\n'
        'exit 0\n',
    )

    for name in ("vasp_ncl", "vasp_std", "vasp_gam"):
        p = vasp_path / "bin" / name
        p.write_text(
            "#!/bin/sh\n"
            f'echo "vasp {name}" >> "$GORUN_FAKE_LOG"\n'
            'touch vasp_ran\n'
            'exit 0\n'
        )
        _chmod_x(p)

    monkeypatch.setenv(
        "PATH",
        str(fakebin)
        + os.pathsep
        + (os.environ.get("PATH") or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    )
    monkeypatch.setenv("GORUN_FAKE_LOG", str(log))
    monkeypatch.setenv("VASP_PATH", str(vasp_path))
    monkeypatch.setenv("VASP_PP_PATH", str(vasp_path / "pp"))
    monkeypatch.setenv("CLUSTER_NAME", "testhost")

    return SimpleNamespace(fakebin=fakebin, vasp_path=vasp_path, log=log)


@pytest.fixture
def converged_vasp_dir(tmp_path):
    """Decompress a real converged VASP run into a fresh tmp directory.

    Reads =tests/fixtures/vasp_converged/=, gunzips the large output
    files (=OUTCAR.gz=, =vasprun.xml.gz=) and copies the small plain
    files.  Used by the =integration= tests to exercise the gorun
    pipeline against realistic, genuinely converged VASP output
    (produced by =scripts/prepare_vasp_fixtures.sh=).

    Skips when the compressed output files are absent, since they are
    not committed as plain text and must be generated first.
    """
    src = Path(__file__).parent / "fixtures" / "vasp_converged"
    required_gz = ("OUTCAR.gz", "vasprun.xml.gz")
    missing = [name for name in required_gz if not (src / name).is_file()]
    if missing:
        pytest.skip(
            "missing converged VASP fixtures: "
            + ", ".join(missing)
            + ".  Run scripts/prepare_vasp_fixtures.sh <vasp-dir> first."
        )

    for entry in src.iterdir():
        if entry.suffix == ".gz":
            target = tmp_path / entry.stem
            with gzip.open(entry, "rb") as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copy(entry, tmp_path / entry.name)
    return tmp_path
