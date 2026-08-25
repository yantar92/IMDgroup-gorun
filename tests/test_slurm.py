"""Tests for IMDgroup.gorun.slurm."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

import IMDgroup.gorun.slurm as slurm_mod

from IMDgroup.gorun.slurm import (
    _executable_find,
    clear_slurm_logs,
    get_best_script,
    sbatch_estimate_start,
    sbatch_script,
)


# --- sbatch_script ----------------------------------------------------------


def test_sbatch_script_empty_args() -> None:
    """With no args, the script carries only the fixed signal line."""
    assert sbatch_script("#!/usr/bin/env bash", {}, "echo hi") == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        "\n"
        "echo hi\n"
    )


def test_sbatch_script_single_arg() -> None:
    """A single sbatch arg is emitted with its key and quoted value."""
    assert sbatch_script(
        "#!/usr/bin/env bash", {"partition": "cpu"}, "echo hi"
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        "echo hi\n"
    )


def test_sbatch_script_multiple_args_preserve_order() -> None:
    """Args are emitted in dict insertion order, before the script."""
    assert sbatch_script(
        "#!/usr/bin/env bash",
        {"partition": "cpu", "time": "01:00:00"},
        "run.sh",
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        '#SBATCH --time="01:00:00"\n'
        "run.sh\n"
    )


# --- _executable_find -------------------------------------------------------


def test_executable_find_present(monkeypatch) -> None:
    """A command found by shutil.which reports as present."""
    monkeypatch.setattr(slurm_mod.shutil, "which", lambda cmd: "/usr/bin/x")
    assert _executable_find("x") is True


def test_executable_find_absent(monkeypatch) -> None:
    """A command not found by shutil.which reports as absent."""
    monkeypatch.setattr(slurm_mod.shutil, "which", lambda cmd: None)
    assert _executable_find("x") is False


# --- clear_slurm_logs -------------------------------------------------------


def test_clear_slurm_logs_removes_slurm_and_extra(tmp_path: Path) -> None:
    """slurm-*.out and listed extra logs are removed; unrelated files stay."""
    (tmp_path / "slurm-1.out").touch()
    (tmp_path / "slurm-2.out").touch()
    (tmp_path / "vasp.out").touch()
    (tmp_path / "keep.log").touch()

    clear_slurm_logs(str(tmp_path), extra_logs=["vasp.out"])

    assert not (tmp_path / "slurm-1.out").exists()
    assert not (tmp_path / "slurm-2.out").exists()
    assert not (tmp_path / "vasp.out").exists()
    assert (tmp_path / "keep.log").exists()


def test_clear_slurm_logs_no_extra(tmp_path: Path) -> None:
    """Without extra_logs only slurm-*.out files are removed."""
    (tmp_path / "slurm-1.out").touch()
    (tmp_path / "other.out").touch()

    clear_slurm_logs(str(tmp_path))

    assert not (tmp_path / "slurm-1.out").exists()
    assert (tmp_path / "other.out").exists()


# --- sbatch_estimate_start --------------------------------------------------


def test_sbatch_estimate_start_success(monkeypatch, tmp_path: Path) -> None:
    """A successful --test-only output yields (wait_time, ncpus)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            return (
                b"sbatch: Job 123 to start at 2026-08-25T14:00:00 "
                b"using 8 processors on nodes n001 in partition cpu"
            )
        if cmd.startswith("date"):
            return b"2026-08-25T13:00:00"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    wait, cpus = sbatch_estimate_start("#!/bin/bash\necho hi\n")
    assert cpus == 8
    assert wait == timedelta(hours=1)


def test_sbatch_estimate_start_bad_account(monkeypatch, tmp_path: Path) -> None:
    """An invalid account error returns None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output=b"Invalid account or account/partition combination specified"
        )

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    assert sbatch_estimate_start("x") is None


def test_sbatch_estimate_start_inside_node(monkeypatch, tmp_path: Path) -> None:
    """An inside-node access error yields a zero-wait, single-cpu estimate."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            raise subprocess.CalledProcessError(
                1, cmd, output=b"allocation failure: Access/permission denied"
            )
        if cmd.startswith("date"):
            return b"2026-08-25T13:00:00"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    wait, cpus = sbatch_estimate_start("x")
    assert cpus == 1
    assert wait == timedelta(0)


def test_sbatch_estimate_start_unavailable_node_config(monkeypatch, tmp_path: Path) -> None:
    """An unavailable node-config error returns None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output=b"Requested node configuration is not available"
        )

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    assert sbatch_estimate_start("x") is None


# --- get_best_script --------------------------------------------------------


def test_get_best_script_picks_earliest_finish(monkeypatch) -> None:
    """The script whose estimated finish time is earliest wins."""
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    monkeypatch.setattr(
        slurm_mod,
        "sbatch_estimate_start",
        lambda s: {
            sbatch_script(shebang, alt_args[0], script): (timedelta(hours=1), 4),
            sbatch_script(shebang, alt_args[1], script): (timedelta(0), 8),
        }[s],
    )
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[1], script)


def test_get_best_script_all_none_raises(monkeypatch) -> None:
    """When every queue is unavailable, OSError is raised."""
    monkeypatch.setattr(slurm_mod, "sbatch_estimate_start", lambda s: None)
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    with pytest.raises(OSError):
        get_best_script(alt_args, "echo hi", "#!/usr/bin/env bash")
