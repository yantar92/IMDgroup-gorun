"""Tests for IMDgroup.gorun.core.pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import IMDgroup.gorun.core.pipeline as pipeline_mod

from IMDgroup.gorun.core.pipeline import (
    _backup_current_dir,
    _last_run_number,
    _next_run_folder,
    _rsync_exclude_args,
    dispatch_run,
)


def _args(**overrides) -> argparse.Namespace:
    """Build a dispatch args Namespace with all attributes present."""
    ns = argparse.Namespace(
        force=False,
        local=False,
        mark=False,
        max_slurm_jobs=0,
        keep=None,
        queue=None,
        number_of_nodes=None,
        time_limit=None,
        config=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _setup_pipeline(monkeypatch, config=None, server="testserver"):
    """Mock every subprocess/config boundary used by dispatch_run."""
    if config is None:
        config = {
            "testserver": {
                "shebang": "#!/usr/bin/env bash",
                "queues": ["cpu"],
                "stub": {},
            }
        }
    monkeypatch.setattr(pipeline_mod, "get_config", lambda args: config)
    monkeypatch.setattr(pipeline_mod, "current_server", lambda config: server)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: False)
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: 0)
    monkeypatch.setattr(
        pipeline_mod, "clear_slurm_logs", lambda path, extra_logs=None: None
    )
    monkeypatch.setattr(
        pipeline_mod, "get_sbatch_args", lambda *a, **k: {"partition": "cpu"}
    )
    monkeypatch.setattr(
        pipeline_mod, "get_best_script", lambda *a, **k: "FULL_SCRIPT"
    )
    return config


# --- _rsync_exclude_args ----------------------------------------------------


def test_rsync_exclude_args_default() -> None:
    """With no adapter excludes, only gorun_* is excluded."""
    assert _rsync_exclude_args([]) == "--exclude 'gorun_*'"


def test_rsync_exclude_args_with_adapter() -> None:
    """Adapter-specific excludes are appended after gorun_*."""
    assert (
        _rsync_exclude_args(["WAVECAR"])
        == "--exclude 'gorun_*' --exclude 'WAVECAR'"
    )


# --- _last_run_number / _next_run_folder ------------------------------------


def test_last_run_number_empty(tmp_path: Path) -> None:
    """No gorun_N folders yields None."""
    assert _last_run_number(tmp_path) is None


def test_last_run_number_max(tmp_path: Path) -> None:
    """The highest numbered folder is returned."""
    (tmp_path / "gorun_1").mkdir()
    (tmp_path / "gorun_3").mkdir()
    assert _last_run_number(tmp_path) == 3


def test_last_run_number_counts_tarball(tmp_path: Path) -> None:
    """A gorun_N.tar.gz also contributes its number."""
    (tmp_path / "gorun_2.tar.gz").touch()
    assert _last_run_number(tmp_path) == 2


def test_next_run_folder_empty(tmp_path: Path) -> None:
    """The first folder is gorun_1."""
    assert _next_run_folder(tmp_path) == "gorun_1"


def test_next_run_folder_after_existing(tmp_path: Path) -> None:
    """The next folder after gorun_3 is gorun_4."""
    (tmp_path / "gorun_3").mkdir()
    assert _next_run_folder(tmp_path) == "gorun_4"


# --- _backup_current_dir ----------------------------------------------------


def test_backup_current_dir(monkeypatch, tmp_path: Path) -> None:
    """Backup compresses the previous dir, drops gorun_ready, and rsyncs."""
    (tmp_path / "gorun_1").mkdir()
    (tmp_path / "gorun_2").mkdir()
    (tmp_path / "gorun_ready").touch()

    calls = []
    monkeypatch.setattr(pipeline_mod, "barf_if_no_cmd", lambda cmd: None)
    monkeypatch.setattr(
        pipeline_mod.subprocess,
        "check_call",
        lambda cmd, shell=True: calls.append(cmd),
    )

    _backup_current_dir(tmp_path, excludes=["WAVECAR"])

    assert len(calls) == 1
    assert "--exclude 'gorun_*' --exclude 'WAVECAR'" in calls[0]
    assert "gorun_3" in calls[0]
    assert not (tmp_path / "gorun_ready").exists()
    assert (tmp_path / "gorun_2.tar.gz").exists()
    assert not (tmp_path / "gorun_2").exists()


# --- dispatch_run guards ----------------------------------------------------


def test_dispatch_run_running_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A RUNNING file exits 0 before any other work."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "RUNNING").touch()
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_queued_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A queued directory exits 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: True)
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_gorun_ready_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A gorun_ready file exits 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "gorun_ready").touch()
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_no_valid_input(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """No valid input exits 1."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    stub_adapter.responses["is_valid_input"] = False
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_converged_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A converged run exits 1 when not forced."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    stub_adapter.responses["is_converged"] = True
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_unknown_server(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """An unknown server exits 1."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch, server=None)
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_no_queues(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A server with no queues configured exits 1."""
    monkeypatch.chdir(tmp_path)
    config = {"testserver": {"shebang": "#!/usr/bin/env bash", "stub": {}}}
    _setup_pipeline(monkeypatch, config=config)
    assert dispatch_run(_args(), stub_adapter) == 1


# --- dispatch_run mark path -------------------------------------------------


def test_dispatch_run_mark_path(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """The --mark path writes sub and creates gorun_ready."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    assert dispatch_run(_args(mark=True), stub_adapter) == 0
    assert (tmp_path / "sub").read_text() == "FULL_SCRIPT"
    assert (tmp_path / "gorun_ready").exists()


def test_dispatch_run_force_bypasses_running(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """--force bypasses the RUNNING guard and reaches the mark path."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "RUNNING").touch()
    assert dispatch_run(_args(force=True, mark=True), stub_adapter) == 0
    assert (tmp_path / "gorun_ready").exists()
