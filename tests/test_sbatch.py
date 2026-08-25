"""Tests for IMDgroup.gorun.sbatch."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import IMDgroup.gorun.sbatch as sbatch_mod

from IMDgroup.gorun.sbatch import (
    current_server,
    get_default_job_name,
    get_sbatch_args,
    get_user_sbatch_args,
)


def _args(nodes=None, time=None) -> SimpleNamespace:
    """Build a minimal script-args namespace."""
    return SimpleNamespace(number_of_nodes=nodes, time_limit=time)


# --- get_user_sbatch_args ---------------------------------------------------


def test_get_user_sbatch_args_empty() -> None:
    """No explicit nodes/time yields an empty dict."""
    assert get_user_sbatch_args(_args()) == {}


def test_get_user_sbatch_args_nodes_only() -> None:
    """An explicit node count maps to a string ``nodes`` key."""
    assert get_user_sbatch_args(_args(nodes=2)) == {"nodes": "2"}


def test_get_user_sbatch_args_time_only() -> None:
    """An explicit time limit maps to a string ``time`` key."""
    assert get_user_sbatch_args(_args(time="02:00:00")) == {"time": "02:00:00"}


def test_get_user_sbatch_args_both() -> None:
    """Both values are mapped to their sbatch keys."""
    assert get_user_sbatch_args(_args(nodes=2, time="02:00:00")) == {
        "nodes": "2",
        "time": "02:00:00",
    }


# --- get_default_job_name ---------------------------------------------------


def test_get_default_job_name_falls_back_to_directory(monkeypatch, tmp_path: Path) -> None:
    """Without an INCAR the job name is the directory basename."""
    monkeypatch.chdir(tmp_path)
    assert get_default_job_name() == {"job-name": os.path.basename(str(tmp_path))}


def test_get_default_job_name_reads_system_from_incar(monkeypatch, tmp_path: Path) -> None:
    """A real INCAR supplies the SYSTEM value as the job name."""
    (tmp_path / "INCAR").write_text("SYSTEM = mytest\n")
    monkeypatch.chdir(tmp_path)
    assert get_default_job_name() == {"job-name": "mytest"}


# --- get_sbatch_args --------------------------------------------------------


def test_get_sbatch_args_precedence(monkeypatch, fake_config) -> None:
    """Later sources override earlier ones in the merge chain."""
    monkeypatch.setattr(sbatch_mod, "get_default_job_name", lambda: {"job-name": "myname"})
    script_args = _args(nodes=3, time="02:00:00")
    result = get_sbatch_args(
        script_args, fake_config, "testserver", "cpu", adapter_name="vasp"
    )
    assert result == {
        "account": "default",   # server_defaults
        "nodes": "3",           # user args override adapter defaults
        "time": "02:00:00",     # user args override queue
        "job-name": "myname",   # job name
        "partition": "cpu",     # always the queue
    }


def test_get_sbatch_args_no_adapter_defaults(monkeypatch, fake_config) -> None:
    """Without an adapter name, only server defaults and queue apply."""
    monkeypatch.setattr(sbatch_mod, "get_default_job_name", lambda: {"job-name": "myname"})
    result = get_sbatch_args(
        _args(), fake_config, "testserver", "cpu", adapter_name=None
    )
    assert result == {
        "account": "default",
        "time": "01:00:00",
        "job-name": "myname",
        "partition": "cpu",
    }


# --- current_server ---------------------------------------------------------


def test_current_server_uses_cluster_name_override(monkeypatch, fake_config) -> None:
    """CLUSTER_NAME maps through the cluster names table to a server."""
    monkeypatch.setenv("CLUSTER_NAME", "testhost")
    assert current_server(fake_config) == "testserver"


def test_current_server_unknown_host_returns_uname(monkeypatch, fake_config) -> None:
    """A host absent from the names table is returned unchanged."""
    monkeypatch.setenv("CLUSTER_NAME", "unknownhost")
    assert current_server(fake_config) == "unknownhost"


def test_current_server_uname_fallback(monkeypatch, fake_config) -> None:
    """Without CLUSTER_NAME, uname -n is consulted and matched."""
    monkeypatch.delenv("CLUSTER_NAME", raising=False)
    monkeypatch.setattr(sbatch_mod, "barf_if_no_cmd", lambda cmd: None)
    monkeypatch.setattr(
        sbatch_mod.subprocess,
        "check_output",
        lambda cmd, **kwargs: b"testhost\n",
    )
    assert current_server(fake_config) == "testserver"
