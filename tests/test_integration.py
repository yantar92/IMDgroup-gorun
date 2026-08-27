"""End-to-end integration scenarios for gorun.

These tests run the real =gorun= entry points on sample data against a
fake cluster boundary (=tests/conftest.py::fake_cluster=).  They are
not per-function unit tests; each one exercises a complete workflow
from the README, from a fresh directory through to a generated (or
executed) run script.

Every test is marked =integration= and shells out through the fake
binaries on =PATH=, so they are kept opt-in behind
=``pytest -m integration``=.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from IMDgroup.gorun.gorun import main as gorun_main

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPTS = Path(__file__).parent.parent / "scripts"
CONFIG = str(FIXTURES / "config.toml")


def _copy_vasp_inputs(target: Path) -> None:
    """Copy a minimal VASP input set into *target*."""
    for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        shutil.copy(FIXTURES / name, target / name)


# --- VASP -------------------------------------------------------------------


def test_vasp_mark_pipeline(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README "Job Preparation Pipeline" + --mark end to end."""
    monkeypatch.chdir(tmp_path)
    _copy_vasp_inputs(tmp_path)

    rc = gorun_main(
        ["vasp", "--mark", "--keep", "POTCAR", "--config", CONFIG]
    )

    assert rc == 0
    assert (tmp_path / "gorun_ready").exists()
    sub = (tmp_path / "sub").read_text()
    assert "#SBATCH" in sub
    assert "VASP_COMMAND" in sub
    # The placeholder POTCAR must be preserved by --keep POTCAR.
    assert (tmp_path / "POTCAR").read_text().startswith("placeholder")
    # INCAR was sanitized (no CRLF/BOM/tab lines added by copy).
    assert "\r" not in (tmp_path / "INCAR").read_text()


def test_vasp_local_runs_fake_binary(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README --local: the fake vasp_ncl binary actually runs."""
    monkeypatch.chdir(tmp_path)
    _copy_vasp_inputs(tmp_path)

    rc = gorun_main(
        ["vasp", "--local", "--keep", "POTCAR", "--config", CONFIG]
    )

    assert rc == 0
    assert (tmp_path / "vasp_ran").exists()
    assert "vasp vasp_ncl" in fake_cluster.log.read_text()


def test_vasp_mark_multiple_dirs(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """--dir runs the full mark pipeline once per directory."""
    monkeypatch.chdir(tmp_path)
    for name in ("d1", "d2"):
        d = tmp_path / name
        d.mkdir()
        _copy_vasp_inputs(d)

    rc = gorun_main(
        ["vasp", "--mark", "--keep", "POTCAR", "--config", CONFIG,
         "--dir", "d1", "d2"]
    )

    assert rc == 0
    assert (tmp_path / "d1" / "gorun_ready").exists()
    assert (tmp_path / "d2" / "gorun_ready").exists()
    assert (tmp_path / "d1" / "sub").exists()
    assert (tmp_path / "d2" / "sub").exists()


def test_vasp_already_converged_exits(fake_cluster, converged_vasp_dir, monkeypatch) -> None:
    """README convergence check: a converged directory exits without re-running."""
    monkeypatch.chdir(converged_vasp_dir)

    rc = gorun_main(["vasp", "--config", CONFIG])

    assert rc == 1


def test_vasp_incar_chain_rotation(fake_cluster, converged_vasp_dir, monkeypatch) -> None:
    """README "converged + INCAR.*": the first INCAR.N is rotated into INCAR."""
    monkeypatch.chdir(converged_vasp_dir)

    original_incar = (converged_vasp_dir / "INCAR").read_text()
    # The converged fixture has no POTCAR (VASP license); supply a dummy
    # so --keep POTCAR skips real pseudopotential generation.
    shutil.copy(FIXTURES / "POTCAR", converged_vasp_dir / "POTCAR")
    (converged_vasp_dir / "INCAR.0").write_text("ENCUT = 600\nEDIFF = 1e-06\n")
    (converged_vasp_dir / "INCAR.1").write_text("ENCUT = 700\nEDIFF = 1e-06\n")

    rc = gorun_main(
        ["vasp", "--mark", "--keep", "POTCAR", "--config", CONFIG]
    )

    assert rc == 0
    assert (converged_vasp_dir / "INCAR").read_text() == "ENCUT = 600\nEDIFF = 1e-06\n"
    assert (converged_vasp_dir / "INCAR.old").read_text() == original_incar
    assert (converged_vasp_dir / "INCAR.1").exists()


# --- MACE -------------------------------------------------------------------


def test_mace_mark_with_split(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README mace-finetune example: split data into train/val and mark."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(FIXTURES / "structures.xyz", tmp_path / "structures.xyz")
    (tmp_path / "foundation.model").touch()
    (tmp_path / "replay.xyz").touch()

    rc = gorun_main(
        [
            "mace-finetune", "--mark",
            "--model-path", "foundation.model",
            "--replay-xyz", "replay.xyz",
            "--data-path", "structures.xyz",
            "--config", CONFIG,
        ]
    )

    assert rc == 0
    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "val.xyz").exists()
    assert (tmp_path / "heads.json").exists()
    assert (tmp_path / "gorun_ready").exists()
    sub = (tmp_path / "sub").read_text()
    assert "Stage 1: fine_tuning_select" in sub
    assert "Stage 2: run_train" in sub
    heads = (tmp_path / "heads.json").read_text()
    assert "train.xyz" in heads
    assert "val.xyz" in heads


def test_mace_bootstrap(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README: first run in an empty dir bootstraps INCAR.toml and exits 1."""
    monkeypatch.chdir(tmp_path)

    rc = gorun_main(["mace-finetune"])

    assert rc == 1
    assert (tmp_path / "INCAR.toml").exists()


def test_mace_runfile_override(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README: RUNFILE.sh replaces the canned two-stage workflow."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(FIXTURES / "structures.xyz", tmp_path / "structures.xyz")
    (tmp_path / "foundation.model").touch()
    (tmp_path / "replay.xyz").touch()
    (tmp_path / "RUNFILE.sh").write_text("#!/bin/sh\necho runfile\n")

    rc = gorun_main(
        [
            "mace-finetune", "--mark",
            "--model-path", "foundation.model",
            "--replay-xyz", "replay.xyz",
            "--data-path", "structures.xyz",
            "--config", CONFIG,
        ]
    )

    assert rc == 0
    sub = (tmp_path / "sub").read_text()
    assert "RUNFILE.sh" in sub
    assert "Stage 1" not in sub


# --- gpu --------------------------------------------------------------------


def test_gpu_mark(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README gorun gpu --mark wraps the RUNFILE in the heredoc."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "RUNFILE.sh").write_text("#!/bin/sh\ntouch gpu_ran\n")

    rc = gorun_main(["gpu", "--mark", "--config", CONFIG])

    assert rc == 0
    assert (tmp_path / "gorun_ready").exists()
    sub = (tmp_path / "sub").read_text()
    assert "RUNFILE.sh" in sub
    assert "GORUN_WRAPPER" in sub


def test_gpu_local_executes_runfile(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README gorun gpu --local actually runs the RUNFILE."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "RUNFILE.sh").write_text("#!/bin/sh\ntouch gpu_ran\n")

    rc = gorun_main(["gpu", "--local", "--config", CONFIG])

    assert rc == 0
    assert (tmp_path / "gpu_ran").exists()


# --- batch submission -------------------------------------------------------


def test_batch_submission(fake_cluster, tmp_path: Path, monkeypatch) -> None:
    """README "Batch Submission": gorun-all-ready.sh submits marked dirs."""
    for name in ("d1", "d2"):
        d = tmp_path / name
        d.mkdir()
        (d / "sub").write_text("#!/bin/sh\necho job\n")
        (d / "gorun_ready").touch()

    script = SCRIPTS / "gorun-all-ready.sh"
    r = subprocess.run(
        ["bash", str(script), "-d", str(tmp_path), "-q"],
        capture_output=True,
        text=True,
    )

    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "d1" / "gorun_ready").exists()
    assert not (tmp_path / "d2" / "gorun_ready").exists()
    assert fake_cluster.log.read_text().count("sbatch sub") == 2
