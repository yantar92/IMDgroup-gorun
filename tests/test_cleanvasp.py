"""Tests for IMDgroup.gorun.cleanVASP."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.io.vasp.inputs import BadIncarWarning

import IMDgroup.gorun.cleanVASP as clean_mod

from IMDgroup.gorun.cleanVASP import (
    check_incar,
    clean_vasp_input,
    clean_vasp_inputs,
    clear_useless_vasp_files,
    clear_vasp_outputs,
    contcar_to_poscar,
    directory_contains_vasp_outputp,
    mdp,
    nebp,
)


# --- directory_contains_vasp_outputp ----------------------------------------


def test_directory_contains_vasp_outputp_empty_outcar_is_false(tmp_path: Path) -> None:
    """A zero-size OUTCAR is not considered output."""
    (tmp_path / "OUTCAR").touch()
    assert directory_contains_vasp_outputp(str(tmp_path)) is False


def test_directory_contains_vasp_outputp_nonempty_outcar_is_true(tmp_path: Path) -> None:
    """A non-empty OUTCAR marks the directory as containing output."""
    (tmp_path / "OUTCAR").write_text("data")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


def test_directory_contains_vasp_outputp_neb_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    """With a NEB layout, output in a numbered subdir counts."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    sub = tmp_path / "00"
    sub.mkdir()
    (sub / "OUTCAR").write_text("data")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


# --- nebp / mdp -------------------------------------------------------------


def test_nebp_true_with_images(tmp_path: Path) -> None:
    """An INCAR with IMAGES marks a NEB run."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    assert nebp(str(tmp_path)) is True


def test_nebp_false_without_images(tmp_path: Path) -> None:
    """An INCAR without IMAGES is not a NEB run."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    assert nebp(str(tmp_path)) is False


def test_nebp_false_when_incar_missing(tmp_path: Path) -> None:
    """No INCAR means not a NEB run."""
    assert nebp(str(tmp_path)) is False


def test_mdp_true_with_ibrion_zero(tmp_path: Path) -> None:
    """IBRION=0 marks an MD run."""
    (tmp_path / "INCAR").write_text("IBRION = 0\n")
    assert mdp(str(tmp_path)) is True


def test_mdp_false_with_ibrion_nonzero(tmp_path: Path) -> None:
    """A nonzero IBRION is not an MD run."""
    (tmp_path / "INCAR").write_text("IBRION = 2\n")
    assert mdp(str(tmp_path)) is False


# --- clear_useless_vasp_files / clear_vasp_outputs --------------------------


def test_clear_useless_vasp_files_removes_only_empty(tmp_path: Path) -> None:
    """Only zero-size cleanup targets are removed."""
    (tmp_path / "CHG").touch()
    (tmp_path / "WAVECAR").write_text("data")
    clear_useless_vasp_files(str(tmp_path))
    assert not (tmp_path / "CHG").exists()
    assert (tmp_path / "WAVECAR").exists()


def test_clear_vasp_outputs_removes_known_outputs(tmp_path: Path) -> None:
    """All known VASP output files are removed; unrelated files stay."""
    for name in ("CHG", "OUTCAR", "vasprun.xml", "XDATCAR"):
        (tmp_path / name).touch()
    (tmp_path / "POSCAR").touch()
    clear_vasp_outputs(str(tmp_path))
    for name in ("CHG", "OUTCAR", "vasprun.xml", "XDATCAR"):
        assert not (tmp_path / name).exists()
    assert (tmp_path / "POSCAR").exists()


# --- contcar_to_poscar ------------------------------------------------------


def test_contcar_to_poscar_copies(tmp_path: Path) -> None:
    """A non-empty CONTCAR is copied over POSCAR."""
    (tmp_path / "CONTCAR").write_text("relaxed")
    (tmp_path / "POSCAR").write_text("initial")
    contcar_to_poscar(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "relaxed"


def test_contcar_to_poscar_skips_when_absent(tmp_path: Path) -> None:
    """Without a CONTCAR, POSCAR is left untouched."""
    (tmp_path / "POSCAR").write_text("initial")
    contcar_to_poscar(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "initial"


# --- clean_vasp_input / clean_vasp_inputs -----------------------------------


def test_clean_vasp_input_normalizes_crlf_and_bom(tmp_path: Path) -> None:
    """CRLF line endings and a UTF-8 BOM are stripped."""
    p = tmp_path / "INCAR"
    p.write_bytes(b"\xEF\xBB\xBFENCUT = 400\r\nEDIFF = 1e-6\r\n")
    clean_vasp_input(str(p))
    assert p.read_text(encoding="utf-8") == "ENCUT = 400\nEDIFF = 1e-6\n"


def test_clean_vasp_input_removes_tab_only_lines(tmp_path: Path) -> None:
    """Lines containing only tabs are removed."""
    p = tmp_path / "INCAR"
    p.write_text("ENCUT = 400\n\t\t\nEDIFF = 1e-6\n")
    clean_vasp_input(str(p))
    assert p.read_text() == "ENCUT = 400\n\nEDIFF = 1e-6\n"


def test_clean_vasp_inputs_loops_over_input_files(tmp_path: Path) -> None:
    """POSCAR/INCAR/KPOINTS are each cleaned; absent files are skipped."""
    (tmp_path / "POSCAR").write_bytes(b"\xEF\xBB\xBFline\r\n")
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    clean_vasp_inputs(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "line\n"
    assert (tmp_path / "INCAR").read_text() == "ENCUT = 400\n"


# --- check_incar ------------------------------------------------------------


def test_check_incar_valid(tmp_path: Path) -> None:
    """A valid INCAR passes without warning."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\nEDIFF = 1e-6\n")
    check_incar(str(tmp_path))


def test_check_incar_invalid_value_warns(tmp_path: Path) -> None:
    """A recognized tag with an invalid value raises BadIncarWarning."""
    (tmp_path / "INCAR").write_text("ISPIN = 3\n")
    with pytest.warns(BadIncarWarning):
        check_incar(str(tmp_path))


def test_check_incar_unknown_tag_warns(tmp_path: Path) -> None:
    """An unrecognized tag raises BadIncarWarning."""
    (tmp_path / "INCAR").write_text("FOOBAR = 1\n")
    with pytest.warns(BadIncarWarning):
        check_incar(str(tmp_path))
