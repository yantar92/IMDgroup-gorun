"""Tests for IMDgroup.gorun.adapters.vasp."""

from __future__ import annotations

from pathlib import Path

import pytest

import IMDgroup.gorun.adapters.vasp as vasp_mod

from IMDgroup.gorun.adapters.vasp import (
    VaspAdapter,
    _has_incar_chain,
    _modify_incar,
    _rotate_incar_chain,
)


def _adapter(**kwargs) -> VaspAdapter:
    return VaspAdapter(**kwargs)


# --- validate_environment ---------------------------------------------------


def test_validate_environment_missing_vars_raises(monkeypatch) -> None:
    """Missing VASP_PATH and VASP_PP_PATH raise OSError."""
    monkeypatch.delenv("VASP_PATH", raising=False)
    monkeypatch.delenv("VASP_PP_PATH", raising=False)
    with pytest.raises(OSError):
        _adapter().validate_environment()


def test_validate_environment_one_missing_raises(monkeypatch) -> None:
    """A single missing variable is reported."""
    monkeypatch.setenv("VASP_PATH", "/x")
    monkeypatch.delenv("VASP_PP_PATH", raising=False)
    with pytest.raises(OSError):
        _adapter().validate_environment()


def test_validate_environment_ok(monkeypatch) -> None:
    """Both variables set passes validation."""
    monkeypatch.setenv("VASP_PATH", "/x")
    monkeypatch.setenv("VASP_PP_PATH", "/y")
    _adapter().validate_environment()


# --- is_valid_input ---------------------------------------------------------


def test_is_valid_input_requires_incar(tmp_path: Path) -> None:
    """is_valid_input is True only when INCAR exists."""
    assert _adapter().is_valid_input(tmp_path) is False
    (tmp_path / "INCAR").touch()
    assert _adapter().is_valid_input(tmp_path) is True


# --- is_converged -----------------------------------------------------------


def test_is_converged_false_when_not_converged(monkeypatch, tmp_path: Path) -> None:
    """Not converged according to the cleanVASP helper."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    assert _adapter().is_converged(tmp_path) is False


def test_is_converged_md_true_without_chain(monkeypatch, tmp_path: Path) -> None:
    """An MD run that converged is True regardless of INCAR chains."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: True)
    (tmp_path / "INCAR.0").touch()
    assert _adapter().is_converged(tmp_path) is True


def test_is_converged_false_with_chain(monkeypatch, tmp_path: Path) -> None:
    """Converged but with INCAR chains remaining is not done."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    (tmp_path / "INCAR.0").touch()
    assert _adapter().is_converged(tmp_path) is False


def test_is_converged_true_without_chain(monkeypatch, tmp_path: Path) -> None:
    """Converged, non-MD, and no chains is done."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    assert _adapter().is_converged(tmp_path) is True


# --- has_previous_output ----------------------------------------------------


def test_has_previous_output_incar_py(tmp_path: Path) -> None:
    """An INCAR.py counts as previous output when not suppressed."""
    (tmp_path / "INCAR.py").touch()
    assert _adapter().has_previous_output(tmp_path) is True


def test_has_previous_output_incar_py_suppressed(tmp_path: Path) -> None:
    """--no-incar-py ignores INCAR.py."""
    (tmp_path / "INCAR.py").touch()
    assert _adapter(no_incar_py=True).has_previous_output(tmp_path) is False


def test_has_previous_output_from_vasp(monkeypatch, tmp_path: Path) -> None:
    """Without INCAR.py, the cleanVASP output check decides."""
    monkeypatch.setattr(
        vasp_mod, "directory_contains_vasp_outputp", lambda path: True
    )
    assert _adapter(no_incar_py=True).has_previous_output(tmp_path) is True


# --- _has_incar_chain / _rotate_incar_chain ---------------------------------


def test_has_incar_chain_false_when_empty(tmp_path: Path) -> None:
    """No INCAR.N files means no chain."""
    assert _has_incar_chain(tmp_path) is False


def test_has_incar_chain_true_with_numbered_files(tmp_path: Path) -> None:
    """INCAR.0 / INCAR.1 files mark a chain."""
    (tmp_path / "INCAR.0").touch()
    (tmp_path / "INCAR.1").touch()
    assert _has_incar_chain(tmp_path) is True


def test_has_incar_chain_ignores_incar_old(tmp_path: Path) -> None:
    """INCAR.old is not a chain member."""
    (tmp_path / "INCAR.old").touch()
    assert _has_incar_chain(tmp_path) is False


def test_rotate_incar_chain_noop_when_not_converged(monkeypatch, tmp_path: Path) -> None:
    """Not converged means no rotation."""
    (tmp_path / "INCAR.0").touch()
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    _rotate_incar_chain(tmp_path)
    assert (tmp_path / "INCAR").exists() is False
    assert (tmp_path / "INCAR.0").exists()


def test_rotate_incar_chain_rotates_first(monkeypatch, tmp_path: Path) -> None:
    """The first INCAR.N is renamed to INCAR; the old INCAR becomes INCAR.old."""
    (tmp_path / "INCAR").write_text("old")
    (tmp_path / "INCAR.0").write_text("first")
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    _rotate_incar_chain(tmp_path)
    assert (tmp_path / "INCAR").read_text() == "first"
    assert (tmp_path / "INCAR.old").read_text() == "old"
    assert not (tmp_path / "INCAR.0").exists()


# --- _modify_incar ----------------------------------------------------------


def test_modify_incar_changes_value(tmp_path: Path) -> None:
    """A key is set and written back to INCAR."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    _modify_incar(tmp_path, {"ALGO": "Normal"})
    content = (tmp_path / "INCAR").read_text()
    assert "ALGO = Normal" in content
    assert "ENCUT" in content


def test_modify_incar_none_removes_key(tmp_path: Path) -> None:
    """A value of the string None removes the key."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\nEDIFF = 1e-6\n")
    _modify_incar(tmp_path, {"EDIFF": "None"})
    assert "EDIFF" not in (tmp_path / "INCAR").read_text()


# --- setup_commands ---------------------------------------------------------


def test_setup_commands_vasp_section(tmp_path: Path) -> None:
    """The new [vasp] setup key is preferred."""
    cfg = {"vasp": {"setup": "module load vasp"}}
    assert _adapter().setup_commands(cfg) == "module load vasp"


def test_setup_commands_legacy_fallback() -> None:
    """Without [vasp], the legacy VASP-setup key is used."""
    cfg = {"VASP-setup": "module load legacy"}
    assert _adapter().setup_commands(cfg) == "module load legacy"


def test_setup_commands_no_vasp_config() -> None:
    """--no-vasp-config returns an empty string."""
    cfg = {"vasp": {"setup": "module load vasp"}}
    assert _adapter(no_vasp_config=True).setup_commands(cfg) == ""


# --- _direct_vasp_script ----------------------------------------------------


def test_direct_vasp_script_default_mpiexec(monkeypatch) -> None:
    """The direct script exports VASP_COMMAND with the default launcher."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter(vasp_variant="std")
    script = adapter._direct_vasp_script({})
    assert 'export VASP_COMMAND="mpiexec /vasp/bin/vasp_std"' in script
    assert script.endswith("$VASP_COMMAND")


def test_direct_vasp_script_custom_mpiexec(monkeypatch) -> None:
    """A server mpiexec key overrides the default launcher."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter()
    script = adapter._direct_vasp_script({"mpiexec": "srun"})
    assert 'export VASP_COMMAND="srun /vasp/bin/vasp_ncl"' in script


# --- _incar_py_script -------------------------------------------------------


def test_incar_py_script_string(monkeypatch) -> None:
    """The INCAR.py script wraps the python heredoc and exports VASP_COMMAND."""
    adapter = _adapter()
    script = adapter._incar_py_script(Path("."))
    assert script.startswith('export VASP_COMMAND="gorun vasp --local')
    assert "python <<EOF" in script
    assert "$(cat INCAR.py)" in script
    assert "Path(\"UNCONVERGED\").touch()" in script


def test_generate_run_commands_prefers_incar_py(monkeypatch, tmp_path: Path) -> None:
    """With INCAR.py present, the ASE wrapper path is used."""
    (tmp_path / "INCAR.py").touch()
    adapter = _adapter()
    script = adapter.generate_run_commands(tmp_path, {})
    assert "python <<EOF" in script


def test_generate_run_commands_direct_without_incar_py(monkeypatch, tmp_path: Path) -> None:
    """Without INCAR.py the direct VASP path is used."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter()
    script = adapter.generate_run_commands(tmp_path, {})
    assert "export VASP_COMMAND=" in script


# --- prepare_inputs ---------------------------------------------------------


def test_prepare_inputs_potcar_keep_skip(monkeypatch, tmp_path: Path) -> None:
    """--keep POTCAR does not regenerate an existing POTCAR."""
    (tmp_path / "INCAR").touch()
    (tmp_path / "POTCAR").touch()
    monkeypatch.setattr(vasp_mod, "check_incar", lambda path: None)
    monkeypatch.setattr(vasp_mod, "contcar_to_poscar", lambda path: None)
    monkeypatch.setattr(vasp_mod, "clean_vasp_inputs", lambda path: None)
    monkeypatch.setattr(vasp_mod, "put_vdw_kernel", lambda path: None)
    monkeypatch.setattr(vasp_mod, "clear_vasp_outputs", lambda path: None)

    regenerated = []
    monkeypatch.setattr(
        vasp_mod, "_regenerate_potcar", lambda path: regenerated.append(path)
    )

    adapter = _adapter()
    adapter.prepare_inputs(tmp_path, keep={"POTCAR"})
    assert regenerated == []


def test_prepare_inputs_potcar_missing_regenerates(monkeypatch, tmp_path: Path) -> None:
    """A missing POTCAR triggers regeneration via _regenerate_potcar."""
    (tmp_path / "INCAR").touch()
    monkeypatch.setattr(vasp_mod, "check_incar", lambda path: None)
    monkeypatch.setattr(vasp_mod, "contcar_to_poscar", lambda path: None)
    monkeypatch.setattr(vasp_mod, "clean_vasp_inputs", lambda path: None)
    monkeypatch.setattr(vasp_mod, "put_vdw_kernel", lambda path: None)
    monkeypatch.setattr(vasp_mod, "clear_vasp_outputs", lambda path: None)

    regenerated = []
    monkeypatch.setattr(
        vasp_mod, "_regenerate_potcar", lambda path: regenerated.append(path)
    )

    adapter = _adapter()
    adapter.prepare_inputs(tmp_path, keep=set())
    assert regenerated == [str(tmp_path)]
