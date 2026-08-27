"""Tests for IMDgroup.gorun.adapters.mace (marked heavy).

The module imports pandas, ase, and sklearn at the top, so this whole
module is marked =heavy= in pyproject.toml.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import write as ase_write

import IMDgroup.gorun.adapters.mace as mace_mod

from IMDgroup.gorun.adapters.mace import (
    MaceMultiheadFinetuneAdapter,
    _format_cli_args,
    _read_structures,
    _strip_constraints,
)

pytestmark = pytest.mark.heavy


@pytest.fixture(autouse=True)
def _no_toml(monkeypatch):
    """Keep the constructor from reading a real INCAR.toml in cwd."""
    monkeypatch.setattr(mace_mod, "read_incar_toml", lambda: {})


def _make(**kwargs) -> MaceMultiheadFinetuneAdapter:
    """Build an adapter with required params set so bootstrap is skipped."""
    params: dict = {"model_path": "foundation.model", "replay_xyz": "replay.xyz"}
    params.update(kwargs)
    return MaceMultiheadFinetuneAdapter(**params)


# --- _format_cli_args -------------------------------------------------------


def test_format_cli_args_empty() -> None:
    """An empty dict formats to an empty string."""
    assert _format_cli_args({}) == ""


def test_format_cli_args_bool_true_becomes_flag() -> None:
    """A True boolean produces a bare --flag."""
    assert _format_cli_args({"swa": True}) == "  --swa \\"


def test_format_cli_args_skips_false_none_empty() -> None:
    """False, None, and empty-string values are silently skipped."""
    assert _format_cli_args({"a": False, "b": None, "c": ""}) == ""


def test_format_cli_args_underscore_to_hyphen() -> None:
    """Underscores in keys become hyphens."""
    assert _format_cli_args({"batch_size": 12}) == "  --batch-size 12 \\"


def test_format_cli_args_mixed() -> None:
    """A mix of flags and valued args is joined in insertion order."""
    assert _format_cli_args({"lr": 0.001, "ema": True, "skip": False}) == (
        "  --lr 0.001 \\\n  --ema \\"
    )


# --- _read_structures -------------------------------------------------------


def test_read_structures_xyz(tmp_path: Path) -> None:
    """An .xyz file is read into a list of Atoms."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    ase_write(str(tmp_path / "s.xyz"), atoms)
    structures = _read_structures(str(tmp_path / "s.xyz"))
    assert len(structures) == 1
    assert isinstance(structures[0], Atoms)


def test_read_structures_pkl(tmp_path: Path) -> None:
    """A .pkl DataFrame with a structure column is read."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    pd.DataFrame({"structure": [atoms, atoms.copy()]}).to_pickle(
        str(tmp_path / "s.pkl")
    )
    structures = _read_structures(str(tmp_path / "s.pkl"))
    assert len(structures) == 2


def test_read_structures_unsupported_raises(tmp_path: Path) -> None:
    """An unknown suffix raises ValueError."""
    (tmp_path / "s.foo").write_text("x")
    with pytest.raises(ValueError):
        _read_structures(str(tmp_path / "s.foo"))


# --- _strip_constraints -----------------------------------------------------


def test_strip_constraints_preserves_calculator() -> None:
    """Constraints are removed; energy is preserved via SinglePointCalculator."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=True)
    atoms.calc = EMT()
    energy_before = atoms.get_potential_energy()
    atoms.set_constraint(FixAtoms([0]))

    assert _strip_constraints([atoms]) == 1
    assert len(atoms.constraints) == 0
    assert atoms.calc is not None
    assert atoms.get_potential_energy() == pytest.approx(energy_before)


def test_strip_constraints_noop_without_constraints() -> None:
    """Structures without constraints are left alone and not counted."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert _strip_constraints([atoms]) == 0
    assert len(atoms.constraints) == 0


# --- constructor merge logic -------------------------------------------------


def test_constructor_cli_kwargs_set_attributes() -> None:
    """CLI kwargs populate the corresponding attributes."""
    adapter = _make(model_path="m", replay_xyz="r", data_path="d", seed=3)
    assert adapter.model_path == "m"
    assert adapter.replay_xyz == "r"
    assert adapter.data_path == "d"
    assert adapter.seed == 3


def test_constructor_toml_scalars_and_sections(monkeypatch) -> None:
    """TOML top-level scalars and section tables are merged in."""
    monkeypatch.setattr(
        mace_mod,
        "read_incar_toml",
        lambda: {
            "model_path": "m_toml",
            "replay_xyz": "r_toml",
            "seed": 5,
            "run_train": {"lr": 0.001},
        },
    )
    adapter = MaceMultiheadFinetuneAdapter()
    assert adapter.model_path == "m_toml"
    assert adapter.replay_xyz == "r_toml"
    assert adapter.seed == 5
    assert adapter._train_args["lr"] == 0.001


def test_constructor_cli_overrides_toml(monkeypatch) -> None:
    """Explicit CLI kwargs win over INCAR.toml values."""
    monkeypatch.setattr(
        mace_mod,
        "read_incar_toml",
        lambda: {"model_path": "m_toml", "replay_xyz": "r_toml", "seed": 5},
    )
    adapter = MaceMultiheadFinetuneAdapter(seed=3)
    assert adapter.seed == 3
    assert adapter.model_path == "m_toml"


def test_constructor_incar_overrides_section_routing(monkeypatch) -> None:
    """--incar SECTION.KEY routes explicitly; bare keys auto-route."""
    adapter = _make(
        incar_overrides={
            "run_train": {"lr": "0.01"},
            None: {"batch_size": "12"},
        }
    )
    assert adapter._train_args["lr"] == "0.01"
    assert adapter._train_args["batch_size"] == "12"


def test_constructor_bootstrap_raises(monkeypatch) -> None:
    """No TOML and no CLI input writes a reference and raises ValueError."""
    monkeypatch.setattr(mace_mod, "read_incar_toml", lambda: {})
    writes = []
    monkeypatch.setattr(
        mace_mod,
        "write_incar_toml_sections",
        lambda *args, **kwargs: writes.append(args),
    )
    with pytest.raises(ValueError):
        MaceMultiheadFinetuneAdapter()
    assert len(writes) == 1


def test_constructor_validate_missing_required_raises() -> None:
    """Empty required params reach validate() and raise ValueError."""
    with pytest.raises(ValueError):
        MaceMultiheadFinetuneAdapter(model_path="", replay_xyz="")


# --- validate_environment ---------------------------------------------------


def test_validate_environment_missing_files(monkeypatch, tmp_path: Path) -> None:
    """Missing foundation model or replay file raises FileNotFoundError."""
    adapter = _make()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        adapter.validate_environment()


def test_validate_environment_ok(monkeypatch, tmp_path: Path) -> None:
    """Existing files pass validation."""
    adapter = _make()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foundation.model").touch()
    (tmp_path / "replay.xyz").touch()
    adapter.validate_environment()


# --- is_valid_input / is_converged / has_previous_output --------------------


def test_is_valid_input(monkeypatch, tmp_path: Path) -> None:
    """True only when at least one data source file exists."""
    adapter = _make(data_path="data.xyz")
    monkeypatch.chdir(tmp_path)
    assert adapter.is_valid_input(tmp_path) is False
    (tmp_path / "data.xyz").touch()
    assert adapter.is_valid_input(tmp_path) is True


def test_is_converged(monkeypatch, tmp_path: Path) -> None:
    """True when the output model exists and is non-empty."""
    adapter = _make(new_model_name="out.model")
    monkeypatch.chdir(tmp_path)
    assert adapter.is_converged(tmp_path) is False
    (tmp_path / "out.model").write_text("x")
    assert adapter.is_converged(tmp_path) is True


def test_has_previous_output(monkeypatch, tmp_path: Path) -> None:
    """True when a prior-run marker exists."""
    adapter = _make(new_model_name="out.model")
    monkeypatch.chdir(tmp_path)
    assert adapter.has_previous_output(tmp_path) is False
    (tmp_path / "log_tuning").touch()
    assert adapter.has_previous_output(tmp_path) is True


# --- _write_heads_json ------------------------------------------------------


def test_write_heads_json(tmp_path: Path) -> None:
    """heads.json is written with train/val paths and E0s."""
    adapter = _make(e0s="my_e0")
    adapter._write_heads_json(tmp_path)
    data = json.loads((tmp_path / "heads.json").read_text())
    assert data["default"]["train_file"] == str(tmp_path / "train.xyz")
    assert data["default"]["valid_file"] == str(tmp_path / "val.xyz")
    assert data["default"]["E0s"] == "my_e0"
    assert data["pt_head"]["E0s"] == "foundation"


# --- generate_run_commands / stage scripts ----------------------------------


def test_generate_run_commands_runfile_sh(tmp_path: Path) -> None:
    """RUNFILE.sh takes precedence and is wrapped."""
    (tmp_path / "RUNFILE.sh").touch()
    out = _make().generate_run_commands(tmp_path)
    assert "RUNFILE.sh" in out
    assert "bash" in out


def test_generate_run_commands_runfile_py(tmp_path: Path) -> None:
    """RUNFILE.py runs via python -u."""
    (tmp_path / "RUNFILE.py").touch()
    out = _make().generate_run_commands(tmp_path)
    assert "RUNFILE.py" in out
    assert "python -u" in out


def test_generate_run_commands_two_stages(tmp_path: Path) -> None:
    """Without a RUNFILE, both canned stages are emitted."""
    out = _make().generate_run_commands(tmp_path)
    assert "Stage 1: fine_tuning_select" in out
    assert "Stage 2: run_train" in out


def test_generate_run_commands_skip_select(tmp_path: Path) -> None:
    """--no-fine-tuning-select omits the select stage."""
    out = _make(no_fine_tuning_select=True).generate_run_commands(tmp_path)
    assert "Stage 1" not in out
    assert "Stage 2: run_train" in out


def test_select_stage_string(tmp_path: Path) -> None:
    """The select stage forwards replay/model/output args."""
    out = _make()._select_stage(tmp_path)
    assert "mace.cli.fine_tuning_select" in out
    assert "--configs_pt replay.xyz" in out
    assert "--num_samples 30000" in out
    assert "--model foundation.model" in out


def test_train_stage_default_module_and_toggles(tmp_path: Path) -> None:
    """The default train stage uses mace.cli.run_train with SWA/EMA on."""
    out = _make()._train_stage(tmp_path)
    assert "mace.cli.run_train" in out
    assert "--swa" in out
    assert "--ema" in out


def test_train_stage_masked_loss_module(tmp_path: Path) -> None:
    """masked_loss switches the training module."""
    out = _make(masked_loss=True)._train_stage(tmp_path)
    assert "IMDgroup.gorun.adapters._mace_train_masked" in out


def test_train_stage_swa_ema_off(tmp_path: Path) -> None:
    """When the toggles are off, the SWA/EMA groups are not emitted."""
    adapter = _make()
    adapter._train_args["swa"] = False
    adapter._train_args["ema"] = False
    out = adapter._train_stage(tmp_path)
    assert "--swa" not in out
    assert "--ema" not in out


# --- prepare_inputs ---------------------------------------------------------


def test_prepare_inputs_orchestration(monkeypatch, tmp_path: Path) -> None:
    """prepare_inputs writes back INCAR.toml and regenerates train/heads."""
    adapter = _make()
    writes = []
    monkeypatch.setattr(
        mace_mod,
        "write_incar_toml_sections",
        lambda *args, **kwargs: writes.append(args),
    )
    regenerated = []
    monkeypatch.setattr(
        mace_mod,
        "maybe_regenerate",
        lambda path, filename, keep, regenerate=None, older_than=None: regenerated.append(filename),
    )
    adapter.prepare_inputs(tmp_path, keep=set())
    assert len(writes) == 1
    assert regenerated == [["train.xyz", "val.xyz"], "heads.json"]
