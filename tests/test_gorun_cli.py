"""Tests for IMDgroup.gorun.gorun (CLI argument handling)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import IMDgroup.gorun.gorun as gorun


# --- _parse_args ------------------------------------------------------------


def test_parse_args_detects_vasp_subcommand() -> None:
    """The vasp subcommand sets software and its flag defaults."""
    ns = gorun._parse_args(["vasp"])
    assert ns.software == "vasp"
    assert ns.vasp == "ncl"
    assert ns.number_of_nodes is None
    assert ns.time_limit is None


def test_parse_args_detects_mace_finetune_subcommand() -> None:
    """The mace-finetune subcommand is detected and SUPPRESS defaults absent."""
    ns = gorun._parse_args(["mace-finetune"])
    assert ns.software == "mace-finetune"
    assert ns.time_limit is None
    assert not hasattr(ns, "model_path")


def test_parse_args_plain_gorun_falls_back_to_vasp() -> None:
    """A bare invocation is treated as ``gorun vasp`` for backward compat."""
    ns = gorun._parse_args([])
    assert ns.software == "vasp"


def test_parse_args_plain_positionals_route_to_vasp() -> None:
    """Positional nodes/time without a subcommand parse as vasp args."""
    ns = gorun._parse_args(["2", "01:00:00"])
    assert ns.software == "vasp"
    assert ns.number_of_nodes == "2"
    assert ns.time_limit == "01:00:00"


def test_parse_args_help_without_subcommand_shows_top_level() -> None:
    """``--help`` without a subcommand exits 0 instead of vasp help."""
    with pytest.raises(SystemExit) as exc:
        gorun._parse_args(["--help"])
    assert exc.value.code == 0


def test_parse_args_vasp_incar_flag() -> None:
    """The vasp --incar flag captures KEY:VAL items verbatim."""
    ns = gorun._parse_args(["vasp", "--incar", "ALGO:Normal", "NELM:200"])
    assert ns.incar == ["ALGO:Normal", "NELM:200"]


def test_parse_args_dir_nargs_plus() -> None:
    """--dir collects one or more directories from a single flag."""
    ns = gorun._parse_args(["vasp", "--dir", "a", "b"])
    assert ns.dir == ["a", "b"]


def test_parse_args_dir_default_none() -> None:
    """--dir defaults to None when not provided."""
    ns = gorun._parse_args(["vasp"])
    assert ns.dir is None


# --- _make_vasp_adapter -----------------------------------------------------


def test_make_vasp_adapter_incar_parsing(monkeypatch) -> None:
    """--incar KEY:VAL pairs are parsed into an incar_mods dict."""
    captured: dict = {}

    def fake_vasp_adapter(**kwargs):
        captured.update(kwargs)
        return "adapter"

    monkeypatch.setattr(gorun, "VaspAdapter", fake_vasp_adapter)
    ns = gorun._parse_args(["vasp", "--incar", "ALGO:Normal", "NELM:200"])
    gorun._make_vasp_adapter(ns)
    assert captured["incar_mods"] == {"ALGO": "Normal", "NELM": "200"}


def test_make_vasp_adapter_legacy_keep_flags(monkeypatch) -> None:
    """--keep-potcar / --keep-poscar map to the --keep list."""
    monkeypatch.setattr(gorun, "VaspAdapter", lambda **kwargs: "adapter")
    ns = gorun._parse_args(["vasp", "--keep-potcar", "--keep-poscar"])
    gorun._make_vasp_adapter(ns)
    assert set(ns.keep) == {"POTCAR", "POSCAR"}


def test_make_vasp_adapter_keep_merges_with_legacy(monkeypatch) -> None:
    """An explicit --keep coexists with the legacy keep flags."""
    monkeypatch.setattr(gorun, "VaspAdapter", lambda **kwargs: "adapter")
    ns = gorun._parse_args(["vasp", "--keep", "WAVECAR", "--keep-potcar"])
    gorun._make_vasp_adapter(ns)
    assert set(ns.keep) == {"WAVECAR", "POTCAR"}


def test_make_vasp_adapter_invalid_incar_exits(monkeypatch) -> None:
    """A --incar item without a colon exits with code 1."""
    monkeypatch.setattr(gorun, "VaspAdapter", lambda **kwargs: "adapter")
    ns = gorun._parse_args(["vasp", "--incar", "NO_COLON"])
    with pytest.raises(SystemExit) as exc:
        gorun._make_vasp_adapter(ns)
    assert exc.value.code == 1


# --- _make_mace_adapter -----------------------------------------------------


def test_make_mace_adapter_section_routing(monkeypatch) -> None:
    """SECTION.KEY:VAL routes to a section; bare KEY:VAL routes to None."""
    captured: dict = {}

    def fake_mace_adapter(**kwargs):
        captured.update(kwargs)
        return "adapter"

    monkeypatch.setattr(gorun, "MaceMultiheadFinetuneAdapter", fake_mace_adapter)
    ns = gorun._parse_args(
        ["mace-finetune", "--incar", "run_train.lr:0.001", "batch_size:12"]
    )
    gorun._make_mace_adapter(ns)
    assert captured["incar_overrides"] == {
        "run_train": {"lr": "0.001"},
        None: {"batch_size": "12"},
    }


def test_make_mace_adapter_invalid_incar_exits(monkeypatch) -> None:
    """A mace --incar item without a colon exits with code 1."""
    monkeypatch.setattr(
        gorun, "MaceMultiheadFinetuneAdapter", lambda **kwargs: "adapter"
    )
    ns = gorun._parse_args(["mace-finetune", "--incar", "NO_COLON"])
    with pytest.raises(SystemExit) as exc:
        gorun._make_mace_adapter(ns)
    assert exc.value.code == 1


# --- _fill_namespace_defaults -----------------------------------------------


def test_fill_namespace_defaults_vasp_when_absent() -> None:
    """An empty Namespace is filled with vasp defaults."""
    ns = argparse.Namespace(mark=True)
    gorun._fill_namespace_defaults(ns)
    assert ns.software == "vasp"
    assert ns.mark is True
    assert hasattr(ns, "vasp")


def test_fill_namespace_defaults_unknown_software_falls_back_to_vasp() -> None:
    """An unknown software value is coerced to vasp."""
    ns = argparse.Namespace(software="bogus")
    gorun._fill_namespace_defaults(ns)
    assert ns.software == "vasp"


def test_fill_namespace_defaults_respects_gpu_software() -> None:
    """A gpu namespace keeps its subcommand and gains gpu defaults."""
    ns = argparse.Namespace(software="gpu")
    gorun._fill_namespace_defaults(ns)
    assert ns.software == "gpu"
    assert ns.time_limit is None
    assert hasattr(ns, "script_name")


def test_fill_namespace_defaults_maps_kpoints_placeholder() -> None:
    """maps gets an empty kpoints placeholder so the subcommand parses."""
    ns = argparse.Namespace(software="maps")
    gorun._fill_namespace_defaults(ns)
    assert ns.kpoints == ""


def test_fill_namespace_defaults_atat_local_kpoints_placeholder() -> None:
    """atat-local gets an empty kpoints placeholder."""
    ns = argparse.Namespace(software="atat-local")
    gorun._fill_namespace_defaults(ns)
    assert ns.kpoints == ""


# --- _dispatch_namespace ----------------------------------------------------


def test_dispatch_namespace_multiple_dirs(monkeypatch, tmp_path) -> None:
    """--dir dispatches once per directory, resolving relative paths."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    calls: list[Path] = []

    def fake_dispatch_single(args, directory):
        calls.append(directory)
        return 0

    monkeypatch.setattr(gorun, "_dispatch_single", fake_dispatch_single)
    ns = gorun._parse_args(["vasp", "--dir", "a", "b"])
    assert gorun._dispatch_namespace(ns) == 0
    assert calls == [(tmp_path / "a").resolve(), (tmp_path / "b").resolve()]


def test_dispatch_namespace_missing_dir_skips(monkeypatch, tmp_path) -> None:
    """A non-existent --dir is skipped with a warning, not an error."""
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        gorun, "_dispatch_single", lambda args, d: calls.append(d) or 0
    )
    ns = gorun._parse_args(["vasp", "--dir", "missing"])
    assert gorun._dispatch_namespace(ns) == 0
    assert calls == []


def test_dispatch_namespace_continues_on_error(monkeypatch, tmp_path) -> None:
    """A failing directory is reported and does not stop the loop."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    calls: list[Path] = []

    def fake_dispatch_single(args, directory):
        calls.append(directory)
        if directory.name == "a":
            raise ValueError("boom")
        return 0

    monkeypatch.setattr(gorun, "_dispatch_single", fake_dispatch_single)
    ns = gorun._parse_args(["vasp", "--dir", "a", "b"])
    assert gorun._dispatch_namespace(ns) == 1
    assert calls == [(tmp_path / "a").resolve(), (tmp_path / "b").resolve()]


def test_dispatch_namespace_single_dir_default(monkeypatch, tmp_path) -> None:
    """Without --dir, dispatch runs once in the current directory."""
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        gorun, "_dispatch_single", lambda args, d: calls.append(d) or 0
    )
    ns = gorun._parse_args(["vasp"])
    assert gorun._dispatch_namespace(ns) == 0
    assert calls == [tmp_path.resolve()]


# --- run() ------------------------------------------------------------------


def test_run_namespace_delegation(monkeypatch) -> None:
    """A Namespace is filled and passed to _dispatch_namespace."""
    calls: list = []

    def fake_dispatch(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(gorun, "_dispatch_namespace", fake_dispatch)
    assert gorun.run(argparse.Namespace(software="vasp")) == 0
    assert calls[0].software == "vasp"


def test_run_none_delegates_as_empty_namespace(monkeypatch) -> None:
    """run() with no argument behaves as an empty Namespace."""
    calls: list = []
    monkeypatch.setattr(gorun, "_dispatch_namespace", lambda args: calls.append(args) or 0)
    assert gorun.run() == 0
    assert calls[0].software == "vasp"


def test_run_list_delegates_to_main(monkeypatch) -> None:
    """A list argument is delegated to main() as argv."""
    calls: list = []
    monkeypatch.setattr(gorun, "main", lambda argv: calls.append(argv) or 0)
    assert gorun.run(["vasp"]) == 0
    assert calls == [["vasp"]]
