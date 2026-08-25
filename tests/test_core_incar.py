"""Tests for IMDgroup.gorun.core.incar."""

from __future__ import annotations

from pathlib import Path

from IMDgroup.gorun.core.incar import (
    _NO_DEFAULT,
    _flatten_toml,
    _key_in_defaults,
    _resolve_default_value,
    _toml_format_value,
    read_incar_toml,
    write_incar_toml,
    write_incar_toml_sections,
)


# --- _toml_format_value -----------------------------------------------------


def test_toml_format_value_bool() -> None:
    """Booleans become unquoted true/false literals."""
    assert _toml_format_value(True) == "true"
    assert _toml_format_value(False) == "false"


def test_toml_format_value_numbers() -> None:
    """Ints and floats are written unquoted."""
    assert _toml_format_value(1) == "1"
    assert _toml_format_value(1.5) == "1.5"


def test_toml_format_value_string_escapes_quotes_and_backslashes() -> None:
    """Strings are quoted and escape backslashes plus double quotes."""
    assert _toml_format_value("plain") == '"plain"'
    assert _toml_format_value('say "hi"') == '"say \\"hi\\""'
    assert _toml_format_value("back\\slash") == '"back\\\\slash"'


def test_toml_format_value_nonscalar_falls_back_to_repr() -> None:
    """Non-scalar values are stringified and quoted as a fallback."""
    assert _toml_format_value([1, 2]) == '"[1, 2]"'
    assert _toml_format_value(None) == '"None"'


# --- _flatten_toml ----------------------------------------------------------


def test_flatten_toml_merges_sections_into_top_level() -> None:
    """Nested section dicts are flattened; section names are dropped."""
    raw = {"top": 1, "sec": {"a": 2, "b": 3}}
    assert _flatten_toml(raw) == {"top": 1, "a": 2, "b": 3}


def test_flatten_toml_passes_non_dict_values_through() -> None:
    """Non-dict values are copied unchanged."""
    raw = {"x": "v", "y": 4}
    assert _flatten_toml(raw) == {"x": "v", "y": 4}


def test_flatten_toml_later_sections_win_on_key_collision() -> None:
    """When two sections share a key, the later section wins."""
    raw = {"s1": {"k": 1}, "s2": {"k": 2}}
    assert _flatten_toml(raw) == {"k": 2}


# --- _resolve_default_value / _key_in_defaults ------------------------------


def test_resolve_default_value_from_section() -> None:
    """A key mapped to a section resolves inside that section's dict."""
    defaults = {"sec": {"alpha": 10}}
    key_section = {"alpha": "sec"}
    assert _resolve_default_value(defaults, key_section, "alpha") == 10


def test_resolve_default_value_top_level() -> None:
    """A top-level key (section None) resolves from the top-level dict."""
    defaults = {"alpha": 10}
    key_section = {"alpha": None}
    assert _resolve_default_value(defaults, key_section, "alpha") == 10


def test_resolve_default_value_missing_key() -> None:
    """A key with no default resolves to the _NO_DEFAULT sentinel."""
    defaults = {"sec": {"alpha": 10}}
    key_section = {"beta": "sec"}
    assert _resolve_default_value(defaults, key_section, "beta") is _NO_DEFAULT


def test_resolve_default_value_section_missing_falls_back_to_top_level() -> None:
    """A key mapped to an absent section falls back to the top level."""
    defaults = {"beta": 5}
    key_section = {"beta": "missing_section"}
    assert _resolve_default_value(defaults, key_section, "beta") == 5


def test_key_in_defaults_true() -> None:
    """A key with a defined default reports as present."""
    defaults = {"sec": {"k": 1}}
    key_section = {"k": "sec"}
    assert _key_in_defaults(defaults, key_section, "k") is True


def test_key_in_defaults_false() -> None:
    """A key with no default reports as absent."""
    defaults = {}
    key_section = {"k": "sec"}
    assert _key_in_defaults(defaults, key_section, "k") is False


# --- read_incar_toml --------------------------------------------------------


def test_read_incar_toml_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing file yields an empty dict."""
    assert read_incar_toml(tmp_path / "missing.toml") == {}


def test_read_incar_toml_valid_file(tmp_path: Path) -> None:
    """A valid TOML file is parsed with native value types."""
    path = tmp_path / "INCAR.toml"
    path.write_text('key = "value"\nnumber = 3\nflag = true\n')
    assert read_incar_toml(path) == {"key": "value", "number": 3, "flag": True}


# --- write_incar_toml -------------------------------------------------------


def test_write_incar_toml_bootstrap_writes_all_non_none(tmp_path: Path) -> None:
    """Bootstrap writes every non-None value and skips None."""
    path = tmp_path / "INCAR.toml"
    write_incar_toml({"a": 1, "b": "x", "c": None}, path=path)
    assert path.read_text() == 'a = 1\nb = "x"\n'


def test_write_incar_toml_bootstrap_always_include_writes_none(tmp_path: Path) -> None:
    """Bootstrap writes None values for keys in always_include."""
    path = tmp_path / "INCAR.toml"
    write_incar_toml({"a": None, "b": 2}, always_include={"a"}, path=path)
    assert path.read_text() == 'a = "None"\nb = 2\n'


def test_write_incar_toml_update_skips_defaults(tmp_path: Path) -> None:
    """Update mode omits values equal to defaults unless already present."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 1, "b": 3},
        raw_existing={"a": 1},
        defaults={"a": 1, "b": 2},
        path=path,
    )
    assert path.read_text() == "a = 1\nb = 3\n"


def test_write_incar_toml_update_preserves_user_keys(tmp_path: Path) -> None:
    """Update mode keeps keys absent from defaults as user-owned."""
    path = tmp_path / "INCAR.toml"
    path.write_text('user = "keep"\n')
    write_incar_toml(
        {"a": 2},
        raw_existing={"user": "keep"},
        defaults={"a": 1},
        path=path,
    )
    assert path.read_text() == 'user = "keep"\na = 2\n'


def test_write_incar_toml_update_noop_when_unchanged(tmp_path: Path) -> None:
    """When merged content matches the file, no write or backup occurs."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 1},
        raw_existing={"a": 1},
        defaults={"a": 1},
        path=path,
    )
    assert not (tmp_path / "INCAR.toml.old").exists()
    assert path.read_text() == "a = 1\n"


def test_write_incar_toml_update_backs_up_existing(tmp_path: Path) -> None:
    """A real write backs up the previous file to INCAR.toml.old."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 2},
        raw_existing={"a": 1},
        defaults={"a": 1},
        path=path,
    )
    backup = tmp_path / "INCAR.toml.old"
    assert backup.exists()
    assert backup.read_text() == "a = 1\n"


# --- write_incar_toml_sections ---------------------------------------------


def test_write_incar_toml_sections_bootstrap_groups_and_sorts(tmp_path: Path) -> None:
    """Bootstrap groups keys into sections and sorts both levels."""
    path = tmp_path / "INCAR.toml"
    write_incar_toml_sections(
        {"z": 1, "a": 2, "m": 3},
        key_section={"z": "sec", "a": "sec", "m": None},
        path=path,
    )
    assert path.read_text() == "m = 3\n\n[sec]\na = 2\nz = 1\n"


def test_write_incar_toml_sections_update_writes_changed_and_preserves_user(
    tmp_path: Path,
) -> None:
    """Update writes non-default keys into their section and keeps user keys."""
    path = tmp_path / "INCAR.toml"
    path.write_text('keep = "user"\n')
    write_incar_toml_sections(
        {"a": 2, "keep": "user"},
        key_section={"a": "sec", "keep": None},
        raw_existing={"keep": "user"},
        defaults={"sec": {"a": 1}},
        path=path,
    )
    assert path.read_text() == 'keep = "user"\n\n[sec]\na = 2\n'


def test_write_incar_toml_sections_noop_when_unchanged(tmp_path: Path) -> None:
    """Nested raw content matching the merged flat dict triggers no write."""
    path = tmp_path / "INCAR.toml"
    path.write_text("m = 3\n\n[sec]\na = 2\nz = 1\n")
    write_incar_toml_sections(
        {"m": 3, "a": 2, "z": 1},
        key_section={"a": "sec", "z": "sec", "m": None},
        raw_existing={"m": 3, "sec": {"a": 2, "z": 1}},
        path=path,
    )
    assert not (tmp_path / "INCAR.toml.old").exists()


def test_write_incar_toml_sections_backs_up_existing(tmp_path: Path) -> None:
    """A real sectioned write backs up the previous file."""
    path = tmp_path / "INCAR.toml"
    path.write_text("old = 1\n")
    write_incar_toml_sections(
        {"old": 2},
        key_section={"old": None},
        raw_existing={"old": 1},
        path=path,
    )
    backup = tmp_path / "INCAR.toml.old"
    assert backup.exists()
    assert backup.read_text() == "old = 1\n"
