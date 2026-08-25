# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""INCAR.toml: shared parameter-file I/O for gorun adapters.

Analogous to VASP's INCAR file.  Each adapter can read parameter
defaults from ``INCAR.toml`` in the working directory.  The file uses
TOML; keys typically match CLI argument names with hyphens replaced
by underscores.

Adapters may use flat key-value pairs or group parameters into
``[section]`` tables (see :func:`write_incar_toml_sections`).

This module is adapter-agnostic -- it only reads and writes the file.
Parameter defaults, validation, and the mapping from TOML keys to
adapter attributes are the responsibility of each adapter.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from termcolor import colored

_INCAR_PATH = Path("INCAR.toml")


def read_incar_toml(path: Path = _INCAR_PATH) -> dict[str, object]:
    """Read *path* and return all key-value pairs.

    Returns an empty dict when the file does not exist.
    """
    if not path.is_file():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _flatten_toml(raw: dict[str, object]) -> dict[str, object]:
    """Flatten a possibly-nested TOML dict.

    Values that are themselves dicts (TOML ``[section]`` tables)
    have their keys merged into the top level.  Non-dict values
    are passed through unchanged.  Section names are discarded.
    """
    result: dict[str, object] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            result.update(val)
        else:
            result[key] = val
    return result


_NO_DEFAULT = object()


def _resolve_default_value(
    defaults: dict[str, object],
    key_section: dict[str, str | None],
    key: str,
) -> object:
    """Return the default value for *key* from structured *defaults*.

    When *key* belongs to a section (via *key_section*), the
    default is looked up inside that section's dict.  Otherwise
    the top-level *defaults* dict is checked.

    Returns :data:`_NO_DEFAULT` when no default is defined for
    *key*.
    """
    section = key_section.get(key)
    if section is not None and section in defaults:
        section_dict = defaults[section]
        if isinstance(section_dict, dict) and key in section_dict:
            return section_dict[key]
    if key in defaults and not isinstance(defaults[key], dict):
        return defaults[key]
    return _NO_DEFAULT


def _key_in_defaults(
    defaults: dict[str, object],
    key_section: dict[str, str | None],
    key: str,
) -> bool:
    """Return True when *key* has a default in structured *defaults*."""
    return _resolve_default_value(defaults, key_section, key) is not _NO_DEFAULT


def _toml_format_value(value: object) -> str:
    """Format a Python value for TOML output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return f'"{value}"'


def write_incar_toml(
    values: dict[str, object],
    *,
    raw_existing: dict[str, object] | None = None,
    defaults: dict[str, object] | None = None,
    always_include: set[str] | None = None,
    path: Path = _INCAR_PATH,
) -> None:
    """Write ``INCAR.toml`` (at *path*) with a sparse representation.

    In *bootstrap* mode (no file exists yet), every adapter default
    is written so the user can see all available parameters.
    ``None`` values are skipped (TOML has no ``None`` literal).
    Keys in *always_include* are written even when ``None``.

    In *update* mode (file exists), keys whose value differs from
    *defaults* are written.  Keys that match defaults are still
    written when they were already present in *raw_existing* (the
    user put them there explicitly).  Keys present in *raw_existing*
    but absent from *defaults* are preserved as-is (they belong to
    the user, not the adapter).

    When the resulting content matches the file on disk, no write
    occurs.  When a write does occur, the existing file is backed up
    to ``INCAR.toml.old`` next to *path*.

    Parameters
    ----------
    values:
        Merged parameter values (CLI ∪ TOML ∪ adapter defaults).
        In bootstrap mode these are the adapter defaults.
    raw_existing:
        The previously-read ``INCAR.toml`` content, used to preserve
        non-adapter keys.  Pass ``None`` or an empty dict when no
        file exists yet (bootstrap mode).
    defaults:
        Hardcoded adapter defaults.  Keys present here are considered
        adapter-owned; only non-default values are written (in update
        mode).  Pass ``None`` to write all *values* as-is.
    always_include:
        Keys that must be written even when their value equals the
        default or is empty.  Used in bootstrap mode to surface
        required-but-unset parameters.
    path:
        Destination file.  Defaults to ``INCAR.toml`` in the current
        working directory.
    """
    if raw_existing is None:
        raw_existing = {}

    is_bootstrap = not path.is_file()

    # Build the merged dict for output.
    merged: dict[str, object] = {}

    # Preserve user-owned keys from the existing file.
    for key, val in raw_existing.items():
        if defaults is None or key not in defaults:
            merged[key] = val

    # Add adapter keys.  Bootstrap writes everything; update skips defaults.
    for key, val in values.items():
        if key in (always_include or set()):
            merged[key] = val
        elif is_bootstrap:
            if val is None:
                continue
            merged[key] = val
        elif defaults is not None and key in defaults:
            if val == defaults[key] and key not in raw_existing:
                continue
            merged[key] = val
        else:
            merged[key] = val

    # For non-bootstrap, compare against the existing file content.
    if not is_bootstrap:
        if merged == raw_existing:
            return  # nothing changed

    # --- Write ---
    if path.is_file():
        backup = path.with_name(f"{path.name}.old")
        if backup.is_file():
            backup.unlink()
        shutil.copy2(path, backup)
        print(colored(f"Backed up {path.name} → {backup.name}", "yellow"))

    lines = [f"{key} = {_toml_format_value(val)}" for key, val in merged.items()]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    label = "Bootstrapped" if is_bootstrap else "Updated"
    print(colored(f"{label} INCAR.toml", "green"))


def write_incar_toml_sections(
    values: dict[str, object],
    *,
    key_section: dict[str, str | None],
    raw_existing: dict[str, object] | None = None,
    defaults: dict[str, object] | None = None,
    always_include: set[str] | None = None,
    path: Path = _INCAR_PATH,
) -> None:
    """Write ``INCAR.toml`` (at *path*) with ``[section]`` grouping.

    Like :func:`write_incar_toml` but *values* are grouped into TOML
    ``[section]`` tables based on *key_section*, which maps each
    adapter key to its section name (or ``None`` for top-level).

    *raw_existing* may contain nested dicts from a prior
    ``tomllib.load``; they are flattened for comparison.  Keys not
    in *key_section* are written at the top level.

    Parameters
    ----------
    values:
        Flat dict of merged parameter values (CLI ∪ TOML ∪ defaults).
    key_section:
        Mapping from adapter key to TOML section name or ``None``
        for top-level placement.
    raw_existing:
        The previously-read ``INCAR.toml`` content (may be nested).
    defaults:
        Hardcoded adapter defaults, possibly structured with
        ``[section]`` sub-dicts matching TOML sections.  Only
        non-default values are written in update mode.  Section
        membership is resolved via *key_section*.
    always_include:
        Keys written even when matching defaults or empty.
    path:
        Destination file.  Defaults to ``INCAR.toml`` in the current
        working directory.
    """
    if raw_existing is None:
        raw_existing = {}

    flat_raw = _flatten_toml(raw_existing)

    is_bootstrap = not path.is_file()

    # Build the flattened merged dict (same logic as write_incar_toml).
    merged: dict[str, object] = {}

    # Preserve user-owned keys from the existing file.
    for key, val in flat_raw.items():
        if defaults is None or not _key_in_defaults(defaults, key_section, key):
            merged[key] = val

    # Add adapter keys.
    for key, val in values.items():
        if key in (always_include or set()):
            merged[key] = val
        elif is_bootstrap:
            if val is None:
                continue
            merged[key] = val
        elif defaults is not None:
            default_val = _resolve_default_value(defaults, key_section, key)
            if default_val is not _NO_DEFAULT:
                if val == default_val and key not in flat_raw:
                    continue
            merged[key] = val
        else:
            merged[key] = val

    # For non-bootstrap, compare against the existing file content.
    if not is_bootstrap:
        if merged == flat_raw:
            return  # nothing changed

    # --- Group by section ---
    sections: dict[str, dict[str, object]] = {}
    top_level: dict[str, object] = {}

    for key, val in merged.items():
        section = key_section.get(key)
        if section:
            sections.setdefault(section, {})[key] = val
        else:
            top_level[key] = val

    # --- Write ---
    if path.is_file():
        backup = path.with_name(f"{path.name}.old")
        if backup.is_file():
            backup.unlink()
        shutil.copy2(path, backup)
        print(colored(f"Backed up {path.name} → {backup.name}", "yellow"))

    block: list[str] = []
    for key, val in top_level.items():
        block.append(f"{key} = {_toml_format_value(val)}")
    for section_name in sorted(sections):
        if block:
            block.append("")
        block.append(f"[{section_name}]")
        for key in sorted(sections[section_name]):
            block.append(f"{key} = {_toml_format_value(sections[section_name][key])}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(block) + "\n")

    label = "Bootstrapped" if is_bootstrap else "Updated"
    print(colored(f"{label} INCAR.toml", "green"))
