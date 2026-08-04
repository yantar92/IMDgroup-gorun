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
flat TOML; keys typically match CLI argument names with hyphens
replaced by underscores.

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


def read_incar_toml() -> dict[str, object]:
    """Read ``INCAR.toml`` and return all key-value pairs.

    Returns an empty dict when the file does not exist.
    """
    if not _INCAR_PATH.is_file():
        return {}
    with open(_INCAR_PATH, "rb") as fh:
        return tomllib.load(fh)


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
) -> None:
    """Write ``INCAR.toml`` with a sparse representation.

    In *bootstrap* mode (no file exists yet), every non-empty adapter
    default is written so the user can see all available parameters.

    In *update* mode (file exists), only keys whose value differs
    from *defaults* are written.  Keys present in *raw_existing* but
    absent from *defaults* are preserved as-is (they belong to the
    user, not the adapter).

    When the resulting content matches the file on disk, no write
    occurs.  When a write does occur, the existing file is backed up
    to ``INCAR.toml.old``.

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
    """
    if raw_existing is None:
        raw_existing = {}

    is_bootstrap = not _INCAR_PATH.is_file()

    # Build the merged dict for output.
    merged: dict[str, object] = {}

    # Preserve user-owned keys from the existing file.
    for key, val in raw_existing.items():
        if defaults is None or key not in defaults:
            merged[key] = val

    # Add adapter keys.  Bootstrap writes everything; update skips defaults.
    for key, val in values.items():
        if is_bootstrap:
            # Skip empty placeholders (e.g. model_path = "").
            if val is None or val == "":
                continue
            merged[key] = val
        elif defaults is not None and key in defaults:
            if val == defaults[key]:
                continue
            merged[key] = val
        else:
            merged[key] = val

    # For non-bootstrap, compare against the existing file content.
    if not is_bootstrap:
        if merged == raw_existing:
            return  # nothing changed

    # --- Write ---
    if _INCAR_PATH.is_file():
        backup = Path("INCAR.toml.old")
        if backup.is_file():
            backup.unlink()
        shutil.copy2(_INCAR_PATH, backup)
        print(colored("Backed up INCAR.toml → INCAR.toml.old", "yellow"))

    lines = [f"{key} = {_toml_format_value(val)}" for key, val in merged.items()]
    with open(_INCAR_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    label = "Bootstrapped" if is_bootstrap else "Updated"
    print(colored(f"{label} INCAR.toml", "green"))
