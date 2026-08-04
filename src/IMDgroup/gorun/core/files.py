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

"""File-manipulation helpers used across adapters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _newest_source_mtime(sources: list[Path]) -> float:
    """Return the maximum mtime among *sources*, or 0 if none exist."""
    existing = [src.stat().st_mtime for src in sources if src.is_file()]
    return max(existing) if existing else 0.0


def _any_output_older(
    path: Path,
    filenames: list[str],
    oldest_allowed: float,
) -> bool:
    """Return True when any *filenames* file in *path* is missing or
    older than *oldest_allowed*."""
    for fname in filenames:
        fp = path / fname
        if not fp.is_file():
            return True
        if fp.stat().st_mtime < oldest_allowed:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_regenerate(
    path: Path,
    filename: str | list[str],
    keep: set[str] | None,
    regenerate: Callable[[], None],
    validate: Callable[[], bool] | None = None,
    older_than: list[Path] | None = None,
) -> bool:
    """Regenerate *filename* inside *path* unless it is being kept.

    When *filename* is a list, the decision is made collectively:
    *regenerate* is called unless **all** files exist and are being
    kept (or are all newer than *older_than*).  The log message
    mentions the first filename in the list.

    Decision table (single-file case):

    ============  ===========  ======================================
    File exists?  In *keep*?   Action
    ============  ===========  ======================================
    No            —            Call *regenerate*.
    Yes           No           1. If *older_than* is given and all
                                outputs are newer than every source,
                                skip with log.
                              2. Otherwise, call *regenerate* with
                                log.
    Yes           Yes          1. If *validate* is given and fails,
                                call *regenerate* with log.
                              2. Otherwise, skip with log.
    ============  ===========  ======================================

    Parameters
    ----------
    path:
        Directory containing the file(s).
    filename:
        Name of the file to (possibly) regenerate, or a list of
        filenames treated as a single output group.
    keep:
        Set of filenames that should not be regenerated.
        ``None`` is treated as an empty set.
    regenerate:
        Callable that generates the file(s) from scratch.
    validate:
        Optional callable that checks whether an existing file is
        usable.  Should return ``True`` when the file is valid.
        Called only when the file exists and is in *keep*.
    older_than:
        Optional list of source files.  When given and the output
        file(s) exist but are **not** in *keep*, regeneration is
        skipped if every output is newer than every source.  Sources
        that do not exist are ignored (treated as epoch 0).

    Returns
    -------
    bool
        ``True`` when *regenerate* was called, ``False`` when it was
        skipped.
    """
    if keep is None:
        keep = set()
    if isinstance(filename, str):
        filenames = [filename]
    else:
        filenames = filename
    label = filenames[0]

    # -- all in --keep and exist? ---------------------------------------
    if filenames and all(
        (f in keep) and (path / f).is_file() for f in filenames
    ):
        filepath = path / filenames[0]
        if validate is not None and not validate():
            print(f"Existing {label} is invalid; regenerating")
            regenerate()
            return True
        print(f"Keeping existing {label} (--keep)")
        return False

    # -- any in --keep but missing? -------------------------------------
    kept_missing = [
        f for f in filenames
        if f in keep and not (path / f).is_file()
    ]
    if kept_missing:
        print(
            f"--keep requested for {', '.join(kept_missing)} "
            "but file(s) do not exist.  Generating."
        )
        regenerate()
        return True

    # -- not kept → use older_than / fallback to regenerate --------------
    if older_than is not None:
        newest_source = _newest_source_mtime(older_than)
        if newest_source > 0 and not _any_output_older(
            path, filenames, newest_source
        ):
            print(f"Keeping existing {label} (newer than sources)")
            return False

    print(f"Regenerating {label}")
    regenerate()
    return True
