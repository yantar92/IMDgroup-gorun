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


def maybe_regenerate(
    path: Path,
    filename: str,
    keep: set[str] | None,
    regenerate: Callable[[], None],
    validate: Callable[[], bool] | None = None,
) -> None:
    """Regenerate *filename* inside *path* unless it is being kept.

    Decision table:

    =============  ===========  ====================================
    File exists?   In *keep*?   Action
    =============  ===========  ====================================
    No             —            Call *regenerate*.
    Yes            No           Call *regenerate* with log.
    Yes            Yes          1. If *validate* is given and fails,
                                  call *regenerate* with log.
                                2. Otherwise, skip with log.
    =============  ===========  ====================================

    Parameters
    ----------
    path:
        Directory containing the file.
    filename:
        Name of the file to (possibly) regenerate.
    keep:
        Set of filenames that should not be regenerated.
        ``None`` is treated as an empty set.
    regenerate:
        Callable that generates the file from scratch.
    validate:
        Optional callable that checks whether an existing file is
        usable.  Should return ``True`` when the file is valid.
        Called only when the file exists and is in *keep*.
    """
    if keep is None:
        keep = set()

    filepath = path / filename
    if filepath.exists():
        if filename in keep:
            if validate is None or validate():
                print(f"Keeping existing {filename}")
                return
            print(f"Existing {filename} is invalid; regenerating")
        else:
            print(f"Regenerating {filename}")
    elif filename in keep:
        print(
            f"--keep {filename} requested but {filename} does not exist.  "
            "Generating."
        )
    regenerate()
