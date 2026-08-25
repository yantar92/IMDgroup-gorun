"""Tests for IMDgroup.gorun.core.files."""

from __future__ import annotations

import os
from pathlib import Path

from IMDgroup.gorun.core.files import (
    _any_output_older,
    _newest_source_mtime,
    maybe_regenerate,
)


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


class _Recorder:
    def __init__(self) -> None:
        self.called = False

    def __call__(self) -> None:
        self.called = True


def test_newest_source_mtime_empty() -> None:
    """An empty source list has no mtime, so the result is 0.0."""
    assert _newest_source_mtime([]) == 0.0


def test_newest_source_mtime_returns_max_and_ignores_missing(tmp_path: Path) -> None:
    """The max mtime is returned and non-existent sources are skipped."""
    newer = tmp_path / "newer"
    older = tmp_path / "older"
    missing = tmp_path / "missing"
    newer.touch()
    older.touch()
    _set_mtime(newer, 200.0)
    _set_mtime(older, 100.0)
    assert _newest_source_mtime([older, missing, newer]) == 200.0


def test_any_output_older_missing(tmp_path: Path) -> None:
    """A missing output file counts as older than any allowed timestamp."""
    assert _any_output_older(tmp_path, ["out"], 0.0) is True


def test_any_output_older_compares_mtime(tmp_path: Path) -> None:
    """Outputs older than the threshold trigger True; newer ones do not."""
    out = tmp_path / "out"
    out.touch()
    _set_mtime(out, 100.0)
    assert _any_output_older(tmp_path, ["out"], 200.0) is True
    assert _any_output_older(tmp_path, ["out"], 50.0) is False


def test_maybe_regenerate_missing_file(tmp_path: Path) -> None:
    """A missing output file is regenerated."""
    regen = _Recorder()
    assert maybe_regenerate(tmp_path, "POTCAR", None, regen) is True
    assert regen.called is True


def test_maybe_regenerate_not_kept_newer_than_sources(tmp_path: Path) -> None:
    """An unkept output newer than its sources is skipped."""
    out = tmp_path / "POTCAR"
    src = tmp_path / "POSCAR"
    out.touch()
    src.touch()
    _set_mtime(out, 200.0)
    _set_mtime(src, 100.0)
    regen = _Recorder()
    assert (
        maybe_regenerate(tmp_path, "POTCAR", set(), regen, older_than=[src]) is False
    )
    assert regen.called is False


def test_maybe_regenerate_not_kept_older_than_sources(tmp_path: Path) -> None:
    """An unkept output older than its sources is regenerated."""
    out = tmp_path / "POTCAR"
    src = tmp_path / "POSCAR"
    out.touch()
    src.touch()
    _set_mtime(out, 100.0)
    _set_mtime(src, 200.0)
    regen = _Recorder()
    assert (
        maybe_regenerate(tmp_path, "POTCAR", set(), regen, older_than=[src]) is True
    )
    assert regen.called is True


def test_maybe_regenerate_kept_validate_passes(tmp_path: Path) -> None:
    """A kept file that passes validation is not regenerated."""
    (tmp_path / "POTCAR").touch()
    regen = _Recorder()
    assert (
        maybe_regenerate(tmp_path, "POTCAR", {"POTCAR"}, regen, validate=lambda: True)
        is False
    )
    assert regen.called is False


def test_maybe_regenerate_kept_validate_fails(tmp_path: Path) -> None:
    """A kept file that fails validation is regenerated."""
    (tmp_path / "POTCAR").touch()
    regen = _Recorder()
    assert (
        maybe_regenerate(tmp_path, "POTCAR", {"POTCAR"}, regen, validate=lambda: False)
        is True
    )
    assert regen.called is True


def test_maybe_regenerate_list_collective_skip(tmp_path: Path) -> None:
    """A list output group is skipped when every file exists and is kept."""
    (tmp_path / "a").touch()
    (tmp_path / "b").touch()
    regen = _Recorder()
    assert maybe_regenerate(tmp_path, ["a", "b"], {"a", "b"}, regen) is False
    assert regen.called is False


def test_maybe_regenerate_list_collective_missing(tmp_path: Path) -> None:
    """A list output group is regenerated when any member is missing."""
    (tmp_path / "a").touch()
    regen = _Recorder()
    assert maybe_regenerate(tmp_path, ["a", "b"], {"a", "b"}, regen) is True
    assert regen.called is True
