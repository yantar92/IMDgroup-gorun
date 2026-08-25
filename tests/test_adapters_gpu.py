"""Tests for IMDgroup.gorun.adapters.gpu (phase 1: _wrap_command only)."""

from __future__ import annotations

from IMDgroup.gorun.adapters.gpu import _wrap_command


def test_wrap_command_exact_output() -> None:
    """A simple command is wrapped in the heredoc template verbatim."""
    assert _wrap_command("echo hi") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "echo hi\n"
        "EOF"
    )


def test_wrap_command_multiline_command() -> None:
    """A multi-line command is placed between the setup line and EOF."""
    assert _wrap_command("line1\nline2") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "line1\nline2\n"
        "EOF"
    )


def test_wrap_command_empty_command() -> None:
    """An empty command still emits the wrapper and setup lines."""
    assert _wrap_command("") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "\n"
        "EOF"
    )
