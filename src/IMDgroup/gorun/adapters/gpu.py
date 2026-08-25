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

"""Generic GPU adapter for gorun.

Provides ``GpuAdapter``, a base class for GPU-accelerated backends
that only require a ``RUNFILE.sh`` or ``RUNFILE.py`` in the working
directory.  Derived adapters (e.g. MACE) can override
``generate_run_commands`` to add canned fallback stages after the
RUNFILE check.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Wrapper helper
# ---------------------------------------------------------------------------

def _wrap_command(command: str) -> str:
    """Wrap *command* in a heredoc passed to ``GORUN_WRAPPER``.

    ``${GORUN_WRAPPER:-bash}`` ensures the heredoc is always piped to
    a shell: the configured wrapper when ``GORUN_WRAPPER`` is set, or
    plain ``bash`` otherwise.  This avoids duplicating *command* in
    an ``if/else`` branch.

    The heredoc is quoted (``<<'EOF'``) so the outer shell does not
    expand ``${GORUN_INNER_SETUP}`` before passing it through; the
    inner shell expands it.  Both variables are expected to be exported
    by the config's ``setup`` commands.
    """
    return (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        f"{command}\n"
        "EOF"
    )


# ---------------------------------------------------------------------------
# GpuAdapter
# ---------------------------------------------------------------------------


class GpuAdapter:
    """Generic GPU backend for gorun.

    Requires only a ``RUNFILE.sh`` or ``RUNFILE.py`` in the working
    directory to define the computation.  GPU environment setup is
    read from the server config under ``[<name>]`` → ``setup``.

    Derived classes override ``name``, ``DEFAULTS``, ``REQUIRED``,
    ``validate_environment``, ``is_valid_input``, ``is_converged``,
    ``has_previous_output``, and ``prepare_inputs`` for their
    specific needs.  ``generate_run_commands`` can be overridden to
    add canned fallback stages after the RUNFILE check (see
    ``MaceMultiheadFinetuneAdapter`` for an example).
    """

    name = "gpu"

    def __init__(self, *, script_name: str | None = None) -> None:
        """*script_name* overrides the default RUNFILE.sh/py lookup.

        When set, ``generate_run_commands`` looks for *script_name*
        instead of RUNFILE.sh / RUNFILE.py.  The runner is inferred
        from the extension: ``.py`` → ``python -u``, anything else
        (including ``.sh`` or no extension) → ``bash``.
        """
        self._script_name = script_name

    #: Hardcoded defaults for shared GPU parameters.
    DEFAULTS: dict[str, object] = {
        "device": "cuda",
        "default_dtype": "float64",
    }

    #: Parameters that must be non-empty for a valid run.
    REQUIRED: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        """No default checks.  Override in derived classes."""
        pass

    def setup_commands(self, server_config: dict) -> str:
        """Read GPU environment setup from the server config.

        Looks for ``server_config[self.name]["setup"]`` — a raw shell
        string that typically exports ``GORUN_WRAPPER`` and
        ``GORUN_INNER_SETUP``, along with PyTorch flags, cache
        directories, ``OMP_NUM_THREADS``, etc.
        """
        return server_config.get(self.name, {}).get("setup", "")

    # ------------------------------------------------------------------
    # Directory inspection
    # ------------------------------------------------------------------

    def is_valid_input(self, path: Path) -> bool:
        """True when the configured script or RUNFILE.sh/py exists."""
        if self._script_name:
            return (path / self._script_name).is_file()
        return (path / "RUNFILE.sh").is_file() or (path / "RUNFILE.py").is_file()

    def is_converged(self, path: Path) -> bool:
        """Must be overridden.  Returns ``False`` by default."""
        return False

    def has_previous_output(self, path: Path) -> bool:
        """Must be overridden.  Returns ``False`` by default."""
        return False

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        path: Path, *,
        keep: set[str] | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """No default preparation.  Override in derived classes."""
        pass

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_run_commands(self, path: Path, server_config: dict | None = None) -> str:
        """Return the RUNFILE command, or raise if none is found.

        When ``_script_name`` is set on the instance, that file is used
        instead of RUNFILE.sh / RUNFILE.py.  ``.py`` files are run with
        ``python -u``; everything else with ``bash``.

        All commands are wrapped through ``_wrap_command`` for
        ``GORUN_WRAPPER`` / ``GORUN_INNER_SETUP`` support.

        Raises ``FileNotFoundError`` when no script is found.
        """
        if self._script_name:
            script = path / self._script_name
            if not script.is_file():
                raise FileNotFoundError(
                    f"{self._script_name} not found in {path}."
                )
            runner = (
                "python -u"
                if self._script_name.endswith(".py")
                else "bash"
            )
            return _wrap_command(f"{runner} {script}")

        runfile_sh = path / "RUNFILE.sh"
        runfile_py = path / "RUNFILE.py"

        if runfile_sh.is_file():
            return _wrap_command(f"bash {runfile_sh}")

        if runfile_py.is_file():
            return _wrap_command(f"python -u {runfile_py}")

        raise FileNotFoundError(
            "No RUNFILE.sh or RUNFILE.py found in the working directory.  "
            "Place one to define the computation."
        )

    # ------------------------------------------------------------------
    # Backup excludes
    # ------------------------------------------------------------------

    def backup_excludes(self) -> list[str]:
        return []
