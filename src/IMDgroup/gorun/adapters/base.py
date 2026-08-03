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

"""Protocol that every software backend must implement for gorun."""

from __future__ import annotations

from typing import Protocol
from pathlib import Path


class SoftwareAdapter(Protocol):
    """Interface for a computational backend supported by gorun.

    Each backend (VASP, MACE, etc.) provides an implementation of
    this protocol.  The gorun dispatch pipeline calls methods in the
    order defined by `dispatch_run`.
    """

    # -- identity --
    name: str
    """Unique short name, e.g. ``"vasp"``, ``"mace"``.

    Used for subcommand dispatch and config namespace lookup."""

    # -- environment --
    def validate_environment(self) -> None:
        """Check that everything needed to run is available.

        Raise an exception if required environment variables, files,
        or commands are missing.  Called before any other method.
        """
        ...

    # -- directory inspection --
    def is_valid_input(self, path: Path) -> bool:
        """Return True when *path* contains the input files needed to start."""
        ...

    def is_converged(self, path: Path) -> bool:
        """Return True when the computation in *path* has finished successfully."""
        ...

    def has_previous_output(self, path: Path) -> bool:
        """Return True when *path* contains output from a previous run.

        Controls whether the pipeline creates a backup before
        re-running.
        """
        ...

    # -- preparation --
    def prepare_inputs(
        self,
        path: Path, *,
        keep: set[str] | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """Make *path* ready to run.

        Called after backup.  The adapter should:
        - clean or regenerate input files;
        - generate files missing from the directory;
        - skip files listed in *keep* if they already exist.

        *force* propagates the ``--force`` flag from the CLI.
        *kwargs* carry adapter-specific flags.
        """
        ...

    # -- script generation --
    def setup_commands(self, server_config: dict) -> str:
        """Return shell commands to configure the environment.

        *server_config* is the full server section from the TOML
        config (e.g. ``config["lumi"]``).  The adapter extracts its
        own subsection using ``self.name``, falling back to flat keys
        for backward compatibility.
        """
        ...

    def generate_run_commands(self, path: Path) -> str:
        """Return the bash body that executes the computation.

        The pipeline wraps this inside RUNNING-file management and
        (for Slurm) sbatch headers.
        """
        ...

    # -- backup & cleanup --
    def backup_excludes(self) -> list[str]:
        """Return rsync ``--exclude`` patterns for the backup step.

        Return patterns like ``["*.tar.gz", "gorun_*", "WAVECAR"]``.
        """
        ...
