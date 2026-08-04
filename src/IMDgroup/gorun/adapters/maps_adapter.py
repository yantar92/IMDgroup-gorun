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

"""Maps adapter for gorun — ATAT cluster expansion launcher."""

from __future__ import annotations

from pathlib import Path
import subprocess
import shutil

from IMDgroup.gorun.adapters.base import SoftwareAdapter
from IMDgroup.gorun.adapters.vasp import VaspAdapter


class MapsAdapter:
    """``gorun maps`` backend for ATAT cluster expansion with maps+pollmach."""

    name = "maps"
    log_file = "maps.out"

    def __init__(
        self, *,
        kpoints: str,
        frac_tol: float = 0.5,
        max_strain: float = 0.1,
        skip_relax: bool = False,
        sublattice_cutoff: float | None = None,
        maps_args: list[str] | None = None,
    ) -> None:
        self.kpoints = kpoints
        self.frac_tol = frac_tol
        self.max_strain = max_strain
        self.skip_relax = skip_relax
        self.sublattice_cutoff = sublattice_cutoff
        self.maps_args = maps_args or []

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        """Require maps, pollmach, and a working VASP environment."""
        import shutil as _shutil
        missing = []
        for cmd in ("maps", "pollmach"):
            if _shutil.which(cmd) is None:
                missing.append(cmd)
        if missing:
            raise OSError(
                f"Required commands not found: {', '.join(missing)}"
            )
        # Defer VASP checks to VaspAdapter.
        VaspAdapter().validate_environment()

    # ------------------------------------------------------------------
    # Directory inspection
    # ------------------------------------------------------------------

    def is_valid_input(self, path: Path) -> bool:
        """True when lat.in and a valid prototype VASP input exist."""
        if not (path / "lat.in").is_file():
            return False
        # Validate prototype VASP input
        from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
        try:
            test_dir = path / ".__test"
            test_set = IMDDerivedInputSet(directory=str(path))
            test_set.write_input(test_dir)
            shutil.rmtree(test_dir)
            return True
        except Exception:
            return False

    def is_converged(self, path: Path) -> bool:
        """Always returns False — maps runs until killed or exhausted."""
        return False

    def has_previous_output(self, path: Path) -> bool:
        """True when maps lock files or ATAT output directories exist."""
        markers = [
            path / "maps.log",
            path / "pollmach.log",
        ]
        return any(m.is_file() for m in markers)

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
        """Clean lock files from previous runs."""
        (path / "pollmach_is_running").unlink(missing_ok=True)
        (path / "maps_is_running").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def setup_commands(self, server_config: dict) -> str:
        """Inherit VASP environment from config.

        The maps wrapper needs the same module environment as VASP
        (maps and pollmach run VASP as subprocesses).  The VASP
        module loads are done here; child gorun processes inherit
        them via ``GORUN_SKIP_SETUP=1``.
        """
        return VaspAdapter().setup_commands(server_config)

    def generate_run_commands(self, path: Path) -> str:
        """Return the maps+pollmach wrapper script.

        Exports ``GORUN_SKIP_SETUP=1`` so every child gorun process
        skips re-running module loads.
        """
        maps_cmd = "maps " + " ".join(self.maps_args)

        # Build pollmach args
        pollmach_parts = [
            "pollmach",
            "gorun", "atat-local", "--local",
            f"--kpoints={self.kpoints}",
            f"--frac-tol={self.frac_tol}",
            f"--max-strain={self.max_strain}",
        ]
        if self.skip_relax:
            pollmach_parts.append("--skip-relax")
        if self.sublattice_cutoff is not None:
            pollmach_parts.append(f"--sublattice-cutoff={self.sublattice_cutoff}")

        pollmach_cmd = " ".join(pollmach_parts)

        return (
            "export GORUN_SKIP_SETUP=1\n\n"
            f"{maps_cmd} &\n"
            "sleep 5\n"
            f"{pollmach_cmd}"
        )

    # ------------------------------------------------------------------
    # Backup excludes
    # ------------------------------------------------------------------

    def backup_excludes(self) -> list[str]:
        return [
            "WAVECAR",
            "__*",
        ]
