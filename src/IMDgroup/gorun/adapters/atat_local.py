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

"""ATAT-local adapter for gorun — run VASP in an ATAT-generated directory."""

from __future__ import annotations

from pathlib import Path

from IMDgroup.gorun.adapters.vasp import VaspAdapter


class AtatLocalAdapter:
    """``gorun atat-local`` backend for processing a single ATAT structure.

    Called by ``pollmach`` from inside the maps wrapper.  Always
    runs locally (``--local``).  Generates a Python heredoc that
    derives ATAT inputs, runs VASP relaxation and SCF, and writes
    an ``energy`` file.  VASP subprocesses inherit
    ``GORUN_SKIP_SETUP=1`` from the maps wrapper.
    """

    name = "atat-local"
    log_file = "atat-local.out"

    def __init__(
        self, *,
        kpoints: str,
        frac_tol: float = 0,
        max_strain: float = 0.1,
        skip_relax: bool = False,
        sublattice_cutoff: float | None = None,
    ) -> None:
        self.kpoints = kpoints
        self.frac_tol = frac_tol
        self.max_strain = max_strain
        self.skip_relax = skip_relax
        self.sublattice_cutoff = sublattice_cutoff

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        """No additional checks — pollmach calls gorun, which handles VASP."""
        pass

    # ------------------------------------------------------------------
    # Directory inspection
    # ------------------------------------------------------------------

    def is_valid_input(self, path: Path) -> bool:
        """True when str.out exists."""
        return (path / "str.out").is_file()

    def is_converged(self, path: Path) -> bool:
        """True when the energy file exists."""
        return (path / "energy").is_file()

    def has_previous_output(self, path: Path) -> bool:
        """True when the ATAT directory or error markers exist."""
        markers = [
            path / "ATAT",
            path / "energy",
            path / "error",
        ]
        return any(m.exists() for m in markers)

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
        """No preparation — ATAT input is derived inside the run command."""
        pass

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def setup_commands(self, server_config: dict) -> str:
        """Inherit VASP environment from config.

        The code executing inside the heredoc calls ``gorun vasp
        --local``, which needs the same module environment.  Since
        maps already exported ``GORUN_SKIP_SETUP=1``, this normally
        returns empty (dispatch_run skips it).  If run stand-alone
        for testing, the VASP environment is set up.
        """
        return VaspAdapter().setup_commands(server_config)

    def generate_run_commands(self, path: Path, server_config: dict | None = None) -> str:
        """Return a Python heredoc calling ``run_atat_structure``.

        Produces::

            python3 <<'PYEOF'
            from IMDgroup.gorun.gorun_atat_local import run_atat_structure
            import sys
            sys.exit(run_atat_structure(kpoints=..., frac_tol=...))
            PYEOF
        """
        sublattice = (
            repr(self.sublattice_cutoff)
            if self.sublattice_cutoff is not None
            else "None"
        )
        return (
            "python3 <<'PYEOF'\n"
            "from IMDgroup.gorun.gorun_atat_local import run_atat_structure\n"
            "import sys\n"
            f"sys.exit(run_atat_structure(\n"
            f"    kpoints={self.kpoints!r},\n"
            f"    frac_tol={self.frac_tol!r},\n"
            f"    max_strain={self.max_strain!r},\n"
            f"    skip_relax={self.skip_relax!r},\n"
            f"    sublattice_cutoff={sublattice},\n"
            "))\n"
            "PYEOF"
        )

    # ------------------------------------------------------------------
    # Backup excludes
    # ------------------------------------------------------------------

    def backup_excludes(self) -> list[str]:
        return ["WAVECAR"]
