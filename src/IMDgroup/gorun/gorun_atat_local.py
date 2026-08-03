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


"""Run VASP according to ATAT-generated structure.

Use VASP configuration from parent directory as reference.
Run (1) parent directory configuration; (2) SCF run; (3) Write energy
or error files.  (4) Mark structures that deviate too much from sublattice
with error.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
from termcolor import colored
from IMDgroup.pymatgen.cli.imdg_derive import atat as derive_atat
from IMDgroup.pymatgen.cli.imdg_derive import scf as derive_scf
from IMDgroup.pymatgen.core.structure import structure_is_valid2
import IMDgroup.pymatgen.io.atat as atat
from pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.pymatgen.core.structure import IMDStructure as Structure, structure_distance
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


# ---------------------------------------------------------------------------
# Core logic — callable from adapters
# ---------------------------------------------------------------------------

_VASP_COMMAND = ["gorun", "vasp", "--local"]


def _run_vasp(vasp_command: list[str], directory: str) -> Vasprun | bool:
    """Run *vasp_command* in *directory*.

    Returns a ``Vasprun`` on success, ``False`` on failure.
    """
    existing = IMDGVaspDir(Path(directory))
    if existing.converged:
        print(f"{directory} already contains converged output.  Not running VASP")
        return Vasprun(Path(directory) / "vasprun.xml")

    print(f"{datetime.datetime.now()} Running {vasp_command} in {directory}")
    result = subprocess.run(vasp_command, shell=False, cwd=directory, check=False)
    vaspdir = IMDGVaspDir(Path(directory))
    if (result.returncode != 0
            or (vaspdir['OSZICAR'] is not None
                and not (vaspdir.converged_electronic and vaspdir.converged_ionic))):
        Path('error').touch()
        Path('error_unconverged').touch()
        return False
    if not existing.converged_sequence:
        return False
    if not existing.converged_manual:
        return False
    if vaspdir['vasprun.xml'] is None:
        return False
    return vaspdir['vasprun.xml']


def run_atat_structure(
    *,
    kpoints: str,
    frac_tol: float = 0,
    max_strain: float = 0.1,
    skip_relax: bool = False,
    sublattice_cutoff: float | None = None,
    vasp_command: list[str] | None = None,
) -> int:
    """Process a single ATAT structure in the current directory.

    Called by the ``atat-local`` adapter's heredoc.  Derives ATAT
    and SCF inputs, runs VASP, validates, and writes an ``energy``
    file.

    Returns 0 on success, 1 on failure.
    """
    if vasp_command is None:
        vasp_command = _VASP_COMMAND

    # -- derive ATAT input --
    if Path('ATAT').is_dir():
        print(colored("ATAT already exists.  Not modifying", "yellow"))
    else:
        args = Namespace(
            kpoints=kpoints,
            atat_structure="str.out",
            input_directory="../",
            inherit_prev_incarpy=True,
        )
        inputset_data = derive_atat(args)
        assert len(inputset_data['inputsets']) == 1
        inputset = inputset_data['inputsets'][0]

        if not structure_is_valid2(inputset.structure, frac_tol=frac_tol):
            Path('error').touch()
            Path('error_atoms_too_close').touch()
            print(colored("str.out has atoms too close to each other", "red"))
            return 1

        kpts = np.array(inputset.kpoints.kpts[0])
        if (np.any(kpts <= 3) and not np.all(kpts <= 3)
                and kpts[kpts <= 3].size > 1):
            Path('error').touch()
            Path('error_kpoints_dim_sparse').touch()
            print(colored(
                f"KPOINTS has too few points along one of the axes: {kpts}", "red"
            ))
            return 1

        inputset.write_input(output_dir="ATAT")

    # -- relaxation --
    if skip_relax:
        print(colored("--skip-relax passed.  Not running relaxation in ./ATAT", "yellow"))
    else:
        run = _run_vasp(vasp_command, "ATAT")
        if not run:
            return 1
        vaspdir = IMDGVaspDir("ATAT")
        str_before = vaspdir.initial_structure
        str_after = vaspdir.structure
        try:
            if not atat.check_volume_distortion(str_before, str_after, max_strain):
                print(colored(f"POSCAR->CONTCAR strain exceeds {max_strain*100}%", "red"))
                Path('error').touch()
                Path('error_strain').touch()
                return 1
            sublattice = Structure.from_file('str.out')
            if not atat.check_sublattice_flip(str_before, str_after, sublattice):
                print(colored(
                    "POSCAR&CONTCAR flipped sublattice configuration.", "yellow"))
                sublattice2 = atat.fit_sublattice_to_structure(sublattice, str_after)
                if not Path('str.out.old').is_file():
                    Path('str.out').rename('str.out.old')
                    sublattice2.to_file('str.out', fmt='atat')
                    print(colored("Updating str.out", "yellow"))
                else:
                    print(colored("str.out.old exists.  Not overwriting", "yellow"))
                sublattice = sublattice2
            str_after_norm = str_after.copy()
            str_after_norm.lattice = sublattice.lattice
            dist = structure_distance(
                str_after_norm, sublattice,
                match_first=True, match_species=False,
            )
            Path("sublattice_deviation").write_text(f"{dist:.4f}\n", encoding='utf-8')
            if sublattice_cutoff is not None and dist >= sublattice_cutoff:
                print(colored(
                    f"Sublattice deviation {dist:.2f} >= cutoff "
                    f"{sublattice_cutoff:.2f}.  Marking as error", "red"))
                Path('error').touch()
                Path('error_sublattice').touch()
        except Exception as e:
            print(f"Caught exception while comparing str.out and VASP output: {e}")
            print("Continuing anyway, but marking with error")
            Path('error').touch()
            Path('error_structure').touch()

    # -- SCF --
    if Path('ATAT.SCF').is_dir():
        print(colored("ATAT.SCF already exists.  Not modifying", "yellow"))
    else:
        args = Namespace(input_directory="ATAT")
        inputset_data = derive_scf(args)
        assert len(inputset_data['inputsets']) == 1
        inputset = inputset_data['inputsets'][0]
        inputset.write_input(output_dir="ATAT.SCF")

    run = _run_vasp(vasp_command, "ATAT.SCF")
    if not run:
        return 1

    Path('energy').write_text(f"{float(run.final_energy)}\n")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point (backward-compatible thin wrapper)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Backward-compat entry point; delegates to the ``atat-local`` adapter.

    Called by ``gorun-atat-local`` script.  Forwards to
    ``gorun.main(["atat-local", "--local"] + argv)``.
    """
    from IMDgroup.gorun.gorun import main as gorun_main
    if argv is None:
        argv = sys.argv[1:]
    return gorun_main(["atat-local", "--local"] + argv)
