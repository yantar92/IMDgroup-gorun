# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-gorun package
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

import datetime
import argparse
import subprocess
from pathlib import Path
from xml.etree.ElementTree import ParseError
import numpy as np
from termcolor import colored
from IMDgroup.pymatgen.cli.imdg_derive import atat as derive_atat
from IMDgroup.pymatgen.cli.imdg_derive import scf as derive_scf
from IMDgroup.pymatgen.core.structure import structure_is_valid2
import IMDgroup.pymatgen.io.atat as atat
from pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.pymatgen.core.structure import IMDStructure as Structure, structure_distance
from IMDgroup.gorun.cleanVASP import directory_converged_p
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


def get_args():
    parser = argparse.ArgumentParser(
        description="Run VASP in current dir, according to str.out and parent dir.")

    parser.add_argument(
        "--kpoints",
        required=True,
        help="Kpoint density")
    parser.add_argument(
        "--frac_tol",
        default=0,
        type=float,
        help="Distance tolerance to reject structure (default: 0 = no rejections)")
    parser.add_argument(
        "--max_strain",
        default=0.1,
        type=float,
        help="Maximum strain allowed (default: 0.1)")
    parser.add_argument(
        "--skip_relax",
        help="Whether to skip relaxation run",
        action="store_true")
    parser.add_argument(
        "--sublattice_cutoff",
        help="Maximum allowed sublattice deviation",
        type=float,
        default=None
    )
    parser.add_argument(
        "vasp_command",
        help="VASP command to run",
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def run_vasp(vasp_command, directory):
    """Run VASP_COMMAND in DIRECTORY.
    Return Vasprun object if VASP succeeds and converges and False
    otherwise.
    """
    if directory_converged_p(directory):
        print(f"{directory} already contains converged output. Not running VASP")
        return Vasprun(Path(directory) / "vasprun.xml")

    print(f"{datetime.datetime.now()} Running {vasp_command} in {directory}")
    result = subprocess.run(
        vasp_command,
        shell=False,
        cwd=directory,
        check=False,
    )
    try:
        run = Vasprun(Path(directory) / "vasprun.xml")
        vaspdir = IMDGVaspDir(Path(directory))
    except (ValueError, ParseError):
        run = 'failed'
    except FileNotFoundError:
        run = None
    if result.returncode != 0 or run == 'failed' or (run is not None and not vaspdir.converged):
        Path('error').touch()
        Path('error_unconverged').touch()
        return False
    if run is None:
        return False
    return run


def main(args=None):
    if args is None:
        args = get_args()
    if Path('ATAT').is_dir():
        print(colored("ATAT already exists.  Not modifying", "yellow"))
    else:
        # Generate VASP input
        args.atat_structure = "str.out"
        args.input_directory = "../"
        inputset_data = derive_atat(args)
        assert len(inputset_data['inputsets']) == 1
        inputset = inputset_data['inputsets'][0]
        if not structure_is_valid2(inputset.structure, frac_tol=args.frac_tol):
            Path('error').touch()
            Path('error_atoms_too_close').touch()
            print(colored("str.out has atoms too close to each other", "red"))
            return 1
        kpoints = inputset.kpoints
        assert kpoints is not None
        kpoints = np.array(kpoints.kpts[0])
        # We had cases like KPOINTS 2x2x9 (denity=2500)
        # that distorted energy outputs due to small number (2) of kpoints
        # along the individual axis.  Filter out such cases as they
        # lead to energies that are not comparable with kpoint grids with
        # the same energy for smaller supercells: 11x11x7 (density=2500)
        if np.all(kpoints > 3) or np.all(kpoints <= 3):
            pass
        elif kpoints[kpoints <= 3].size == 1:
            # According to light testing, a 11x11x2 is convergent.
            pass
        else:
            Path('error').touch()
            Path('error_kpoints_dim_sparse').touch()
            print(colored(f"KPOINTS has too few points along one of the axes: {kpoints}", "red"))
            return 1
        inputset.write_input(output_dir="ATAT")

    # Run VASP
    if args.skip_relax:
        print(colored("--skip_relax passed.  Not running relaxation in ./ATAT", "yellow"))
    else:
        if not run_vasp(args.vasp_command, "ATAT"):
            return 1
        vaspdir = IMDGVaspDir("ATAT")
        str_before = vaspdir.initial_structure
        str_after = vaspdir.structure
        if not atat.check_volume_distortion(str_before, str_after, args.max_strain):
            print(colored(f"POSCAR->CONTCAR strain exceeds {args.max_strain*100}%", "red"))
            Path('error').touch()
            Path('error_strain').touch()
            return 1
        sublattice = Structure.from_file('str.out')
        if not atat.check_sublattice_flip(str_before, str_after, sublattice):
            print(colored(
                "POSCAR&CONTCAR flipped sublattice configuration.", "yellow"))
            sublattice2 =\
                atat.fit_sublattice_to_structure(sublattice, str_after)
            if not Path('str.out.old').is_file():
                Path('str.out').rename('str.out.old')
                sublattice2.to_file('str.out', fmt='atat')
                print(colored("Updating str.out", "yellow"))
            else:
                print(colored("str.out.old exists.  Not overwriting", "yellow"))
            sublattice = sublattice2
        str_after_normalized = str_after.copy()
        str_after_normalized.lattice = sublattice.lattice
        dist_sublattice = structure_distance(
            str_after_normalized, sublattice,
            # Compare specie-insensitively
            match_first=True,
            match_species=False)
        Path("sublattice_deviation").write_text(
            f"{dist_sublattice:.4f}\n", encoding='utf-8')
        if args.sublattice_cutoff is not None:
            if dist_sublattice >= args.sublattice_cutoff:
                print(colored(
                    f"Sublattice deviation {dist_sublattice:.2f} >= cutoff"
                    f" {args.sublattice_cutoff:.2f}.  Marking as error", "red"))
                Path('error').touch()
                Path('error_sublattice').touch()

    if Path('ATAT.SCF').is_dir():
        print(colored("ATAT.SCF already exists.  Not modifying", "yellow"))
    else:
        # Create SCF input
        args.input_directory = "ATAT"
        inputset_data = derive_scf(args)
        assert len(inputset_data['inputsets']) == 1
        inputset = inputset_data['inputsets'][0]
        inputset.write_input(output_dir="ATAT.SCF")

    # Run VASP
    run = run_vasp(args.vasp_command, "ATAT.SCF")
    if not run:
        return 1

    Path('energy').write_text(f"{float(run.final_energy)}\n")
    return 0
