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
import math
from pathlib import Path
import numpy as np
from termcolor import colored
from IMDgroup.pymatgen.cli.imdg_derive import atat, scf
from IMDgroup.pymatgen.core.structure import structure_distance, structure_is_valid2
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Structure, DummySpecie
from xml.etree.ElementTree import ParseError
from IMDgroup.gorun.cleanVASP import directory_converged_p


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
        "--skip_relax",
        help="Whether to skip relaxation run",
        action="store_true")
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
    except (ValueError, ParseError):
        run = 'failed'
    except FileNotFoundError:
        run = None
    if result.returncode != 0 or run == 'failed' or (run is not None and not run.converged):
        Path('error').touch()
        return False
    if run is None:
        return False
    return run


def check_volume_distortion(
        str_before: Structure,
        str_after: Structure,
        # 0.1 is what is done by ATAT in checkcell subroutine
        threshold: float = 0.1) -> bool:
    """Return False when lattice distortion is too large for STR_BEFORE and STR_AFTER.
    The lattice distortion is a norm of engineering strain tensor.
    The distortion is considered "too large" when it is no less than THRESHOLD.
    The default 0.1 threshold is following ATAT source code.
    """
    lat_before = str_before.lattice.matrix
    lat_after = str_after.lattice.matrix
    # normalize matrices
    lat_before = lat_before / math.pow(np.linalg.det(lat_before), 1.0/3.0)
    lat_after = lat_after / math.pow(np.linalg.det(lat_after), 1.0/3.0)
    transform = np.dot(np.linalg.inv(lat_before), lat_after) - np.eye(3)
    strain = (transform + np.linalg.matrix_transpose(transform))/2.0
    distortion = np.linalg.norm(strain)
    if distortion < threshold:
        return True
    print(colored(f"POSCAR->CONTCAR strain exceeds {threshold*100}%: {distortion}"))
    return False


def check_sublattice_flip(
        str_before: Structure,
        str_after: Structure,
        sublattice: Structure) -> bool:
    """Check if STR_AFTER flipped its SUBLATTICE sites compared to STR_BEFORE.
    Return True when STR_AFTER occupies the same sublattice configuration as
    STR_BEFORE.  The SUBLATTICE is full sublattice with all the sites
    occupied (as per str.in).

    Note that specie sites that are scanned by cluster expansion must be
    marked with the same specie name in all the arguments.  For
    example, if ATAT is running on Li, Vac system, both Li and Vac species
    should be replaced with, say X dummy specie.
    """
    # First, scale STR_AFTER lattice to fit STR_BEFORE and SUBLATTICE
    # Assume that STR_BEFORE and SUBLATTICE have the same lattices
    str_after_normalized = str_after.copy()
    str_after_normalized.lattice = str_before.lattice
    dist_relax = structure_distance(str_before, str_after_normalized)

    # Replace all species with X to compare with anonymous sublattice
    str_after_normalized = str_after_normalized.copy()
    for site in str_after_normalized:
        site.species = DummySpecie('X')
    sublattice = sublattice.copy()
    for site in sublattice:
        site.species = DummySpecie('X')
    dist_sublattice = structure_distance(str_after_normalized, sublattice)

    if np.isclose(dist_relax, dist_sublattice, rtol=0.001):
        return True
    print(colored("POSCAR&CONTCAR flipped sublattice configuration"))
    return False


def read_sublattice() -> Structure:
    """Read full sublattice from str.out.
    Replace vacancies (Vac) with X dummy elements.
    """
    from pymatgen.io.atat import Mcsqs
    # We manually replace Vac with X instances that can be read by pymatgen.
    atat_structure_text = Path('str.out').read_text(encoding='utf-8')
    atat_structure_text = atat_structure_text.replace("Vac", "X")
    return Mcsqs.structure_from_str(atat_structure_text)


def main(args=None):
    if args is None:
        args = get_args()
    if Path('ATAT').is_dir():
        print(colored("ATAT already exists.  Not modifying", "yellow"))
    else:
        # Generate VASP input
        args.atat_structure = "str.out"
        args.input_directory = "../"
        inputset_data = atat(args)
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
        str_before = Structure.from_file('ATAT/POSCAR')
        str_after = Structure.from_file('ATAT/CONTCAR')
        if not check_volume_distortion(str_before, str_after):
            Path('error').touch()
            Path('error_strain').touch()
            return 1
        sublattice = read_sublattice()
        if not check_sublattice_flip(str_before, str_after, sublattice):
            Path('error').touch()
            Path('error_sublattice').touch()
            return 1

    if Path('ATAT.SCF').is_dir():
        print(colored("ATAT.SCF already exists.  Not modifying", "yellow"))
    else:
        # Create SCF input
        args.input_directory = "ATAT"
        inputset_data = scf(args)
        assert len(inputset_data['inputsets']) == 1
        inputset = inputset_data['inputsets'][0]
        inputset.write_input(output_dir="ATAT.SCF")

    # Run VASP
    run = run_vasp(args.vasp_command, "ATAT.SCF")
    if not run:
        return 1

    Path('energy').write_text(f"{float(run.final_energy)}\n")
    return 0
