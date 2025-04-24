"""Run VASP according to ATAT-generated structure.
Use VASP configuration from parent directory as reference.
Run (1) parent directory configuration; (2) SCF run; (3) Write energy
or error files.
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
from IMDgroup.pymatgen.cli.imdg_derive import atat, scf
from IMDgroup.pymatgen.core.structure import structure_is_valid2
from pymatgen.io.vasp.outputs import Vasprun
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
        default=0.5,
        type=float,
        help="Distance tolerance to reject structure")
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
        print(f"{directory} already contains converged output. Skipping.")
        return Vasprun(Path(directory) / "vasprun.xml")

    with open(Path(directory) / 'vasp.out', 'a') as f:
        print(f"Running {vasp_command} in {directory}")
        result = subprocess.run(
            vasp_command,
            cwd=directory,
            check=False,
            stdout=f,
            stderr=subprocess.STDOUT)
    try:
        run = Vasprun(Path(directory) / "vasprun.xml")
    except (ValueError, FileNotFoundError, ParseError):
        run = None
    if result.returncode != 0 or (run is not None and not run.converged):
        Path('error').touch()
        return False
    if run is None:
        return False
    return run


def main(args=None):
    if args is None:
        args = get_args()
    # Generate VASP input
    args.atat_structure = "str.out"
    args.input_directory = "../"
    inputset_data = atat(args)
    assert len(inputset_data['inputsets']) == 1
    inputset = inputset_data['inputsets'][0]
    if not structure_is_valid2(inputset.structure, frac_tol=args.frac_tol):
        Path('error').touch()
        Path('error_atoms_too_close').touch()
        print("str.out has atoms too close to each other")
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
    elif kpoints[kpoints < 3].size == 1:
        # According to light testing, a 11x11x2 is convergent.
        pass
    else:
        Path('error').touch()
        Path('error_kpoints_dim_sparse').touch()
        print(f"KPOINTS has too few points along one of the axes: {kpoints}")
        return 1
    inputset.write_input(output_dir="ATAT")

    # Run VASP
    if args.skip_relax:
        print("--skip_relax passed.  Not running relaxation in ./ATAT")
    else:
        if not run_vasp(args.vasp_command, "ATAT"):
            return 1

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
