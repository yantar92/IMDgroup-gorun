"""Cleanup and check VASP inputs.
"""

import os
import shutil
import re
import warnings
import ase.io.vasp
from ase.calculators.vasp import Vasp
from pymatgen.io.vasp.inputs import Incar


def nebp(path):
    """Return True when PATH is a NEB-like run.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        if 'IMAGES' in incar:
            return True
    return False


def contcar_to_poscar(path) -> None:
    """When CONTCAR exists, copy it over to POSCAR in PATH.
    """
    contcar_path = os.path.join(path, 'CONTCAR')
    poscar_path = os.path.join(path, 'POSCAR')
    if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
        shutil.copy2(contcar_path, poscar_path)
        print(f"{path}: Found CONTCAR file.  Copying over to POSCAR.")


def put_vdw_kernel(path) -> None:
    """Copy vdw_kernel.bindat from $VASP_PATH.
    1. Try $VASP_PATH/vdw_kernel.bindat
    2. Try $VASP_PATH/testsuite/tests/CuC_vdW/vdw_kernel.bindat
    3. If not present, display a warning.
    """
    vdw_path_1 = os.path.join(
        os.environ['VASP_PATH'], 'vdw_kernel.bindat')
    vdw_path_2 = os.path.join(
        os.environ['VASP_PATH'], 'testsuite', 'tests',
        'CuC_vdW', 'vdw_kernel.bindat')
    vdw_target = os.path.join(path, 'vdw_kernel.bindat')
    for vdw_path in [vdw_path_1, vdw_path_2]:
        if os.path.exists(vdw_path) and os.path.getsize(vdw_path) > 0:
            shutil.copy2(vdw_path, vdw_target)
            print(f"{path}: Copied vdw_kernel.bindat from {vdw_path}")
            return
    warnings.warn(
        "Cannot find vdw_kernel.bindat.  "
        "VASP.6.4.2 and older may take hours to compute the kernel."
    )


def clean_vasp_input(file_path: str) -> None:
    """Cleanup VASP input file at FILE_PATH.
    Cleanup newlines, non-printable chars, and blank lines.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove any unprintable characters (e.g., BOM) and fix line endings.
    clean_content = content.replace('\r\n', '\n').replace('\xEF\xBB\xBF', '')
    # Remove blank lines with tabs.
    # See https://www.vasp.at/wiki/index.php/INCAR
    clean_content = re.sub(r"^\t+$", "", clean_content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(clean_content)

    if clean_content != content:
        print(f'Cleaned file: {file_path}')


def clean_vasp_inputs(path='.') -> None:
    """Clean all the VASP input files in PATH.
    """
    for file in ['POSCAR', 'INCAR', 'KPOINTS']:
        if os.path.exists(os.path.join(path, file)):
            clean_vasp_input(os.path.join(path, file))


def generate_potcar(path='.') -> None:
    """Generate POTCAR from POSCAR file in PATH.
    """
    poscar_paths = [os.path.join(path, 'POSCAR')]
    if nebp(path):
        poscar_paths += [os.path.join(path, "00", "POSCAR")]
    poscar_path = None
    for p in poscar_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            poscar_path = p
            break
    if poscar_path is not None:
        atoms = ase.io.vasp.read_vasp(filename=poscar_path)
        calc_temp = Vasp(xc='PBE', setups={'base': 'recommended'})
        calc_temp.initialize(atoms)
        potcar_path = os.path.join(path, 'POTCAR')
        size_before = 0
        if os.path.isfile(potcar_path):
            size_before = os.path.getsize(potcar_path)
        calc_temp.write_potcar()
        # Sometimes, for initial/final NEB inputs, POTCAR is not written
        if os.path.isfile(potcar_path):
            size_after = os.path.getsize(potcar_path)
            if size_before == 0:
                print(f'{path}: Generated POTCAR.')
            elif size_after != size_before:
                print(f'{path}: Updated POTCAR.')


def check_incar(path):
    """Check INCAR in PATH.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        incar.check_params()


def prepare_vasp_dir(path='.') -> None:
    """Prepare and cleanup VASP inputs in PATH.
    """
    check_incar(path)
    # If CONTCAR exists and is non-empty, copy it to POSCAR.
    contcar_to_poscar(path)
    # Clean the POSCAR, INCAR, and KPOINTS files before running the job.
    clean_vasp_inputs(path)
    # If POSCAR exists, initialize ASE and generate the POTCAR file.
    generate_potcar(path)
    # Copy over vdw kernel
    put_vdw_kernel(path)
