"""Cleanup and check VASP inputs.
"""

import os
import shutil
import re
import ase.io.vasp
from ase.calculators.vasp import Vasp


def contcar_to_poscar(path) -> None:
    """When CONTCAR exists, copy it over to POSCAR in PATH.
    """
    contcar_path = os.path.join(path, 'CONTCAR')
    poscar_path = os.path.join(path, 'POSCAR')
    if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
        shutil.copy2(contcar_path, poscar_path)
        print(f"{path}: Found CONTCAR file.  Copying over to POSCAR.")


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
    poscar_path = os.path.join(path, 'POSCAR')
    if os.path.exists(poscar_path) and os.path.getsize(poscar_path) > 0:
        atoms = ase.io.vasp.read_vasp(file=poscar_path)
        calc_temp = Vasp(xc='PBE', setups={'base': 'recommended'})
        calc_temp.initialize(atoms)
        calc_temp.write_potcar()
        print(f'{path}: Generated POTCAR.')


def prepare_vasp_dir(path='.') -> None:
    """Prepare and cleanup VASP inputs in PATH.
    """
    # If CONTCAR exists and is non-empty, copy it to POSCAR.
    contcar_to_poscar(path)
    # Clean the POSCAR, INCAR, and KPOINTS files before running the job.
    clean_vasp_inputs(path)
    # If POSCAR exists, initialize ASE and generate the POTCAR file.
    generate_potcar(path)
