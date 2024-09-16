"""Cleanup and check VASP inputs.
"""

import os
import shutil
import re
import ase.io.vasp
from ase.calculators.vasp import Vasp


def contcar_to_poscar() -> None:
    """When CONTCAR exists, copy it over to POSCAR.
    """
    if os.path.exists('CONTCAR') and os.path.getsize('CONTCAR') > 0:
        shutil.copy2('CONTCAR', 'POSCAR')
        print("Found CONTCAR file.  Copying over to POSCAR.")


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


def clean_vasp_inputs() -> None:
    """Clean all the VASP input files.
    """
    for file in ['POSCAR', 'INCAR', 'KPOINTS']:
        if os.path.exists(file):
            clean_vasp_input(file)


def generate_potcar() -> None:
    """Generate POTCAR from POSCAR file.
    """
    if os.path.exists('POSCAR') and os.path.getsize('POSCAR') > 0:
        atoms = ase.io.vasp.read_vasp(file='POSCAR')
        calc_temp = Vasp(xc='PBE', setups={'base': 'recommended'})
        calc_temp.initialize(atoms)
        calc_temp.write_potcar()
        print('Generated POTCAR.')


def prepare_vasp_dir() -> None:
    """Prepare and cleanup VASP inputs in current directory.
    """
    # If CONTCAR exists and is non-empty, copy it to POSCAR.
    contcar_to_poscar()
    # Clean the POSCAR, INCAR, and KPOINTS files before running the job.
    clean_vasp_inputs()
    # If POSCAR exists, initialize ASE and generate the POTCAR file.
    generate_potcar()
