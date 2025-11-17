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


"""Cleanup and check VASP inputs.
"""

import os
import shutil
import re
import warnings
import ase.io.vasp
from ase.calculators.vasp import Vasp
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Potcar
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from xml.etree.ElementTree import ParseError


def directory_contains_vasp_outputp(path):
    """Return True when PATH contains VASP outputs.
    """
    outcar_path = os.path.join(path, 'OUTCAR')
    if os.path.exists(outcar_path) and os.path.getsize(outcar_path) > 0:
        return True
    if nebp(path):
        for dirname in os.listdir(path):
            dirpath = os.path.join(path, dirname)
            if os.path.isdir(dirpath) and re.match(r'[0-9]+', dirname):
                if directory_contains_vasp_outputp(dirpath):
                    return True
    return False


def directory_converged_p(path):
    """Return True when PATH contains converged VASP output.
    """
    if directory_contains_vasp_outputp(path):
        vaspdir = IMDGVaspDir(path)
        return vaspdir.converged_electronic and vaspdir.converged_ionic
    return False


def nebp(path):
    """Return True when PATH is a NEB-like run.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        if 'IMAGES' in incar:
            return True
    return False


def mdp(path):
    """Return True when PATH is an MD run.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        if incar.get('IBRION') == 0:
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


def generate_potcar(path='.', keep_existing=False) -> None:
    """Generate POTCAR from POSCAR file in PATH.
    When KEEP_EXISTING is True and POTCAR already exists, do not
    re-generate it.
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
        potcar_path = os.path.join(path, 'POTCAR')
        if os.path.isfile(potcar_path) and keep_existing:
            print(f'{path}: Not updating existing POTCAR.')
            return
        atoms = ase.io.vasp.read_vasp(poscar_path)
        calc_temp = Vasp(xc='PBE', setups={'base': 'recommended'})
        calc_temp.initialize(atoms)
        size_before = 0
        prev_potcar = None
        if os.path.isfile(potcar_path):
            prev_potcar = Potcar.from_file(potcar_path)
            size_before = os.path.getsize(potcar_path)
            shutil.copy2(potcar_path, potcar_path + '.old')
        calc_temp.write_potcar()
        # Sometimes, for initial/final NEB inputs, POTCAR is not written
        if os.path.isfile(potcar_path):
            size_after = os.path.getsize(potcar_path)
            if size_before == 0:
                print(f'{path}: Generated POTCAR.')
            elif size_after != size_before:
                print(f'{path}: Updated POTCAR.  (old POTCAR saved to POTCAR.old)')
                new_potcar = Potcar.from_file(potcar_path)
                assert prev_potcar is not None
                if prev_potcar.symbols != new_potcar.symbols:
                    warnings.warn(
                        f"Changed pseudopotentials from {prev_potcar.symbols}"
                        f" to {new_potcar.symbols}"
                    )
            else:
                os.unlink(potcar_path + '.old')


def check_incar(path):
    """Check INCAR in PATH.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        incar.check_params()


def prepare_vasp_dir(path='.', keep_potcar=False, keep_poscar=False) -> None:
    """Prepare and cleanup VASP inputs in PATH.
    When KEEP_POTCAR is True, and POTCAR file already exist do not
    re-generate it.
    When KEEP_POSCAR is True, keep POSCAR file and do not overwrite it
    with CONTCAR, even if CONTCAR is present.
    """
    check_incar(path)
    # If CONTCAR exists and is non-empty, copy it to POSCAR.
    if not keep_poscar:
        contcar_to_poscar(path)
    # Clean the POSCAR, INCAR, and KPOINTS files before running the job.
    clean_vasp_inputs(path)
    # If POSCAR exists, initialize ASE and generate the POTCAR file.
    generate_potcar(path, keep_potcar)
    # Copy over vdw kernel
    put_vdw_kernel(path)
