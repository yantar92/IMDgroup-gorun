import os, sys, shutil, subprocess, datetime
import ase.io.vasp
from ase.calculators.vasp import Vasp


def main():
    if 'VASP_PP_PATH' not in os.environ:
        print('VASP_PP_PATH is not set, while it must be in IMD Group bashrc')
        sys.exit(1)

    if 'IMDGroup' not in os.environ:
        print('IMDGroup is not set, while it must be in IMD Group bashrc')
        sys.exit(1)

    if os.environ['IMDGroup'] not in os.environ['VASP_PP_PATH']:
        print("VASP_PP_PATH is not in IMDGroup dir.  Refusing to use non-standard potentials.")
        print(f"{os.environ['IMDGroup']} not in {os.environ['VASP_PP_PATH']}")
        sys.exit(1)

    result = subprocess.check_output("squeue -u plgoimalyi -o %Z | tail -n +2", shell=True).split()

    if os.path.exists('CONTCAR') and os.path.getsize('CONTCAR') > 0:
        shutil.copy2('CONTCAR', 'POSCAR')
        print('File is there')

    if os.path.exists('POSCAR') and os.path.getsize('POSCAR') > 0:
        Atoms = ase.io.vasp.read_vasp(file='POSCAR')
        calc_temp = Vasp(xc='PBE', setups={'base': 'recommended'})
        calc_temp.initialize(Atoms)
        calc_temp.write_potcar()

    CURRENT_DIRECTORY = os.getcwd()
    JOB_ALREADY_RUNNING = 0
    for s in result:
        if CURRENT_DIRECTORY == s.decode("utf-8"):
            JOB_ALREADY_RUNNING = 1

    if JOB_ALREADY_RUNNING == 0:
        with open('sub', 'w') as file:
            file.write('#!/bin/env bash\n')
            file.write('#SBATCH --job-name=test_run\n')
            if len(sys.argv) > 2:
                file.write(f'#SBATCH -N {sys.argv[1]}\n')
                file.write(f'#SBATCH -t {sys.argv[2]}\n')
            else:
                file.write('#SBATCH --nodes=4\n')
                file.write('#SBATCH --time=72:00:00\n')

            file.write('#SBATCH --ntasks-per-node=48\n')
            file.write('#SBATCH --partition=plgrid\n')
            file.write('<<ares-vasp-modules>>\n')
            file.write('file_outcar=./OUTCAR\n')
            file.write('if [ -e "$file_outcar" ]; then\n')
            file.write('    rsync * '+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'/\n')
            file.write('fi\n')
            file.write(f'mpiexec {os.environ["VASP_PATH"]}/bin/vasp_ncl\n')

        os.system("sbatch sub")
