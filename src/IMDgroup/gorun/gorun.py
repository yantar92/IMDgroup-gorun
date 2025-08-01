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


"""Automate VASP job submission to Slurm queue from current VASP dir.
"""
import os
import sys
import re
import warnings
import argparse
import datetime
import time
import subprocess
import glob
from termcolor import colored
from IMDgroup.gorun.slurm import\
    (barf_if_no_cmd, directory_queued_p,
     clear_slurm_logs, get_best_script, user_job_count)
from IMDgroup.gorun.cleanVASP import\
    (prepare_vasp_dir, nebp, mdp, directory_converged_p,
     directory_contains_vasp_outputp)
from IMDgroup.gorun.sbatch import\
    (barf_if_no_env, get_config, current_server, get_sbatch_args)


def _showwarning(message, category, _filename, _lineno, file=None, _line=None):
    """Print warning in nicer way."""
    output = colored(
        f"{category.__name__}: ", "yellow", attrs=['bold']) +\
        f"{message}"
    print(output, file=file or sys.stderr)


warnings.showwarning = _showwarning


def get_args():
    """Parse command line args and return arg dictionary."""
    argparser = argparse\
        .ArgumentParser(
            description="""Queue VASP run for current directory
Do nothing when VASP run is already queued for the current directory.
Also,
1. Make sure that vdw_kernel.bindat is copied over from VASP source dir
2. If CONTCAR is present, copy it over to POSCAR
3. Generate POTCAR file
4. Backup old VASP files and slurm logs

Slurm script will be saved under name 'sub'.""",
            epilog="""Example:
gorun 2 24:00:00
- Submits a job requesting 2 nodes with a 24-hour time limit.""")

    argparser.add_argument(
        "number_of_nodes", help="number of nodes to request (optional)",
        nargs="?",
        default=None)
    argparser.add_argument(
        "time_limit",
        nargs="?",
        help="Time limit for the job in the format hh:mm:ss (optional)",
        default=None)
    argparser.add_argument(
        "--config",
        help="Path to configuration file " +
        "(default: $IMDGroup/dist/etc/gorun.toml)",
        default=None)
    argparser.add_argument(
        "--queue",
        help="Queue to be used (default: find best)",
        type=str,
        default=None)
    argparser.add_argument(
        "--no_vasp_config",
        help="Whether to configure environment for VASP",
        action="store_true")
    argparser.add_argument(
        "--local",
        help="Whether to run locally (do not use sbatch)",
        action="store_true")
    argparser.add_argument(
        "--force",
        help="When provided, force running VASP",
        action="store_true")
    argparser.add_argument(
        "--keep_potcar",
        help="When POTCAR is already present, do not re-generate it",
        action="store_true")
    argparser.add_argument(
        "--max_slurm_jobs",
        help="Maximum of slurm jobs allowed to run simultaneously. "
        "Wait until squeue shortens below this number before running VASP.",
        type=int,
        default=0
    )
    return argparser.parse_args()


def get_next_run_folder() -> str:
    """Get the next available 'gorun_*' folder name."""
    prefix = 'gorun'
    existing_folders = glob.glob(prefix + "_*")
    # Extract numeric parts from folder names
    # and find the next available number
    run_numbers = [int(folder.split('_')[1])
                   for folder in existing_folders
                   if folder.split('_')[1].isdigit()]
    next_run_number = max(run_numbers) + 1 if run_numbers else 1
    return f"{prefix}_{next_run_number}_" +\
        datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")


def backup_current_dir(to: str) -> None:
    """Backup current directory to directory TO.
    """
    barf_if_no_cmd('rsync')
    print(f"Backing up {os.getcwd()}")
    subprocess.check_call(f"rsync -q * './{to}'", shell=True)
    if nebp('.'):
        print("Detected NEB-like input")
        for dirname in os.listdir('.'):
            if os.path.isdir(dirname) and re.match(r'[0-9]+', dirname):
                subprocess.check_call(f"rsync -qr {dirname} './{to}'", shell=True)


def main():
    """Run the script."""
    barf_if_no_env("VASP_PATH")
    barf_if_no_env("VASP_PP_PATH")
    args = get_args()
    config = get_config(args)
    server = current_server(config)
    queues = config[server]['queues']
    if args.queue is not None:
        queues = [args.queue]

    if server is None:
        print(colored(
            'Running on unknown server. Please adjust the config.',
            "red"))
        return 1

    working_dir = os.getcwd()
    if directory_queued_p(working_dir) and not args.force:
        print(colored(
            "A job is already running in this directory. "
            "Exiting without submitting a new job.",
            "yellow"))
        # This is useful to return normally here as
        # the purpose of gorun is achieved.
        return 0

    if not os.path.isfile('INCAR'):
        print(colored(
            'No INCAR found in current dir. '
            'Exiting without submitting a new job.',
            "yellow"))
        return 1

    if not mdp('.') and directory_converged_p('.') and not args.force:
        print(colored(
            'VASP run already converged. '
            'Exiting without submitting a new job.',
            "yellow"))
        return 1

    if args.max_slurm_jobs > 0:
        while user_job_count() >= args.max_slurm_jobs:
            print("Waiting for submitted jobs to finish")
            time.sleep(10)

    if directory_contains_vasp_outputp('.'):
        run_folder = get_next_run_folder()
        backup_current_dir(run_folder)

    prepare_vasp_dir('.', args.keep_potcar)
    if nebp('.'):
        for dirname in sorted(os.listdir('.')):
            if os.path.isdir(dirname) and re.match(r'[0-9]+', dirname):
                prepare_vasp_dir(dirname)

    clear_slurm_logs('.')
    if nebp('.'):
        for dirname in sorted(os.listdir('.')):
            if os.path.isdir(dirname) and re.match(r'[0-9]+', dirname):
                clear_slurm_logs(dirname)

    base_script = f"""
{config[server]['VASP-setup'] if not args.no_vasp_config else ""}

{config[server].get('mpiexec', 'mpiexec')} {os.environ["VASP_PATH"]}/bin/vasp_ncl
        """
    shebang = config[server].get('shebang', "#!/usr/bin/bash")
    if args.local:
        script = f"{shebang}\n{base_script}"
    else:
        script = get_best_script(
            [get_sbatch_args(args, config, server, queue) for queue in queues],
            base_script,
            shebang)
        with open('sub', 'w', encoding='utf-8') as f:
            f.write(script)

    # Submit the job using sbatch.
    if args.local:
        with open("vasp.out", "a", encoding='utf-8') as f:
            subprocess.run(
                script,
                shell=True,
                check=True,
                stdout=f,
                stderr=f,
                text=True
            )
        # os.system("bash sub > vasp.out 2>&1")
        print(colored('Running job locally...', "green"))
    else:
        os.system("sbatch sub")
        print(colored('Job submitted to SLURM scheduler.', "green"))
    return 0
