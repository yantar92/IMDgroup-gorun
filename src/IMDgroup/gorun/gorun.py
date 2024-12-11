"""Automate VASP job submission to Slurm queue from current VASP dir.
"""
import os
import sys
import re
import warnings
import argparse
import datetime
import tomllib
import subprocess
import glob
from termcolor import colored
from xml.etree.ElementTree import ParseError
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.gorun.slurm import\
    (barf_if_no_cmd, directory_queued_p,
     clear_slurm_logs, get_best_script)
from IMDgroup.gorun.cleanVASP import prepare_vasp_dir


def _showwarning(message, category, _filename, _lineno, file=None, _line=None):
    """Print warning in nicer way."""
    output = colored(
        f"{category.__name__}: ", "yellow", attrs=['bold']) +\
        f"{message}"
    print(output, file=file or sys.stderr)


warnings.showwarning = _showwarning


def nebp(path):
    """Return True when PATH is a NEB-like run.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        if 'IMAGES' in incar:
            return True
    return False


def get_args():
    """Parse command line args and return arg dictionary."""
    argparser = argparse\
        .ArgumentParser(
            description="""Queue VASP run for current directory
Do nothing when VASP run is already queued for the current directory.
Also,
1. If CONTCAR is present, copy it over to POSCAR
2. Generate POTCAR file
3. Backup old VASP files and slurm logs

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
    return argparser.parse_args()


def barf_if_no_env(variable: str) -> None:
    """Throw an error when environment VARIABLE is not set.
    """
    if variable not in os.environ:
        print(colored(
            f'{variable} is not set, while it must be in IMD Group bashrc',
            'red'))
        sys.exit(1)


def default_config_path():
    """Return path to default config file location.
    """
    barf_if_no_env('IMDGroup')
    return f"{os.environ['IMDGroup']}/dist/etc/gorun.toml"


def get_config(args: dict) -> dict:
    """Read config file, if any.
    """
    if args.config is None:
        config_path = default_config_path()
    else:
        config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    raise FileNotFoundError(f"Cannot find config in {config_path}")


def current_server(config: dict) -> str:
    """Get current server name, as named in the CONFIG.
    """
    barf_if_no_cmd('uname')
    uname = subprocess.check_output(
        'uname -n', shell=True).decode('utf-8').strip()
    for server, names in config['cluster']['names'].items():
        if uname in names:
            return server
    return None


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


def get_user_sbatch_args(script_args) -> dict[str, str]:
    """Extract explicit sbtach arguments from SCRIPT_ARGS.
    """
    sbatch_args = {}
    if script_args.number_of_nodes is not None:
        sbatch_args['nodes'] = str(script_args.number_of_nodes)
    if script_args.time_limit is not None:
        sbatch_args['time'] = str(script_args.time_limit)
    return sbatch_args


def get_default_job_name():
    """Generate sensible job name for a given INCAR.
    """
    incar = Incar.from_file('INCAR')
    return {'job-name': incar.get('SYSTEM', 'unknown')}


def get_sbatch_args(script_args: dict, config: dict,
                    server: str, queue: str) -> dict:
    """Return a dict of arguments for sbatch command.
    Combine SCRIPT_ARGS with script CONFIG for QUEUE in SERVER.
    """
    return config[server]['defaults']['sbatch'] |\
        config[server][queue]['sbatch'] |\
        get_default_job_name() |\
        get_user_sbatch_args(script_args) |\
        {'partition': queue}


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
        if nebp(path):
            for dirname in os.listdir(path):
                dirpath = os.path.join(path, dirname)
                if os.path.isdir(dirpath) and re.match(r'[0-9]+', dirname):
                    if not directory_converged_p(dirpath):
                        return False
        else:
            try:
                run = Vasprun(os.path.join(path, 'vasprun.xml'))
                if not run.converged:
                    return False
            except (ParseError, FileNotFoundError):
                return False
        return True
    return False


def main():
    """Run the script."""
    barf_if_no_env("VASP_PATH")
    barf_if_no_env("VASP_PP_PATH")
    args = get_args()
    config = get_config(args)
    server = current_server(config)
    queues = config[server]['queues']

    if server is None:
        print(colored(
            'Running on unknown server. Please adjust the config.',
            "red"))
        return 1

    working_dir = os.getcwd()
    if directory_queued_p(working_dir):
        print(colored(
            "A job is already running in this directory. "
            "Exiting without submitting a new job.",
            "yellow"))
        return 1

    if directory_converged_p('.'):
        print(colored('VASP run already converged. Skipping', "yellow"))

    if directory_contains_vasp_outputp('.'):
        run_folder = get_next_run_folder()
        backup_current_dir(run_folder)

    prepare_vasp_dir('.')
    if nebp('.'):
        for dirname in sorted(os.listdir('.')):
            if os.path.isdir(dirname) and re.match(r'[0-9]+', dirname):
                prepare_vasp_dir(dirname)

    clear_slurm_logs('.')
    if nebp('.'):
        for dirname in sorted(os.listdir('.')):
            if os.path.isdir(dirname) and re.match(r'[0-9]+', dirname):
                clear_slurm_logs(dirname)

    script = get_best_script(
        [get_sbatch_args(args, config, server, queue) for queue in queues],
        f"""
{config[server]['VASP-setup']}

mpiexec {os.environ["VASP_PATH"]}/bin/vasp_ncl
        """)
    with open('sub', 'w', encoding='utf-8') as f:
        f.write(script)

    # Submit the job using sbatch.
    os.system("sbatch sub")
    print(colored('Job submitted to SLURM scheduler.', "green"))
    return 0
