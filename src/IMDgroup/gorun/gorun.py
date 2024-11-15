"""Automate VASP job submission to Slurm queue from current VASP dir.
"""
import os
import sys
import argparse
import datetime
import tomllib
import subprocess
import glob
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.gorun.slurm import\
    (barf_if_no_cmd, directory_queued_p,
     clear_slurm_logs, get_best_script)
from IMDgroup.gorun.cleanVASP import prepare_vasp_dir


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
        print(f'{variable} is not set, while it must be in IMD Group bashrc')
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
    print(f"Backing up {os.getcwd()} ...")
    subprocess.check_call(f"rsync * './{to}'", shell=True)
    print(f"Backing up {os.getcwd()} ... done")


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


def main():
    """Run the script."""
    barf_if_no_env("VASP_PATH")
    barf_if_no_env("VASP_PP_PATH")
    args = get_args()
    config = get_config(args)
    server = current_server(config)
    queues = config[server]['queues']

    if server is None:
        print('Running on unknown server. Please adjust the config.')
        return 1

    working_dir = os.getcwd()
    if directory_queued_p(working_dir):
        print("A job is already running in this directory. " +
              "Exiting without submitting a new job.")
        return 1

    if os.path.exists('OUTCAR') and os.path.getsize('OUTCAR') > 0:
        run_folder = get_next_run_folder()
        backup_current_dir(run_folder)

    clear_slurm_logs()
    prepare_vasp_dir()

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
    print('Job submitted to SLURM scheduler.')
    return 0
