"""sbatch config generator.
"""
import os
import sys
import subprocess
import tomllib
from termcolor import colored
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.gorun.slurm import barf_if_no_cmd


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
    if "CLUSTER_NAME" in os.environ:
        uname = os.environ['CLUSTER_NAME']
    else:
        barf_if_no_cmd('uname')
        uname = subprocess.check_output(
            'uname -n', shell=True).decode('utf-8').strip()
    for server, names in config['cluster']['names'].items():
        if uname in names:
            return server
    return uname


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

