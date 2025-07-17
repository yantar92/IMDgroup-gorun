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


"""Run maps cluster expansion in current directory.
"""
from copy import Error
import os
import shutil
import argparse
from pathlib import Path
from termcolor import colored
from IMDgroup.gorun.sbatch import\
    (get_config, current_server, barf_if_no_env, get_sbatch_args)
from IMDgroup.gorun.slurm import\
    (directory_queued_p, get_best_script, barf_if_no_cmd)
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet


def get_args():
    """Parse command line args and return arg dictionary."""
    argparser = argparse.ArgumentParser(
        description="""Queue maps run for current directory.
        Do nothing when maps is already running.
        Also,
        1. Make sure that lat.in is present
        2. Make sure that reference VASP input is present
        """
    )
    argparser.add_argument(
        "--time_limit",
        help="Time limit for the job in the format hh:mm:ss (optional)",
        default=None
    )
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
        "--local",
        help="Whether to run locally (do not use sbatch)",
        action="store_true")
    argparser.add_argument(
        "--kpoints",
        required=True,
        help="Kpoint density")
    argparser.add_argument(
        "--frac_tol",
        required=False,
        default=0.5,
        help="Tolerance for distance between atoms to reject structures")
    argparser.add_argument(
        "--skip_relax",
        help="Whether to skip relaxation run",
        action="store_true")
    argparser.add_argument(
        "--sublattice_cutoff",
        help="Maximum allowed sublattice deviation",
        type=float,
        default=None
    )
    argparser.add_argument(
        "--maps_args",
        help="maps arguments to pass",
        nargs="+")
    args = argparser.parse_args()
    # Force single node for ATAT
    args.number_of_nodes = 1
    return args


def main():
    """Run the script.
    """
    barf_if_no_env("VASP_PATH")
    barf_if_no_cmd("maps")
    barf_if_no_cmd("pollmach")
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
    if directory_queued_p(working_dir):
        print(colored(
            "A job is already running in this directory. "
            "Exiting without submitting a new job.",
            "yellow"))
        return 1

    if not Path('lat.in').is_file():
        print(colored(
            'No lat.in found in current dir. '
            'Exiting without submitting a new job.',
            "yellow"))
        return 1

    test_dir = Path(".__test")
    try:
        test_set = IMDDerivedInputSet(directory=working_dir)
        test_set.write_input(test_dir)
        shutil.rmtree(test_dir)
    except Exception as e:
        print(colored(
            'Cannot use VASP input in current dir sa prototype. '
            f"Error:\n{e}"
        ))
        if test_dir.exists():
            shutil.rmtree(test_dir)
        return 1

    # Cleanup maps lock files
    Path("pollmach_is_running").unlink(missing_ok=True)
    Path("maps_is_running").unlink(missing_ok=True)

    script = get_best_script(
        [get_sbatch_args(args, config, server, queue) for queue in queues],
        f"""
{config[server]['VASP-setup']}

maps {' '.join(args.maps_args)} &
sleep 5
pollmach gorun-atat-local {'--skip_relax' if args.skip_relax else ''} --kpoints={args.kpoints} --frac_tol={args.frac_tol} {'--sublattice_cutoff='+str(args.sublattice_cutoff) if args.sublattice_cutoff else ''} gorun --local --no_vasp_config
""",
        config[server].get('shebang', "#!/usr/bin/bash"))
    with open('sub', 'w', encoding='utf-8') as f:
        f.write(script)

    # Submit the job using sbatch.
    if args.local:
        os.system("bash sub")
        print(colored('Running job locally...', "green"))
    else:
        os.system("sbatch sub")
        print(colored('Job submitted to SLURM scheduler.', "green"))
    return 0
