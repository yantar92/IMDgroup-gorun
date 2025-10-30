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


"""Slurm utils.
"""

import subprocess
import logging
import tempfile
import re
import glob
import os
import dateutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _executable_find(cmd: str) -> bool:
    """Return True when CMD executable is available.
    CMD is a string.
    """
    try:
        return bool(subprocess.check_output(['which', cmd]))
    except subprocess.CalledProcessError:
        return False


def barf_if_no_cmd(cmd: str) -> None:
    """Raise exeption when CMD (string) is not available.
    """
    if not _executable_find(cmd):
        raise FileNotFoundError(f'Command not available: {cmd}')


def directory_queued_p(path: str) -> bool:
    """Return True when there an existing queued task in DIR (string).
    """
    barf_if_no_cmd('squeue')
    return bool(path in subprocess.check_output
                ("squeue -u $USER -o %Z | tail -n +2",
                 shell=True).decode('utf-8').split())


def user_job_count() -> int:
    """Return the number of currently running slurm jobs.
    """
    barf_if_no_cmd('squeue')
    barf_if_no_cmd('wc')
    return int(subprocess.check_output("squeue -u $USER -o %Z | wc -l", shell=True)) - 1


def clear_slurm_logs(path='.'):
    """Clear all the slurm logs in PATH.
    """
    # Find all SLURM output files.
    vaspout = Path(path) / 'vasp.out'
    if vaspout.is_file():
        extra_logs = [vaspout]
    else:
        extra_logs = []
    for slurm_file in glob.glob(os.path.join(path, "slurm-*.out")) + extra_logs:
        try:
            os.remove(slurm_file)
            print(f"Deleted old SLURM file: {slurm_file}")
        except OSError as e:
            print(f"Error deleting SLURM file {slurm_file}: {e}")


def sbatch_script(shebang, args: dict[str, str], script: str) -> str:
    """Generate sbatch SCRIPT using SHEBANG, passing ARGS to sbatch.
    SCRIPT is a bash script to be queued.
    Return generated script, as a string.
    """
    sbatch_lines = [f'#SBATCH --{arg}="{value}"' for arg, value in args.items()]
    return f"""{shebang}
{"\n".join(sbatch_lines)}
{script}
"""


def sbatch_estimate_start(script: str):
    """Estimate waiting time until SCRIPT starts running.
    Return a tuple of (waiting_time, nproc), as (dateutil.timedelta,
    int) object.  If SCRIPT cannot start, return None.
    """
    barf_if_no_cmd('sbatch')
    barf_if_no_cmd('date')
    with tempfile.NamedTemporaryFile('w', dir="./") as sub:
        sub.write(script)
        sub.flush()
        try:
            output = subprocess.check_output(
                f"sbatch --test-only {sub.name}",
                shell=True,
                stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError as e:
            if "Invalid account or account/partition combination specified"\
               in e.output.decode('utf-8'):
                logger.info("Unavailable queue/account combination.  Skipping")
                return None
            if "allocation failure: Access/permission denied" in e.output.decode('utf-8'):
                logger.info("Running from inside a node.  Making up time estimate")
                now_time_str = subprocess.check_output(
                    "date +%Y-%m-%dT%H:%M:%S", shell=True).decode('utf-8').strip()
                now_time = dateutil.parser.isoparse(now_time_str)
                return (now_time - now_time, 1)
            print("Failed to execute sbatch in test mode:")
            print(e.output)
            print(f"Script:\n-----\n{script}\n-----\n")
            raise e
    pattern = "sbatch: Job [0-9]+ to start at ([^ ]+) " +\
        "using ([0-9]+) processors on nodes [^ ]+ in partition [^ ]+"
    match = re.match(pattern, output)
    if match is None:
        return output
    scheduled_time_str = match.group(1)
    ncpus = int(match.group(2))
    scheduled_time = dateutil.parser.isoparse(scheduled_time_str)
    now_time_str = subprocess.check_output(
        "date +%Y-%m-%dT%H:%M:%S", shell=True).decode('utf-8').strip()
    now_time = dateutil.parser.isoparse(now_time_str)
    return (scheduled_time - now_time, ncpus)


def get_best_script(
        alt_args: list[dict],
        script,
        shebang: str = "#!/usr/bin/env bash",
) -> str:
    """Choose across ALT_ARGS lists, selecting the best sbatch script.
    The best script will finish running the earliest.
    SHEBANG is shebang line to be used.
    """
    scripts = [sbatch_script(shebang, args, script) for args in alt_args]
    schedule_estimates = [sbatch_estimate_start(script) for script in scripts]

    if all(data is None for data in schedule_estimates):
        raise OSError(
            "No single queue is available for running.  Check grant limits")

    now = dateutil.utils.today()
    best_finish_time = now + \
        dateutil.relativedelta.relativedelta(hours=9999999)
    best_script = scripts[0]

    max_cpus = max(data[1] for data in schedule_estimates
                   if data is not None)

    for script_args, script_, schedule_estimate in zip(
            alt_args, scripts, schedule_estimates):
        if schedule_estimate is None:
            continue
        hours, minutes, seconds = script_args['time'].split(":")
        scheduled_delta, cpus = schedule_estimate
        finish_time = now + scheduled_delta +\
            max_cpus/cpus *\
            dateutil.relativedelta.relativedelta(
                hours=int(hours), minutes=int(minutes), seconds=int(seconds))
        print(f'Candidate time ({script_args["partition"]}): {finish_time}')
        if finish_time < best_finish_time:
            best_finish_time = finish_time
            best_script = script_
    print(f'Best finish time: {best_finish_time}')
    return best_script
