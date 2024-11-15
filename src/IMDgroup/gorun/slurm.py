"""Slurm utils.
"""

import subprocess
import logging
import tempfile
import re
import glob
import os
import dateutil

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
                 shell=True).split())


def clear_slurm_logs():
    """Clear all the slurm logs in current directory.
    """
    for slurm_file in glob.glob("slurm-*.out"):  # Find all SLURM output files.
        try:
            os.remove(slurm_file)
            print(f"Deleted old SLURM file: {slurm_file}")
        except OSError as e:
            print(f"Error deleting SLURM file {slurm_file}: {e}")


def sbatch_script(args: dict[str, str], script: str) -> str:
    """Generate sbatch SCRIPT passing ARGS to sbatch.
    SCRIPT is a bash script to be queued.
    Return generated script, as a string.
    """
    sbatch_lines = [f"#SBATCH --{arg}={value}" for arg, value in args.items()]
    return f"""#!/bin/env bash
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
    ncpus = match.group(2)
    scheduled_time = dateutil.parser.isoparse(scheduled_time_str)
    now_time_str = subprocess.check_output(
        "date +%Y-%m-%dT%H:%M:%S", shell=True).decode('utf-8').strip()
    now_time = dateutil.parser.isoparse(now_time_str)
    return (scheduled_time - now_time, ncpus)


def get_best_script(alt_args: list[dict], script) -> str:
    """Choose across ALT_ARGS lists, selecting the best sbatch script.
    The best script will finish running the earliest.
    """
    scripts = [sbatch_script(args, script) for args in alt_args]
    schedule_estimates = [sbatch_estimate_start(script) for script in scripts]

    now = dateutil.utils.today()
    best_finish_time = now
    best_script = scripts[0]

    max_cpus = max(data[1] for data in schedule_estimates
                   if data is not None)

    for script_args, script, schedule_estimate in zip(
            alt_args, scripts, schedule_estimates):
        if schedule_estimate is None:
            continue
        hours, minutes, seconds = script_args['time'].split(":")
        scheduled_delta, cpus = schedule_estimate
        finish_time = now + scheduled_delta +\
            max_cpus/cpus *\
            dateutil.relativedelta.relativedelta(
                hours=int(hours), minutes=int(minutes), seconds=int(seconds))
        print('Candidate time (%s): %s', script_args['partition'], finish_time)
        if finish_time < best_finish_time:
            best_finish_time = finish_time
            best_script = script
    print('Best finish time: %s', best_finish_time)
    return best_script
