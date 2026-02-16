#!/usr/bin/env bash

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

# gorun-all-ready.sh - Submit pending VASP jobs marked with 'gorun_ready' files
#
# This script searches for directories containing a 'gorun_ready' file,
# waits until the user's Slurm job count drops below a limit, then submits
# the job with 'sbatch sub' and removes the marker file.
#
# Usage: gorun-all-ready.sh [OPTIONS]
#
# Options:
#   -n, --max-jobs NUM   Maximum concurrent Slurm jobs (default: 100)
#   -d, --directory DIR  Search root directory (default: current directory)
#   -t, --timeout SEC    Sleep interval between queue checks (default: 5)
#   -v, --verbose        Print extra information
#   -h, --help           Show this help text
#
# Example:
#   gorun-all-ready.sh -n 20 -d /path/to/projects
#
# Exit codes:
#   0  Success
#   1  Argument error
#   2  Missing required command (squeue, sbatch, find)
#   3  No 'gorun_ready' files found
#   4  Sub‑mission failure (sbatch returned non‑zero)
#
# The script requires a working Slurm environment and the 'sub' script
# in each target directory.

set -euo pipefail

# ------------------------------------------------------------
# Configuration and defaults
# ------------------------------------------------------------
readonly PROGNAME="${0##*/}"
readonly VERSION="1.0.0"

MAX_JOBS=100
SEARCH_DIR="."
SLEEP_INTERVAL=5
VERBOSE=false

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

err() {
    printf "%s: %s\n" "$PROGNAME" "$*" >&2
}

log() {
    if "$VERBOSE"; then
        printf "%s: %s\n" "$PROGNAME" "$*"
    fi
}

die() {
    err "$@"
    exit 1
}

usage() {
    cat <<EOF
$PROGNAME - Submit pending VASP jobs marked with 'gorun_ready' files

Usage: $PROGNAME [OPTIONS]

Options:
  -n, --max-jobs NUM   Maximum concurrent Slurm jobs (default: 100)
  -d, --directory DIR  Search root directory (default: current directory)
  -t, --timeout SEC    Sleep interval between queue checks (default: 5)
  -v, --verbose        Print extra information
  -h, --help           Show this help text
  --version            Show version information

Example:
  $PROGNAME -n 20 -d /path/to/projects

Exit codes:
  0  Success
  1  Argument error
  2  Missing required command
  3  No 'gorun_ready' files found
  4  Submission failure
EOF
}

version() {
    echo "$PROGNAME $VERSION"
}

# Check that required commands are available
check_commands() {
    local cmd
    for cmd in squeue sbatch find; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            die "Required command '$cmd' not found in PATH"
        fi
    done
}

# Count current Slurm jobs for the current user
# Output: number of jobs (integer)
get_n_jobs() {
    # Use -o %Z to get job state column; tail -n +2 skips header line
    # The count includes pending and running jobs.
    local count
    if ! count=$(squeue -u "$USER" -o %Z 2>/dev/null | tail -n +2 | wc -l); then
        err "squeue failed; assuming 0 jobs"
        echo 0
        return 1
    fi
    echo "$count"
}

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--max-jobs)
            if [[ -z "${2:-}" ]]; then
                die "Missing argument for $1"
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                die "Max jobs must be a positive integer"
            fi
            MAX_JOBS="$2"
            shift 2
            ;;
        -d|--directory)
            if [[ -z "${2:-}" ]]; then
                die "Missing argument for $1"
            fi
            if [[ ! -d "$2" ]]; then
                die "Directory '$2' does not exist"
            fi
            SEARCH_DIR="$2"
            shift 2
            ;;
        -t|--timeout)
            if [[ -z "${2:-}" ]]; then
                die "Missing argument for $1"
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                die "Timeout must be a positive integer"
            fi
            SLEEP_INTERVAL="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --version)
            version
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
check_commands

log "Starting with max_jobs=$MAX_JOBS, search_dir=$SEARCH_DIR, sleep=$SLEEP_INTERVAL"

# Collect directories containing gorun_ready
readarray -t ready_dirs < <(
    find "$SEARCH_DIR" -type f -name 'gorun_ready' -printf '%h\n' | sort -u
)

if [[ ${#ready_dirs[@]} -eq 0 ]]; then
    log "No 'gorun_ready' files found in $SEARCH_DIR"
    exit 3
fi

log "Found ${#ready_dirs[@]} directories with gorun_ready"

for dir in "${ready_dirs[@]}"; do
    log "Processing $dir"

    # Wait until job count drops below limit
    while [[ $(get_n_jobs) -ge "$MAX_JOBS" ]]; do
        log "Job count $(get_n_jobs) >= $MAX_JOBS, sleeping $SLEEP_INTERVAL seconds"
        sleep "$SLEEP_INTERVAL"
    done

    # Enter directory and submit
    (
        cd "$dir" || die "Cannot cd to $dir"

        if [[ ! -f "sub" ]]; then
            err "Missing 'sub' script in $dir; skipping"
            exit 0
        fi

        log "Submitting job in $dir"
        if sbatch sub; then
            rm -f gorun_ready
            log "Submitted and removed gorun_ready in $dir"
        else
            err "sbatch failed in $dir; keeping gorun_ready"
            exit 4
        fi
    )
done

log "All pending jobs submitted"
exit 0
