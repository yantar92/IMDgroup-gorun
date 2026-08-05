

# IMDgroup-gorun

This package provides a set of scripts for supercomputer job
submission for VASP, MACE multihead fine-tuning, and ATAT via
[Slurm](https://slurm.schedmd.com/). It is tailored to research performed in the
[Inverse Materials Design group](https://www.oimalyi.org/), creating a unified interface for
running calculations across different high-performance computing
clusters (e.g., Athena, Ares, Helios, LUMI).


# Installation

    git clone https://git.sr.ht/~yantar92/IMDgroup-gorun
    cd IMDgroup-gorun
    pip install .


# Configuration

The tool relies on specific environment variables and a TOML
configuration file to adapt to different cluster environments.


## Environment Variables

Add the following to your `.bashrc` or submission environment:

-   **`IMDGroup`:** Path to the directory containing configuration files
    (specifically `$IMDGroup/dist/etc/gorun.toml`).
-   **`VASP_PATH`:** Root directory of the VASP installation. The scripts
    expect binaries at `$VASP_PATH/bin/vasp_std`, etc.
-   **`VASP_PP_PATH`:** Path to VASP pseudopotentials (required for
    automatic POTCAR generation).
-   **`CLUSTER_NAME`:** (Optional) Manually override the cluster name
    detection (usually automatic via `uname -n`).


## Configuration File (`gorun.toml`)

The behavior of `gorun` is defined in a TOML file. This file maps
hostnames to cluster definitions and specifies available queues,
modules to load, and default resource limits.

Software-specific settings are namespaced under the server section.

    [cluster.names]
    lumi = ['uan01', 'uan02']
    
    [lumi]
    queues = ['standard', 'small']
    
    [lumi.vasp]
    setup = "module load LUMI/24.03 ..."
    mpiexec = "srun"
    
    [lumi.mace]
    setup = "module use /appl/local/laifs/modules && module load lumi-aif-singularity-bindings"
    sif = "/appl/local/laifs/containers/..."
    python_env = "/projappl/.../mace_env"
    
    [lumi.standard]
    type = 'CPU'
    partition = 'standard'
    max-nodes = 512
    max-time = '48:00:00'


# Command Line Interface


## `gorun vasp`

Submit a VASP job from the current directory.  The plain `gorun`
command (without subcommand) is treated as `gorun vasp`.  Positional
arguments specify the number of nodes and the time limit:

    # Submit job requesting 2 nodes for 24 hours
    gorun vasp 2 24:00:00
    # Shorthand:
    gorun 2 24:00:00
    
    # Use default nodes/time from config (positional args optional)
    gorun vasp


### Options

-   **`--config PATH`:** Path to TOML configuration file (default:
    `$IMDGroup/dist/etc/gorun.toml`).

-   **`--queue NAME`:** Submit to a specific queue (default: let the
    scheduler pick the one with the earliest finish time).

-   **`--vasp {ncl,gam,std}`:** Select the VASP binary variant (default:
    `ncl`).

-   **`--local`:** Run VASP directly on the login node instead of
    submitting via `sbatch`. Useful for testing short jobs.

-   **`--mark`:** Prepare the directory and create a `gorun_ready` marker
    file, but do not submit. Use `gorun-all-ready.sh` to submit
    multiple marked directories later (see [5](#org07ba5ff)).

-   **`--force`:** Skip convergence checks and run VASP even if the
    directory already contains converged output.

-   **`--keep FILE`:** Do not regenerate `FILE` if it already exists
    (repeatable).  For VASP, useful values are `POTCAR` and
    `POSCAR`. Example: `--keep POTCAR --keep POSCAR`.

-   **`--no-clean`:** Skip cleanup of old SLURM log files in NEB
    subdirectories.

-   **`--no-incar-py`:** Ignore `INCAR.py` even if present; run VASP
    directly.

-   **`--no-vasp-config`:** Do not source the cluster's VASP module setup
    (the `VASP-setup` line from the config). Use when the environment
    is already configured upstream.

-   **`--incar KEY:VAL ...`:** Modify INCAR parameters before
    submission. Space-separated `KEY:VAL` pairs. Set `VAL` to `None`
    to delete a key. Example:
    
        gorun vasp --incar ALGO:Normal NELM:200 NSW:50

-   **`--max_slurm_jobs N`:** Wait until the number of running Slurm jobs
    drops below N before submitting (default: 0 = no limit).

-   **`NODES`, `TIME`:** Positional arguments: number of nodes and time
    limit in `hh:mm:ss` format. Both are optional; defaults come from
    the config file.


### Examples

    # Submit to a specific GPU queue with 8 nodes for 48 hours
    gorun vasp --queue plgrid-gpu-a100 8 48:00:00
    
    # Use gamma-point-only VASP binary, 1 node, 1 hour
    gorun vasp --vasp gam 1 1:00:00
    
    # Run locally (no sbatch) for quick testing
    gorun vasp --local 1 0:30:00
    
    # Prepare directory for later batch submission
    gorun vasp --mark 4 24:00:00
    
    # Override INCAR settings before submission
    gorun vasp --incar ISIF:3 NSW:100 2 24:00:00
    
    # Force re-run even if already converged
    gorun vasp --force 2 24:00:00
    
    # Limit concurrent jobs to 20
    gorun vasp --max-slurm-jobs 20 2 24:00:00
    
    # Keep existing POTCAR (and use --keep for POSCAR too)
    gorun vasp --keep POTCAR --keep POSCAR


### Job Preparation Pipeline

Before submission, `gorun` runs the following steps in order:

1.  Safety checks :: Aborts if `INCAR` is missing, or if a job is
    already `RUNNING` or queued in the same directory.
2.  Convergence check :: If the directory contains converged output,
    exits (unless `--force` is set). If multiple sequential `INCAR.*`
    files exist, replaces `INCAR` with the next one and continues.
3.  Backup :: If VASP outputs already exist, copies the directory to
    `gorun_N` (incrementing N) and compresses the previous backup.
    `WAVECAR` is excluded from backups.
4.  INCAR modification :: If `--incar` was given, applies the changes
    now.
5.  Input sanitization :: Cleans `INCAR`, `POSCAR`, `KPOINTS`
    (newlines, BOMs, blank lines).
6.  POTCAR generation :: Generates `POTCAR` from `POSCAR` using ASE
    with VASP-recommended pseudopotentials (PBE functional).  Skips if
    `--keep_potcar` is set.
7.  vdW kernel :: Copies `vdw_kernel.bindat` from `$VASP_PATH` if the
    INCAR requests vdW corrections.
8.  Cleanup :: Removes old SLURM log files and zero-size VASP outputs.


### Smart Queue Selection

If no `--queue` is given, `gorun` calls `sbatch --test-only` for
every partition listed in the config. It estimates the finish time
for each option (waiting time + run time scaled by CPU count) and
selects the partition that finishes earliest.


### Python Scripting via `INCAR.py`

If `INCAR.py` is present in the directory and `--no_incar_py` is not
set, `gorun` wraps VASP execution inside a Python script using ASE.
The script has the calculator pre-configured and can access a
`IMDGVaspDir` object via the variable `vaspdir`:

    # Example INCAR.py contents
    # Access the pre-set ASE calculator
    from ase.constraints import FixAtoms
    atoms.set_constraint(FixAtoms(mask=[atom.symbol == 'Li' for atom in atoms]))
    energy = atoms.get_potential_energy()

The generated Slurm script exports `VASP_COMMAND` that points back to
`gorun vasp --local --no-incar-py --force --no-clean --keep POSCAR` for
the ASE calculator to invoke.


## `gorun mace-finetune` / `gorun mace`

Submit a MACE multihead fine-tuning job.  The canonical subcommand is
`mace-finetune`.  For backward compatibility, `gorun-mace-finetune`
and `gorun-mace` are also available as top-level entry points.

The workflow implements Method 1 from the MACE documentation &ndash; a
two-stage multihead replay protocol:

1.  `fine_tuning_select` — select replay configurations that best match
    the target dataset chemistry.
2.  `run_train` — train with a dual-head architecture (target head +
    pretraining head) to prevent catastrophic forgetting.

Both stages run sequentially in a single Slurm job.


### Custom protocol: RUNFILE

If a `RUNFILE.sh` or `RUNFILE.py` exists in the working directory, its
content replaces the canned two-stage workflow.  `RUNFILE.sh` takes
precedence over `RUNFILE.py`.  Both are wrapped so they inherit the
cluster environment (see [3.2](#org0cfb43e)).

Use this when the standard two-stage protocol does not fit your use
case &ndash; for example, when you want to run `mace.cli.eval` or a custom
training loop.


### Parameter file: INCAR.toml

All MACE parameters are stored in `INCAR.toml` in the working
directory, analogous to how VASP reads `INCAR`.  This avoids repeating
long CLI invocations and keeps the run configuration alongside the
data.

`INCAR.toml` uses TOML sections to route parameters to the correct
stage:

-   **Top-level:** gorun-specific keys (`model_path`, `replay_xyz`,
    `data_path`, `split_ratio`, `e0s`, etc.) and shared keys
    (`device`, `default_dtype`).
-   **`[fine_tuning_select]`:** parameters forwarded to Stage 1
    (`mace.cli.fine_tuning_select`): `num_samples`, `subselect`,
    `filtering_type`, `weight_pt`, `weight_ft`, etc.
-   **`[run_train]`:** parameters forwarded to Stage 2
    (`mace.cli.run_train`): `batch_size`, `lr`, `max_num_epochs`,
    `swa`, `ema`, and all their sub-options.

Keys the adapter does not recognise default to `[run_train]`, so
flat `INCAR.toml` files from earlier versions still work.

Example:

    model_path = "13_MACE-MATPES-PBE-0_medium.model"
    replay_xyz = "matpes-pbe-replay-data.xyz"
    data_path = "result.pkl"
    e0s = "{3:-0.01686124,6:-1.26246048,11:-0.22829243}"
    
    [fine_tuning_select]
    num_samples = 50000
    subselect = "fps"
    
    [run_train]
    batch_size = 12
    lr = 0.0009
    swa = true
    ema = true
    max_num_epochs = 200

Almost all MACE training hyperparameters live in `INCAR.toml`:
`batch_size`, `lr`, `weight_decay`, `max_num_epochs`, `swa`,
`swa_lr`, `start_swa`, `ema`, `ema_decay`, `energy_weight`,
`forces_weight`, `stress_weight`, `valid_fraction`, `compute_stress`,
`loss`, `force_mh_ft_lr`, `device`, `default_dtype`, etc.

When no `INCAR.toml` exists, `gorun mace-finetune` bootstraps a full
template with every parameter at its default value.  You can then edit
the file to adjust parameters.

**Precedence**: `--incar` overrides (highest) > explicit CLI flags >
`INCAR.toml` values > hardcoded adapter defaults.  For example, with
`INCAR.toml` setting `batch_size = 8` in `[run_train]`, running
`gorun mace-finetune` uses 8.  Adding `--incar run_train.batch_size:16`
overrides to 16.

**Write-back**: After parsing, if the merged parameters differ from the
existing file, `INCAR.toml` is updated (with the previous version
backed up to `INCAR.toml.old`).  Only parameters whose value differs
from the adapter default are written; the file stays sparse.

Non-adapter keys in `INCAR.toml` (e.g., `notes = "my experiment"`) are
preserved across writes.  Gorun-level flags (`force`, `local`,
`keep`, etc.) in `INCAR.toml` trigger a warning and are ignored.


### Options (gorun-specific)

Only workflow and data-source parameters are CLI flags.  All MACE
training hyperparameters live in `INCAR.toml` and can be overridden
via `--incar`.


### Data source

-   **`--model-path PATH`:** Path to the foundation model (`.model` file).
    Required unless set in `INCAR.toml`.

-   **`--replay-xyz PATH`:** Path to the replay/reference data (`.xyz`
    file).  Required unless set in `INCAR.toml`.

-   **`--data-path PATH`:** Single training file (`.xyz` or `.pkl`).
    Split into train/val using `--split-ratio` (default: `0.20`).
    Set in `INCAR.toml` or on the CLI.

-   **`--train-data-path PATH` + `--val-data-path PATH`:** Pre-split
    training and validation files (`.xyz` or `.pkl`).  Both must be
    provided together.  Set in `INCAR.toml` or on the CLI.

-   **`--split-ratio FLOAT`:** Fraction of `--data-path` to use for
    validation (default: `0.20`).

If neither `--data-path` nor the pre-split pair is given, the adapter
expects an existing `train.xyz` in the working directory (use
`--keep` to preserve it).


### Output & workflow

-   **`--new-model-name NAME`:** Name for the output fine-tuned model
    (default: `finetuned_model.model`).

-   **`--seed N`:** Random seed for train/val split and training
    (default: `1`).

-   **`--e0s STR`:** E0s string for `heads.json` (default: empty).

-   **`--no-fine-tuning-select`:** Skip Stage 1 (useful if
    `selected_configs.xyz` already exists).


### INCAR.toml overrides

-   **`--incar KEY:VAL ...`:** Override any `INCAR.toml` parameter from
    the command line.  Use dot notation to target a section:
    `--incar run_train.lr:0.001 fine_tuning_select.num_samples:50000`.
    Bare keys (no dot) default to `[run_train]`:
    `--incar batch_size:16`.

This is the primary mechanism for adjusting MACE training parameters
when running from the CLI.  For example, to override the learning
rate and number of epochs without editing `INCAR.toml`:
`--incar run_train.lr:0.0005 run_train.max_num_epochs:150`.


### Shared flags

-   **`--keep FILE`:** Do not regenerate `FILE` if it already exists
    (repeatable).  Useful values: `train.xyz`, `val.xyz`,
    `heads.json`.

-   **`--time_limit HH:MM:SS`:** Time limit (optional; defaults come
    from config).

-   **`--queue NAME`:** Submit to a specific queue (default:
    auto-select earliest finish).

-   **`--mark`:** Prepare directory and create a `gorun_ready` marker
    without submitting.

-   **`--local`:** Run directly on the login node (for testing).

-   **`--force`:** Skip convergence checks.

-   **`--max-slurm-jobs N`:** Wait until running Slurm jobs drop below
    N (default: 0 = no limit).


### Job Preparation Pipeline

Before submission, `gorun mace-finetune` runs the following steps:

1.  INCAR.toml :: Reads `INCAR.toml` (if present) and merges values
    with CLI arguments.  Bootstraps a template file if none exists.
    Writes back changes if parameters differ.
2.  Validation :: Checks that required parameters (`model_path`,
    `replay_xyz`) are set and that the foundation model and replay
    data exist.
3.  Data preparation :: Reads structures from `--data-path` or
    `--train-data-path` / `--val-data-path`, strips ASE constraints (preserving
    forces/energy/stress via `SinglePointCalculator`), splits if
    needed, writes `train.xyz` and `val.xyz` (unless `--keep` is
    set), and generates `heads.json`.
4.  Backup :: If previous outputs exist (logs, model files, checkpoints),
    backs up the directory to `gorun_N`.
5.  Submission :: Generates and submits the two-stage Slurm script
    (or runs the RUNFILE if present).


### Examples

    # Basic fine-tuning from a pickle file (auto-split 80-20)
    gorun mace-finetune \
        --model-path 13_MACE-MATPES-PBE-0_medium.model \
        --replay-xyz matpes-pbe-replay-data.xyz \
        --data-path result.pkl \
        --e0s "{3:-0.01686124,6:-1.26246048,11:-0.22829243}"
    
    # Pre-split train/val files
    gorun mace-finetune \
        --model-path foundation.model \
        --replay-xyz replay.xyz \
        --train-data-path train.xyz --val-data-path val.xyz
    
    # Skip the select stage, specify time limit
    gorun mace-finetune \
        --model-path foundation.model \
        --replay-xyz replay.xyz \
        --data-path result.pkl \
        --no-fine-tuning-select \
        48:00:00
    
    # Keep existing xyz files, override training params via --incar
    gorun mace-finetune \
        --model-path foundation.model \
        --replay-xyz replay.xyz \
        --data-path result.pkl \
        --keep train.xyz --keep val.xyz \
        --seed 42 \
        --incar run_train.batch_size:4 run_train.lr:0.0005
    
    # Prepare for later batch submission
    gorun mace-finetune \
        --model-path foundation.model \
        --replay-xyz replay.xyz \
        --data-path result.pkl \
        --mark
    
    # Use INCAR.toml for almost all parameters
    gorun mace-finetune 48:00:00
    # (model_path, replay_xyz, data_path, e0s, etc. read from INCAR.toml)
    
    # INCAR.toml provides defaults; override batch size from CLI
    gorun mace-finetune --incar run_train.batch_size:16
    
    # First run in a directory: bootstraps INCAR.toml with all defaults
    gorun mace-finetune --model-path foundation.model --replay-xyz replay.xyz --data-path result.pkl
    
    # Backward-compatible shorthand (gorun mace)
    gorun mace --model-path foundation.model --replay-xyz replay.xyz --data-path result.pkl
    
    # Backward-compatible entry points
    gorun-mace --model-path foundation.model --replay-xyz replay.xyz --data-path result.pkl
    gorun-mace-finetune --model-path foundation.model --replay-xyz replay.xyz --data-path result.pkl


## `gorun maps`

Wrapper to submit ATAT's `maps` (Cluster Expansion) code to Slurm.
Launches `maps` on the compute node and configures it to use
`gorun-atat-local` as the calculation script.


### Options

-   **`--kpoints DENSITY` (required):** K-point density for VASP
    sub-jobs.

-   **`--max_strain FRAC`:** Maximum allowed strain before a structure is
    flagged as an error (default: `0.1`).

-   **`--frac_tol FRAC`:** Distance tolerance for rejecting structures
    with atoms too close together (default: `0.5`).

-   **`--skip_relax`:** Skip the relaxation step in each sub-job (run
    only the SCF calculation).

-   **`--sublattice_cutoff DIST`:** Mark a structure as error if its
    sublattice deviation exceeds this value (default: off).

-   **`--time_limit HH:MM:SS`:** Time limit for the `maps` parent job
    (optional).

-   **`--queue NAME`:** Submit to a specific queue (default: auto-select
    earliest finish).

-   **`--config PATH`:** Path to configuration TOML (default:
    `$IMDGroup/dist/etc/gorun.toml`).

-   **`--gorun_command PATH`:** Path to the `gorun` executable used by
    `pollmach` (default: `gorun` from `$PATH`).

-   **`--local`:** Run `maps` directly on the login node instead of
    submitting via `sbatch`.

-   **`--maps_args ARGS`:** Additional arguments to pass through to the
    `maps` executable (space-separated).


### Examples

    # Basic cluster-expansion run
    gorun maps --kpoints=3000 --max_strain=0.1
    
    # Skip relaxation (SCF only), tighter strain tolerance
    gorun maps --kpoints=4000 --max_strain=0.05 --skip_relax
    
    # Custom sublattice deviation cutoff
    gorun maps --kpoints=3000 --sublattice_cutoff=0.15
    
    # Pass additional arguments to maps
    gorun maps --kpoints=3000 --maps_args="-e=4 -gs=0.001"


## `gorun atat-local`

Worker script invoked by ATAT's `pollmach` to run individual
structural calculations. Each call processes a single `str.out`
generated by `maps`, creating VASP inputs from the parent directory
as template.


### Options

-   **`--kpoints DENSITY` (required):** K-point density for the VASP run.

-   **`--frac_tol FRAC`:** Reject structures where any two atoms are
    closer than this tolerance (default: `0`, meaning no rejection by
    distance).

-   **`--max_strain FRAC`:** Reject structures whose volume change
    exceeds this fraction (default: `0.1`).

-   **`--skip_relax`:** Skip the ionic relaxation step. Only the SCF
    calculation is performed.

-   **`--sublattice_cutoff DIST`:** Flag structures whose sublattice
    deviation equals or exceeds this value. Writes
    `error_sublattice` marker (default: off).

-   **`VASP_COMMAND [ARGS]` (positional, remainder):** Command to invoke
    VASP. All remaining arguments are passed through verbatim.


### Validation Pipeline

Each sub-job goes through the following checks:

1.  Input derivation :: Reads `str.out` and the parent directory to
    create VASP input files (written into `./ATAT`).
2.  Distance check :: Rejects the structure if atoms are too close
    (writes `error_atoms_too_close`).
3.  K-point grid check :: Rejects if the grid has too few points along
    an axis (writes `error_kpoints_dim_sparse`).
4.  Relaxation :: Runs VASP in `./ATAT` (skipped with `--skip_relax`).
5.  Volume distortion check :: Rejects if the cell volume changed by
    more than `--max_strain` (writes `error_strain`).
6.  Sublattice check :: Detects sublattice flips. If a flip is found,
    attempts to fit the refined lattice and updates `str.out`. If
    deviation exceeds `--sublattice_cutoff` (writes
    `error_sublattice`).
7.  SCF :: Runs a static SCF calculation in `./ATAT.SCF`.
8.  Energy output :: Writes the final energy to a file named `energy`.


# Batch Submission

The package includes a companion Bash script `gorun-all-ready.sh` for
submitting multiple jobs in batch. This is useful when you have many
directories prepared with `gorun vasp --mark` and want to submit them while
respecting a limit on concurrent Slurm jobs.


## Options

-   **`-n NUM`, `--max-jobs NUM`:** Maximum number of concurrent Slurm
    jobs allowed (default: `100`).

-   **`-d DIR`, `--directory DIR`:** Root directory to search for
    `gorun_ready` markers (default: current directory).

-   **`-t SEC`, `--timeout SEC`:** Sleep interval (in seconds) between
    queue checks when waiting for jobs to finish (default: `5`).

-   **`-q`, `--quiet`:** Suppress informational output. Errors are still
    printed.

-   **`--version`:** Print version number and exit.


## Examples

    # Submit all ready directories, limit to 4 concurrent jobs
    gorun-all-ready.sh -n 4
    
    # Search in a specific project tree, 8 concurrent jobs
    gorun-all-ready.sh -d /path/to/project -n 8
    
    # Quiet mode, check queue every 30 seconds
    gorun-all-ready.sh -n 10 -t 30 -q


## How it works

1.  Recursively searches for `gorun_ready` marker files under the
    search directory.
2.  For each directory containing a marker, checks the current number
    of running/pending Slurm jobs (via `squeue -u $USER`).
3.  If the count is below the limit, runs `sbatch sub` in that
    directory and removes the marker.
4.  If the limit is reached, waits for `--timeout` seconds, then
    re-checks the queue.

Exit codes: 0 = success, 1 = argument error, 2 = missing command,
3 = no `gorun_ready` files found, 4 = `sbatch` failure.


# Acknowledgements

We acknowledge financial support from the National Centre for Research
and Development (NCBR) under project
WPC3/2022/50/KEYTECH/2024. Computational resources were provided by
the Polish high-performance computing infrastructure PLGrid, including
access to the LUMI supercomputer—owned by the EuroHPC Joint
Undertaking and hosted by CSC in Finland together with the LUMI
Consortium—through allocation PLL/2024/07/017633, as well as
additional resources at the PLGrid HPC centres ACK Cyfronet AGH and
WCSS under allocation PLG/2024/017498.

