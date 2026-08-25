# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
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

"""Slurm job submission for VASP, MACE, and other backends.

Usage:
    gorun vasp [OPTS] [NODES] [TIME-LIMIT]
    gorun mace [OPTS] [TIME-LIMIT]
    gorun gpu [OPTS] [TIME-LIMIT]

For backward compatibility, plain ``gorun [OPTS] [NODES] [TIME-LIMIT]``
is treated as ``gorun vasp``.
"""

from __future__ import annotations

import argparse
import sys
import warnings

from termcolor import colored

from IMDgroup.gorun.core.pipeline import dispatch_run
from IMDgroup.gorun.adapters.vasp import VaspAdapter
from IMDgroup.gorun.adapters.mace import MaceMultiheadFinetuneAdapter
from IMDgroup.gorun.adapters.gpu import GpuAdapter
from IMDgroup.gorun.adapters.maps_adapter import MapsAdapter
from IMDgroup.gorun.adapters.atat_local import AtatLocalAdapter


# ---------------------------------------------------------------------------
# Cosmetic: colourise Python warnings
# ---------------------------------------------------------------------------

def _showwarning(message, category, _filename, _lineno, file=None, _line=None):
    output = (
        colored(f"{category.__name__}: ", "yellow", attrs=["bold"])
        + f"{message}"
    )
    print(output, file=file or sys.stderr)


warnings.showwarning = _showwarning

# ---------------------------------------------------------------------------
# Known subcommands
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {"vasp", "mace-finetune", "maps", "atat-local", "gpu"}


def _is_subcommand(name: str) -> bool:
    return name in _SUBCOMMANDS


# Argument parsing

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands.

    MACE-finetune arguments use ``default=argparse.SUPPRESS`` so that
    ``hasattr`` on the parsed namespace reveals whether a value came
    from the CLI.  The adapter layer (``_make_mace_adapter``) merges
    adapter defaults, INCAR.toml, and CLI overrides.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="software")

    ## shared flags (parent parser reused by all subcommands)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--force", action="store_true",
        help="Force running even if already converged or queued.",
    )
    shared.add_argument(
        "--local", action="store_true",
        help="Run directly on the login node instead of sbatch.",
    )
    shared.add_argument(
        "--mark", action="store_true",
        help="Prepare directory and create 'gorun_ready' but do not submit.",
    )
    shared.add_argument(
        "--max-slurm-jobs", type=int, default=0,
        metavar="N",
        help="Wait until running Slurm jobs drop below N (default: 0 = no limit).",
    )
    shared.add_argument(
        "--queue", type=str, default=None,
        help="Submit to a specific queue (default: auto-select).",
    )
    shared.add_argument(
        "--config", type=str, default=None,
        help="Path to TOML config (default: $IMDGroup/dist/etc/gorun.toml).",
    )
    shared.add_argument(
        "--keep", action="append", default=None,
        metavar="FILE",
        help="Do not regenerate FILE if it already exists (repeatable).",
    )

    ## vasp
    vasp_parser = subparsers.add_parser(
        "vasp",
        parents=[shared],
        help="Submit a VASP job.",
        description="Queue a VASP run from the current directory.",
    )
    vasp_parser.add_argument(
        "number_of_nodes", nargs="?", default=None,
        help="Number of nodes to request (optional; defaults come from config).",
    )
    vasp_parser.add_argument(
        "time_limit", nargs="?", default=None,
        help="Time limit in HH:MM:SS (optional; defaults come from config).",
    )
    vasp_parser.add_argument(
        "--vasp", choices=["ncl", "gam", "std"], default="ncl",
        help="VASP binary variant (default: ncl).",
    )
    vasp_parser.add_argument(
        "--no-incar-py", action="store_true",
        help="Ignore INCAR.py even if present.",
    )
    vasp_parser.add_argument(
        "--no-vasp-config", action="store_true",
        help="Do not source cluster VASP module setup.",
    )
    vasp_parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip cleanup of old slurm logs in NEB subdirectories.",
    )
    vasp_parser.add_argument(
        "--incar", nargs="+", default=None,
        metavar="KEY:VAL",
        help="Modify INCAR before submitting (e.g. --incar ALGO:Normal NELM:200).",
    )
    # Legacy flags mapped to --keep
    vasp_parser.add_argument(
        "--keep-potcar", action="store_true",
        help=argparse.SUPPRESS,
    )
    vasp_parser.add_argument(
        "--keep-poscar", action="store_true",
        help=argparse.SUPPRESS,
    )

    ## mace-finetune
    mace_parser = subparsers.add_parser(
        "mace-finetune",
        parents=[shared],
        help="Submit a MACE multihead fine-tuning job.",
        description="Queue a MACE multihead fine-tuning run from the current directory.",
        epilog=(
            "MACE training hyperparameters:\n"
            "  Set batch_size, lr, device, swa, ema, max_num_epochs, and any\n"
            "  other mace.cli.run_train flag in INCAR.toml under [run_train]\n"
            "  or [fine_tuning_select].  Override via --incar:\n"
            "\n"
            "    --incar run_train.lr:0.001 fine_tuning_select.num_samples:50000\n"
            "\n"
            "  Dot notation routes to the matching TOML section.\n"
            "  Bare keys (no dot) default to run_train.\n"
            "\n"
            "Escape hatch:\n"
            "  Place RUNFILE.sh or RUNFILE.py in the working directory to\n"
            "  override the canned two-stage workflow.  RUNFILE.sh takes\n"
            "  precedence over RUNFILE.py."
        ),
    )
    mace_parser.add_argument(
        "time_limit", nargs="?", default=None,
        help="Time limit in HH:MM:SS (optional; defaults come from config).",
    )
    ## Required
    mace_parser.add_argument(
        "--model-path", default=argparse.SUPPRESS,
        help="Path to the foundation model (.model file).  "
        "Required unless set in INCAR.toml.",
    )
    mace_parser.add_argument(
        "--replay-xyz", default=argparse.SUPPRESS,
        help="Path to the replay/reference data (.xyz file).  "
        "Required unless set in INCAR.toml.",
    )
    ## Data source
    data_group = mace_parser.add_argument_group("data source (pick one)")
    data_group.add_argument(
        "--data-path", default=argparse.SUPPRESS,
        help="Training data file (.xyz or .pkl).  Split into train/val using --split-ratio.",
    )
    data_group.add_argument(
        "--train-data-path", default=argparse.SUPPRESS,
        help="Pre-split training file (.xyz or .pkl).  Requires --val-data-path.",
    )
    data_group.add_argument(
        "--val-data-path", default=argparse.SUPPRESS,
        help="Pre-split validation file (.xyz or .pkl).  Requires --train-data-path.",
    )
    data_group.add_argument(
        "--split-ratio", type=float, default=argparse.SUPPRESS,
        help="Fraction of --data-path to use for validation (default: 0.20).",
    )
    ## Output
    mace_parser.add_argument(
        "--new-model-name", default=argparse.SUPPRESS,
        help="Name for the output fine-tuned model (default: finetuned_model.model).",
    )
    mace_parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS,
        help="Random seed for train/val split (default: 1).",
    )
    mace_parser.add_argument(
        "--e0s", default=argparse.SUPPRESS,
        help="E0s string for heads.json (default: empty).",
    )
    ## Workflow
    mace_parser.add_argument(
        "--no-fine-tuning-select", action="store_true",
        default=argparse.SUPPRESS,
        help="Skip the fine_tuning_select stage.",
    )
    mace_parser.add_argument(
        "--masked-loss", action="store_true",
        default=argparse.SUPPRESS,
        help="Use mask-aware loss that detaches masked atoms' site energies "
        "and atomic stresses from autograd.  Atoms with atom_mask=0 in the "
        "batch are excluded from direct energy/stress supervision but still "
        "contribute via neighbour message passing.  Falls back to standard "
        "loss when atom_mask is absent.",
    )
    mace_parser.add_argument(
        "--incar", nargs="+", default=None,
        metavar="KEY:VAL",
        help="Override INCAR.toml parameters.  "
        "Use SECTION.KEY:VAL to target a section "
        "(e.g. --incar run_train.lr:0.001 "
        "fine_tuning_select.num_samples:50000).  "
        "Bare keys default to run_train.",
    )

    ## maps
    maps_parser = subparsers.add_parser(
        "maps",
        parents=[shared],
        help="Submit a maps cluster expansion job.",
        description="Queue an ATAT maps run from the current directory.",
    )
    maps_parser.add_argument(
        "number_of_nodes", nargs="?", default=None,
        help="Number of nodes (default: 1).",
    )
    maps_parser.add_argument(
        "time_limit", nargs="?", default=None,
        help="Time limit in HH:MM:SS (optional).",
    )
    maps_parser.add_argument(
        "--kpoints", required=True,
        help="Kpoint density.",
    )
    maps_parser.add_argument(
        "--frac-tol", type=float, default=0.5,
        help="Distance tolerance for rejecting structures (default: 0.5).",
    )
    maps_parser.add_argument(
        "--max-strain", type=float, default=0.1,
        help="Maximum strain allowed (default: 0.1).",
    )
    maps_parser.add_argument(
        "--skip-relax", action="store_true",
        help="Skip the relaxation run in ATAT.",
    )
    maps_parser.add_argument(
        "--sublattice-cutoff", type=float, default=None,
        help="Maximum allowed sublattice deviation.",
    )
    maps_parser.add_argument(
        "--maps-args", nargs="+", default=[],
        help="Extra arguments to pass to the maps command.",
    )

    ## atat-local
    atat_local_parser = subparsers.add_parser(
        "atat-local",
        parents=[shared],
        help="Process a single ATAT structure (called by pollmach).",
        description="Run VASP on an ATAT-generated structure in the current directory.",
    )
    atat_local_parser.add_argument(
        "--kpoints", required=True,
        help="Kpoint density.",
    )
    atat_local_parser.add_argument(
        "--frac-tol", type=float, default=0,
        help="Distance tolerance for rejecting structures (default: 0).",
    )
    atat_local_parser.add_argument(
        "--max-strain", type=float, default=0.1,
        help="Maximum strain allowed (default: 0.1).",
    )
    atat_local_parser.add_argument(
        "--skip-relax", action="store_true",
        help="Skip the relaxation run in ATAT.",
    )
    atat_local_parser.add_argument(
        "--sublattice-cutoff", type=float, default=None,
        help="Maximum allowed sublattice deviation.",
    )

    ## gpu
    gpu_parser = subparsers.add_parser(
        "gpu",
        parents=[shared],
        help="Submit a generic GPU job (requires RUNFILE.sh or RUNFILE.py).",
        description="Queue a GPU run defined by RUNFILE.sh or RUNFILE.py.",
        epilog=(
            "Place RUNFILE.sh or RUNFILE.py in the working directory to\n"
            "define the computation.  RUNFILE.sh takes precedence over\n"
            "RUNFILE.py.  Use --script-name to use a different file.\n"
            "\n"
            "GPU environment (module loads, PyTorch flags, etc.) is read\n"
            "from the server config under [gpu] → setup."
        ),
    )
    gpu_parser.add_argument(
        "time_limit", nargs="?", default=None,
        help="Time limit in HH:MM:SS (optional; defaults come from config).",
    )
    gpu_parser.add_argument(
        "--script-name", default=None,
        help="Script to run instead of RUNFILE.sh/py.  .py → python -u; "
        "otherwise → bash.",
    )

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse *argv*, handling backward-compatible plain ``gorun`` invocations."""

    if argv is None:
        argv = sys.argv[1:]

    # If the first argument is a known subcommand, parse normally.
    if argv and _is_subcommand(argv[0]):
        return _build_parser().parse_args(argv)

    # If --help/-h is present without a subcommand, show top-level help
    # (avoids the backward-compat fallback that would show vasp help).
    if "--help" in argv or "-h" in argv:
        return _build_parser().parse_args(argv)

    # Otherwise, treat as ``gorun vasp <argv>`` for backward compat.
    return _build_parser().parse_args(["vasp"] + argv)


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def _make_vasp_adapter(args: argparse.Namespace) -> VaspAdapter:
    """Build a VaspAdapter from parsed CLI args."""
    # --incar parsing: "ALGO:Normal NELM:200" -> {"ALGO": "Normal", "NELM": "200"}
    incar_mods = None
    if args.incar:
        incar_mods = {}
        for item in args.incar:
            if ":" not in item:
                print(
                    colored(
                        f"Invalid --incar value '{item}'.  Expected KEY:VAL format.",
                        "red",
                    )
                )
                sys.exit(1)
            key, val = item.split(":", 1)
            incar_mods[key] = val

    # Backward compat: --keep-potcar / --keep-poscar → --keep
    if args.keep is None:
        args.keep = []
    if args.keep_potcar and "POTCAR" not in args.keep:
        args.keep.append("POTCAR")
    if args.keep_poscar and "POSCAR" not in args.keep:
        args.keep.append("POSCAR")

    return VaspAdapter(
        vasp_variant=args.vasp,
        no_incar_py=args.no_incar_py,
        no_vasp_config=args.no_vasp_config,
        no_clean=args.no_clean,
        incar_mods=incar_mods,
    )


def _make_mace_adapter(args: argparse.Namespace) -> MaceMultiheadFinetuneAdapter:
    """Build a MaceMultiheadFinetuneAdapter from CLI args and INCAR.toml.

    Parses ``--incar`` into a structured dict mapping section name
    (or ``None`` for auto-routing) to ``{key: val_str}``.  The
    adapter constructor reads INCAR.toml internally and handles
    merge, bootstrap, and write-back.
    """
    # Parse --incar overrides: SECTION.KEY:VAL or KEY:VAL
    incar_overrides: dict[str | None, dict[str, str]] = {}
    if args.incar:
        for item in args.incar:
            if ':' not in item:
                print(colored(
                    f"Invalid --incar value '{item}'.  "
                    "Expected KEY:VAL format.",
                    "red",
                ))
                sys.exit(1)
            key_path, val_str = item.split(':', 1)
            if '.' in key_path:
                section, key = key_path.rsplit('.', 1)
            else:
                section, key = None, key_path
            incar_overrides.setdefault(section, {})[key] = val_str

    return MaceMultiheadFinetuneAdapter(
        model_path=getattr(args, 'model_path', None),
        replay_xyz=getattr(args, 'replay_xyz', None),
        data_path=getattr(args, 'data_path', None),
        train_data_path=getattr(args, 'train_data_path', None),
        val_data_path=getattr(args, 'val_data_path', None),
        split_ratio=getattr(args, 'split_ratio', None),
        new_model_name=getattr(args, 'new_model_name', None),
        seed=getattr(args, 'seed', None),
        e0s=getattr(args, 'e0s', None),
        no_fine_tuning_select=getattr(args, 'no_fine_tuning_select', None),
        masked_loss=getattr(args, 'masked_loss', None),
        incar_overrides=incar_overrides,
    )


def _make_maps_adapter(args: argparse.Namespace) -> MapsAdapter:
    """Build a MapsAdapter from parsed CLI args."""
    if args.number_of_nodes is None:
        args.number_of_nodes = 1
    return MapsAdapter(
        kpoints=args.kpoints,
        frac_tol=args.frac_tol,
        max_strain=args.max_strain,
        skip_relax=args.skip_relax,
        sublattice_cutoff=args.sublattice_cutoff,
        maps_args=args.maps_args,
    )


def _make_atat_local_adapter(args: argparse.Namespace) -> AtatLocalAdapter:
    """Build an AtatLocalAdapter from parsed CLI args."""
    return AtatLocalAdapter(
        kpoints=args.kpoints,
        frac_tol=args.frac_tol,
        max_strain=args.max_strain,
        skip_relax=args.skip_relax,
        sublattice_cutoff=args.sublattice_cutoff,
    )


def _make_gpu_adapter(args: argparse.Namespace) -> GpuAdapter:
    """Build a GpuAdapter from parsed CLI args."""
    return GpuAdapter(script_name=args.script_name)


def _fill_namespace_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing attributes in *args* with subcommand defaults.

    Pre-rewrite ``gorun.run`` accepted an ``argparse.Namespace`` and
    merged it with parser defaults.  The subcommand is read from
    ``args.software``; when it is absent or not a registered
    subcommand, ``vasp`` is used.
    """
    software = getattr(args, "software", None) or "vasp"
    if software not in _SUBCOMMANDS:
        software = "vasp"

    argv = [software]
    # ``maps`` and ``atat-local`` require ``--kpoints``; supply an empty
    # placeholder so the bare subcommand parses.  A caller that already
    # set ``kpoints`` is unaffected (we only fill missing attributes).
    if software in ("maps", "atat-local"):
        argv += ["--kpoints", ""]

    defaults = _build_parser().parse_args(argv)
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.software = software
    return args


def _dispatch_namespace(args: argparse.Namespace) -> int:
    """Build the adapter for *args* and run the dispatch pipeline."""
    if not args.software:
        _build_parser().print_help()
        return 1

    if args.software == "vasp":
        adapter = _make_vasp_adapter(args)
    elif args.software == "mace-finetune":
        adapter = _make_mace_adapter(args)
    elif args.software == "maps":
        adapter = _make_maps_adapter(args)
    elif args.software == "atat-local":
        adapter = _make_atat_local_adapter(args)
    elif args.software == "gpu":
        adapter = _make_gpu_adapter(args)
    else:
        print(colored(f"Unknown software: {args.software}", "red"))
        return 1

    # Inject number_of_nodes=None for subcommands that don't use it
    # (get_sbatch_args needs the attribute to exist).
    if args.software in ("mace-finetune", "atat-local", "gpu"):
        args.number_of_nodes = None

    return dispatch_run(args, adapter)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse args, build adapter, and dispatch."""
    if argv is None:
        argv = sys.argv[1:]

    return _dispatch_namespace(_parse_args(argv))


def main_mace(argv: list[str] | None = None) -> int:
    """Convenience entry point for ``gorun-mace`` (backward compat)."""
    if argv is None:
        argv = sys.argv[1:]
    return main(["mace-finetune"] + argv)


def run(args=None) -> int:
    """Run gorun from a Namespace or argv list.

    Pre-rewrite callers passed an ``argparse.Namespace`` (for example
    ``gorun.run(argparse.Namespace(mark=True))``).  Missing attributes
    are filled from the subcommand defaults selected by
    ``args.software`` (``vasp`` when absent or unknown).  Calling with
    no arguments is equivalent to an empty Namespace (the old default).
    A list of command-line arguments delegates to :func:`main`.
    """
    if args is None:
        args = argparse.Namespace()
    if isinstance(args, argparse.Namespace):
        return _dispatch_namespace(_fill_namespace_defaults(args))
    return main(args)
