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
from IMDgroup.gorun.adapters.mace import MaceFinetuneAdapter
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

_SUBCOMMANDS = {"vasp", "mace", "mace-finetune", "maps", "atat-local"}


def _is_subcommand(name: str) -> bool:
    return name in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with ``vasp`` and ``mace`` subcommands."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="software")

    # -- shared flags (parent parser reused by both subcommands) --
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
        "--keep", action="append", default=[],
        metavar="FILE",
        help="Do not regenerate FILE if it already exists (repeatable).",
    )

    # ---- vasp ----
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

    # ---- mace-finetune ----
    mace_parser = subparsers.add_parser(
        "mace-finetune",
        parents=[shared],
        help="Submit a MACE fine-tuning job.",
        description="Queue a MACE fine-tuning run from the current directory.",
    )
    mace_parser.add_argument(
        "time_limit", nargs="?", default=None,
        help="Time limit in HH:MM:SS (optional; defaults come from config).",
    )
    mace_parser.add_argument(
        "--model", required=True,
        help="Path to the foundation model (.model file).",
    )
    mace_parser.add_argument(
        "--replay-xyz", required=True,
        help="Path to the replay/reference data (.xyz file).",
    )
    mace_parser.add_argument(
        "--pkl", default="result.pkl",
        help="Path to the pickle file with Structures (default: result.pkl).",
    )
    mace_parser.add_argument(
        "--new-model-name", default="finetuned_model.model",
        help="Name for the output fine-tuned model (default: finetuned_model.model).",
    )
    mace_parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for train/val split and training (default: 1).",
    )
    mace_parser.add_argument(
        "--e0s", default="",
        help="E0s string for heads.json (default: empty).",
    )
    mace_parser.add_argument(
        "--batch-size", type=int, default=6,
        help="Training batch size (default: 6).",
    )
    mace_parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Maximum training epochs (default: 100).",
    )
    mace_parser.add_argument(
        "--num-samples", type=int, default=30000,
        help="Number of samples for fine_tuning_select (default: 30000).",
    )
    mace_parser.add_argument(
        "--no-fine-tuning-select", action="store_true",
        help="Skip the fine_tuning_select stage.",
    )

    # ---- maps ----
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

    # ---- atat-local ----
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

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse *argv*, handling backward-compatible plain ``gorun`` invocations."""

    if argv is None:
        argv = sys.argv[1:]

    # Normalize backward-compatible subcommand names.
    if argv and argv[0] == "mace":
        argv = ["mace-finetune"] + argv[1:]

    # If the first argument is a known subcommand, parse normally.
    if argv and _is_subcommand(argv[0]):
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


def _make_mace_adapter(args: argparse.Namespace) -> MaceFinetuneAdapter:
    """Build a MaceFinetuneAdapter from parsed CLI args."""
    return MaceFinetuneAdapter(
        model_path=args.model,
        replay_xyz=args.replay_xyz,
        pkl_path=args.pkl,
        new_model_name=args.new_model_name,
        seed=args.seed,
        e0s=args.e0s,
        batch_size=args.batch_size,
        max_num_epochs=args.max_epochs,
        num_samples=args.num_samples,
        no_fine_tuning_select=args.no_fine_tuning_select,
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


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse args, build adapter, and dispatch."""
    args = _parse_args(argv)

    # Subcommand not given?  Show help rather than failing silently.
    if not args.software:
        _build_parser().print_help()
        return 1

    # Build adapter
    if args.software == "vasp":
        adapter = _make_vasp_adapter(args)
    elif args.software == "mace-finetune":
        adapter = _make_mace_adapter(args)
    elif args.software == "maps":
        adapter = _make_maps_adapter(args)
    elif args.software == "atat-local":
        adapter = _make_atat_local_adapter(args)
    else:
        print(colored(f"Unknown software: {args.software}", "red"))
        return 1

    # Inject number_of_nodes=None for subcommands that don't use it
    # (get_sbatch_args needs the attribute to exist).
    if args.software in ("mace-finetune", "atat-local"):
        args.number_of_nodes = None

    return dispatch_run(args, adapter)


def main_mace(argv: list[str] | None = None) -> int:
    """Convenience entry point for ``gorun-mace`` (backward compat)."""
    if argv is None:
        argv = sys.argv[1:]
    return main(["mace-finetune"] + argv)


# Keep run() for backward compatibility with any external callers.
# It delegates to main().
run = main
