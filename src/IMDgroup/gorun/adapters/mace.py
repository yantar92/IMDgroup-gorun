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

"""MACE multihead fine-tuning adapter for gorun.

Implements the two-stage multihead replay fine-tuning protocol
(Method 1 from the MACE documentation):

1. ``fine_tuning_select`` -- select replay configurations that best
   match the target dataset chemistry.
2. ``run_train`` -- train with a dual-head architecture (target head +
   pretraining head) to prevent catastrophic forgetting.

Place a ``RUNFILE.sh`` or ``RUNFILE.py`` in the working directory to
override the canned stages with a custom script.

**MACE CLI passthrough**

Only gorun-specific parameters (data sources, output naming, workflow
control) are exposed as CLI flags and stored in ``INCAR.toml``.
All other MACE training hyperparameters -- ``batch_size``, ``lr``,
``device``, ``max_num_epochs``, ``swa``, ``ema``, etc. -- are
forwarded directly to ``mace.cli.run_train`` and
``mace.cli.fine_tuning_select`` via ``INCAR.toml``.  Add any key
that MACE CLI accepts and gorun will pass it through as
``--key value``.  For example::

    # INCAR.toml
    model_path = "/path/to/foundation.model"
    replay_xyz = "/path/to/replay.xyz"
    data_path = "structures.xyz"
    batch_size = 12
    device = "cpu"
    swa = true
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from sklearn.model_selection import train_test_split
from termcolor import colored

from IMDgroup.gorun.core.files import maybe_regenerate
from IMDgroup.gorun.core.incar import read_incar_toml, write_incar_toml


# ---------------------------------------------------------------------------
# Wrapper helper
# ---------------------------------------------------------------------------

def _wrap_command(command: str) -> str:
    """Wrap *command* in a heredoc if ``GORUN_WRAPPER`` is set.

    When ``GORUN_WRAPPER`` is non-empty::

        if [ -n "${GORUN_WRAPPER:-}" ]; then
          ${GORUN_WRAPPER} <<'EOF'
        ${GORUN_INNER_SETUP:-}
        <command>
        EOF
        else
          <command>
        fi

    ``GORUN_WRAPPER`` must include the shell executable
    (e.g. ``bash``, ``/usr/bin/bash``).  The heredoc is quoted
    (``<<'EOF'``) so the outer bash does not expand
    ``${GORUN_INNER_SETUP}`` before passing it through; the inner
    shell expands it.  Both variables are expected to be exported by
    the config's ``setup`` commands.
    """
    return (
        'if [ -n "${GORUN_WRAPPER:-}" ]; then\n'
        "  ${GORUN_WRAPPER} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        f"{command}\n"
        "EOF\n"
        "else\n"
        f"{command}\n"
        "fi"
    )


# ---------------------------------------------------------------------------
# Data reading helpers
# ---------------------------------------------------------------------------

def _read_structures(filepath: str) -> list:
    """Read ASE Atoms from *filepath*, auto-detecting format.

    Supports ``.xyz`` (extxyz) and ``.pkl`` (pandas DataFrame with a
    ``structure`` column).
    """
    suffix = Path(filepath).suffix.lower()
    if suffix == ".pkl":
        df = pd.read_pickle(filepath)
        return list(df.structure)
    if suffix == ".xyz":
        return read(filepath, index=":")
    raise ValueError(
        f"Unsupported file format '{suffix}' for {filepath}.  "
        "Expected .xyz or .pkl."
    )


def _strip_constraints(structures: list) -> int:
    """Remove ASE constraints from *structures* in place.

    Forces, energy, and stress are preserved via
    ``SinglePointCalculator`` before the constraints are deleted.

    Returns the number of structures that had constraints removed.
    """
    count = 0
    for atoms in structures:
        if len(atoms.constraints) == 0:
            continue
        forces_raw = atoms.get_forces(apply_constraint=False)
        energy_raw = atoms.get_potential_energy()
        stress_raw = atoms.get_stress(apply_constraint=False)
        del atoms.constraints
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy_raw,
            forces=forces_raw,
            stress=stress_raw,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Command formatting
# ---------------------------------------------------------------------------

def _format_cli_args(args: dict[str, object]) -> str:
    """Format a dict as ``--key value \\`` lines.

    Boolean ``True`` values produce ``--key`` (flag without value).
    Boolean ``False`` values are silently skipped (they are the
    default for most CLI flags and passing ``--no-key`` is rarely
    supported).
    """
    if not args:
        return ""
    lines: list[str] = []
    for key, val in args.items():
        cli_key = key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                lines.append(f"  --{cli_key} \\")
        else:
            lines.append(f"  --{cli_key} {val} \\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MaceMultiheadFinetuneAdapter
# ---------------------------------------------------------------------------


class MaceMultiheadFinetuneAdapter:
    """``gorun mace-finetune`` backend for multihead fine-tuning.

    Two-stage workflow (override with ``RUNFILE.sh`` or ``RUNFILE.py``):

    1. ``mace.cli.fine_tuning_select`` -- select replay configurations.
    2. ``mace.cli.run_train`` -- dual-head training.

    Data sources are specified via one of:

    - ``--data-path`` + ``--split-ratio``: single file (.xyz or .pkl),
      auto-split.
    - ``--train-data-path`` + ``--val-data-path``: pre-split files.

    MACE training hyperparameters (``batch_size``, ``lr``, ``swa``,
    ``device``, etc.) are not hard-coded here.  Set them in
    ``INCAR.toml`` and they are forwarded directly to the MACE CLI
    commands as ``--key value`` flags.
    """

    name = "mace"
    log_file = "mace.out"

    #: Hardcoded defaults for gorun-specific parameters.
    #: Used for INCAR.toml bootstrapping and sparse write-back.
    DEFAULTS: dict[str, object] = {
        # Data sources
        "model_path": "",
        "replay_xyz": "",
        "data_path": None,
        "train_data_path": None,
        "val_data_path": None,
        "split_ratio": 0.20,
        # Output
        "new_model_name": "finetuned_model.model",
        "seed": 1,
        "e0s": "",
        # Workflow
        "no_fine_tuning_select": False,
    }

    #: Parameters that must be non-empty for a valid run.
    REQUIRED: frozenset[str] = frozenset({"model_path", "replay_xyz"})

    # pylint: disable=too-many-locals
    def __init__(
        self, *,
        # Data sources --------------------------------------------------
        model_path: str,
        replay_xyz: str,
        data_path: str | None = None,
        train_data_path: str | None = None,
        val_data_path: str | None = None,
        split_ratio: float = 0.20,
        # Output --------------------------------------------------------
        new_model_name: str = "finetuned_model.model",
        seed: int = 1,
        e0s: str = "",
        # Workflow ------------------------------------------------------
        no_fine_tuning_select: bool = False,
        # Passthrough ---------------------------------------------------
        **kwargs: object,
    ) -> None:
        # Data
        self.model_path = model_path
        self.replay_xyz = replay_xyz
        self.data_path = data_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.split_ratio = split_ratio
        # Output
        self.new_model_name = new_model_name
        self.seed = seed
        self.e0s = e0s
        # Workflow
        self.no_fine_tuning_select = no_fine_tuning_select

        #: MACE CLI flags to forward to ``run_train`` and
        #: ``fine_tuning_select`` as ``--key value``.  Populated from
        #: extra ``**kwargs`` passed to the constructor.
        self._passthrough_args: dict[str, object] = dict(kwargs)

    # ------------------------------------------------------------------
    # Construction from CLI + INCAR.toml
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_and_toml(cls, args) -> "MaceMultiheadFinetuneAdapter":
        """Build adapter from CLI args and INCAR.toml, writing back merged values.

        Merge order: adapter defaults -> INCAR.toml -> explicit CLI flags.
        TOML keys unknown to the adapter are forwarded as passthrough
        ``--key value`` flags to ``run_train``.  After construction the
        merged values are written back to ``INCAR.toml``.

        When ``INCAR.toml`` does not exist and required parameters are
        missing, a reference file is written and the program exits.
        """
        raw_toml = read_incar_toml()
        defaults = cls.DEFAULTS

        # Build init kwargs: defaults < TOML < CLI
        init_kwargs: dict[str, object] = dict(defaults)
        for key in defaults:
            if key in raw_toml:
                init_kwargs[key] = raw_toml[key]
            if hasattr(args, key):
                init_kwargs[key] = getattr(args, key)

        # Route remaining TOML keys: gorun-level -> warning, unknown -> passthrough
        gorun_dests = frozenset({
            'software', 'force', 'local', 'mark', 'max_slurm_jobs',
            'queue', 'config', 'keep', 'time_limit', 'number_of_nodes',
        })
        gorun_in_toml: list[str] = []
        for key, val in raw_toml.items():
            if key in defaults:
                continue
            if key in gorun_dests:
                gorun_in_toml.append(key)
                continue
            init_kwargs[key] = val

        if gorun_in_toml:
            print(colored(
                'INCAR.toml: ignoring gorun-level keys: '
                + ', '.join(sorted(gorun_in_toml)),
                'yellow',
            ))

        adapter = cls(**init_kwargs)

        # Bootstrap: no INCAR.toml, no MACE CLI args, required params still empty.
        # Write a reference INCAR.toml and exit.
        cli_mace_args = any(hasattr(args, key) for key in defaults)
        if not raw_toml and not cli_mace_args:
            missing_required = [attr for attr in adapter.REQUIRED
                               if not getattr(adapter, attr, None)]
            if missing_required:
                merged = {key: getattr(adapter, key) for key in defaults}
                write_incar_toml(
                    merged, raw_existing=None, defaults=defaults,
                    always_include=adapter.REQUIRED,
                )
                print(colored(
                    "Edit INCAR.toml to set the required parameters: "
                    + ", ".join(missing_required),
                    "cyan",
                ))
                sys.exit(0)

        adapter.validate()

        # Write back merged values
        merged = {key: getattr(adapter, key) for key in defaults}
        write_incar_toml(merged, raw_existing=raw_toml, defaults=defaults)

        return adapter

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check that required parameters are set.

        Raises ``ValueError`` when a required parameter is missing
        or empty.
        """
        missing = []
        for attr in self.REQUIRED:
            val = getattr(self, attr, None)
            if val is None or val == "":
                missing.append(attr)
        if missing:
            raise ValueError(
                "Missing required MACE parameters: "
                + ", ".join(missing)
                + ".  Provide them via CLI or INCAR.toml."
            )
        if self.data_path is None and self.train_data_path is None:
            print(
                colored(
                    "No data source specified.  Provide --data-path or "
                    "--train-data-path/--val-data-path via CLI or INCAR.toml, "
                    "or place an existing train.xyz in the directory.",
                    "yellow",
                )
            )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        """Check that required external files exist."""
        missing = []
        for label, path in [
            ("foundation model", self.model_path),
            ("replay data", self.replay_xyz),
        ]:
            if not Path(path).exists():
                missing.append(f"{label} ({path})")
        if missing:
            raise FileNotFoundError(
                "Required files not found: " + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # Directory inspection
    # ------------------------------------------------------------------

    def is_valid_input(self, path: Path) -> bool:
        """Return True when at least one data source file exists."""
        sources = [
            p for p in (self.data_path, self.train_data_path, self.val_data_path)
            if p is not None
        ]
        return any(Path(p).is_file() for p in sources)

    def is_converged(self, path: Path) -> bool:
        """True when the output model file exists and is non-empty."""
        model = path / self.new_model_name
        return model.is_file() and model.stat().st_size > 0

    def has_previous_output(self, path: Path) -> bool:
        """True when any output from a prior run exists."""
        markers = [
            path / "log_tuning",
            path / "log_db",
            path / "selected_configs.xyz",
            path / f"{self.new_model_name}.staged",
        ]
        model = path / self.new_model_name
        if model.is_file():
            return True
        return any(m.is_file() for m in markers)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        path: Path, *,
        keep: set[str] | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        if keep is None:
            keep = set()

        train_xyz = path / "train.xyz"
        val_xyz = path / "val.xyz"
        keep_train = "train.xyz" in keep and train_xyz.is_file()
        keep_val = "val.xyz" in keep and val_xyz.is_file()

        if keep_train and keep_val:
            print("Keeping existing train.xyz")
            print("Keeping existing val.xyz")
        else:
            if not keep_train and train_xyz.is_file():
                print("Regenerating train.xyz")
            if not keep_val and val_xyz.is_file():
                print("Regenerating val.xyz")
            self._generate_train_val(path)

        # heads.json
        maybe_regenerate(
            path, "heads.json", keep,
            regenerate=lambda: self._write_heads_json(path),
        )

    def _generate_train_val(self, path: Path) -> None:
        """Read structures from data source(s), strip constraints,
        split if needed, and write ``train.xyz`` and ``val.xyz``.
        """
        if self.train_data_path is not None and self.val_data_path is not None:
            train = self._read_and_clean(self.train_data_path)
            val = self._read_and_clean(self.val_data_path)
        elif self.data_path is not None:
            structures = self._read_and_clean(self.data_path)
            train, val = train_test_split(
                structures, test_size=self.split_ratio,
                random_state=self.seed,
            )
        else:
            print(
                "No data source specified "
                "(--data-path or --train-data-path/--val-data-path)."
            )
            if not (path / "train.xyz").is_file():
                raise FileNotFoundError(
                    "No data source provided and train.xyz does not exist.  "
                    "Specify --data-path or --train-data-path/--val-data-path."
                )
            return

        train_xyz = path / "train.xyz"
        val_xyz = path / "val.xyz"
        write(str(train_xyz), train, format="extxyz")
        write(str(val_xyz), val, format="extxyz")
        print(f"  train: {len(train)} structures → {train_xyz}")
        print(f"  val:   {len(val)} structures → {val_xyz}")

    def _read_and_clean(self, filepath: str) -> list:
        """Read structures from *filepath* and strip constraints."""
        print(f"Reading {filepath} ...")
        structures = _read_structures(filepath)
        print(f"  {len(structures)} structures loaded")
        n_constrained = _strip_constraints(structures)
        if n_constrained:
            print(f"  cleaned constraints from {n_constrained} structures")
        return structures

    def _write_heads_json(self, path: Path) -> None:
        """Write ``heads.json`` referencing train/val xyz files."""
        heads = {
            "default": {
                "train_file": str(path / "train.xyz"),
                "valid_file": str(path / "val.xyz"),
                "E0s": self.e0s,
                "energy_key": "energy",
                "forces_key": "forces",
                "stress_key": "stress",
            },
            "pt_head": {
                "train_file": str(path / "selected_configs.xyz"),
                "E0s": "foundation",
                "energy_key": "energy",
                "forces_key": "forces",
                "stress_key": "stress",
            },
        }
        heads_json = path / "heads.json"
        with open(heads_json, "w") as file:
            json.dump(heads, file, indent=2)
        print(f"  wrote {heads_json}")

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def setup_commands(self, server_config: dict) -> str:
        """Extract MACE environment from config.

        Reads ``server_config["mace"]["setup"]``, a raw shell string
        that must export ``GORUN_WRAPPER`` and
        ``GORUN_INNER_SETUP``.  All other environment variables
        (PyTorch flags, cache directories, ``OMP_NUM_THREADS``,
        etc.) belong in this string as well -- nothing is hard-coded
        here.
        """
        return server_config.get(self.name, {}).get("setup", "")

    def generate_run_commands(self, path: Path) -> str:
        """Return a two-stage bash body, or a RUNFILE if present.

        If ``RUNFILE.sh`` or ``RUNFILE.py`` exists in *path*, its
        content is used as the command instead of the canned stages.
        ``RUNFILE.sh`` takes precedence over ``RUNFILE.py``.  Both
        are wrapped through ``_wrap_command``, so they get
        ``GORUN_WRAPPER`` / ``GORUN_INNER_SETUP`` support.

        Without a RUNFILE, the standard two-stage fine-tuning
        workflow is generated.
        """
        runfile_sh = path / "RUNFILE.sh"
        runfile_py = path / "RUNFILE.py"

        if runfile_sh.is_file():
            return _wrap_command(f"bash {runfile_sh}")

        if runfile_py.is_file():
            return _wrap_command(f"python -u {runfile_py}")

        parts = []
        if not self.no_fine_tuning_select:
            parts.append(_wrap_command(self._select_stage(path)))
        parts.append(_wrap_command(self._train_stage(path)))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Stage commands (private)
    # ------------------------------------------------------------------

    def _select_stage(self, path: Path) -> str:
        """Stage 1: ``fine_tuning_select``.

        Reads gorun-specific arguments directly; all other parameters
        (``num_samples``, ``subselect``, ``filtering_type``,
        ``weight_pt``, ``weight_ft``, ``device``) are pulled from
        ``_passthrough_args`` with sensible defaults.
        """
        args = dict(self._passthrough_args)

        num_samples = args.pop("num_samples", 30000)
        subselect = args.pop("subselect", "fps")
        filtering_type = args.pop("filtering_type", "exclusive")
        weight_pt = args.pop("weight_pt", 1.0)
        weight_ft = args.pop("weight_ft", 10.0)
        device = args.pop("device", "cuda")

        rest = _format_cli_args(args)

        return (
            'echo "=== Stage 1: fine_tuning_select ==="\n'
            "python -u -m mace.cli.fine_tuning_select \\\n"
            f"  --configs_pt {self.replay_xyz} \\\n"
            f"  --configs_ft {path / 'train.xyz'} \\\n"
            f"  --num_samples {num_samples} \\\n"
            f"  --subselect {subselect} \\\n"
            f"  --model {self.model_path} \\\n"
            f"  --output {path / 'selected_configs.xyz'} \\\n"
            f"  --filtering_type {filtering_type} \\\n"
            "  --head_pt pt_head \\\n"
            "  --head_ft target_head \\\n"
            f"  --weight_pt {weight_pt} \\\n"
            f"  --device {device} \\\n"
            f"  --weight_ft {weight_ft}"
            + (f" \\\n{rest}" if rest else "")
            + " >> log_db"
        )

    def _train_stage(self, path: Path) -> str:
        """Stage 2: ``run_train``.

        Reads gorun-specific arguments directly; all training
        hyperparameters are pulled from ``_passthrough_args`` with
        sensible defaults.  SWA and EMA are handled as boolean
        toggle groups (sub-args only emitted when the toggle is on).
        """
        args = dict(self._passthrough_args)
        heads_path = path / "heads.json"

        # --- Pop known training parameters with defaults ---
        valid_fraction = args.pop("valid_fraction", 0.05)
        energy_weight = args.pop("energy_weight", 1.0)
        forces_weight = args.pop("forces_weight", 10.0)
        stress_weight = args.pop("stress_weight", 10.0)
        lr = args.pop("lr", 0.0009)
        weight_decay = args.pop("weight_decay", 5e-9)
        device = args.pop("device", "cuda")
        default_dtype = args.pop("default_dtype", "float64")
        max_num_epochs = args.pop("max_num_epochs", 100)
        batch_size = args.pop("batch_size", 6)
        compute_stress = args.pop("compute_stress", True)
        loss = args.pop("loss", "stress")
        seed = args.pop("seed", 1)
        fmhft = args.pop("force_mh_ft_lr", True)

        # --- SWA group ---
        flags: list[str] = []
        if args.pop("swa", True):
            flags.extend([
                "  --swa \\\n",
                f"  --swa_lr {args.pop('swa_lr', 0.0001)} \\\n",
                f"  --start_swa {args.pop('start_swa', 40)} \\\n",
                f"  --swa_energy_weight {args.pop('swa_energy_weight', 10.0)} \\\n",
                f"  --swa_forces_weight {args.pop('swa_forces_weight', 10.0)} \\\n",
                f"  --swa_stress_weight {args.pop('swa_stress_weight', 10.0)} \\\n",
            ])

        # --- EMA group ---
        if args.pop("ema", True):
            flags.extend([
                "  --ema \\\n",
                f"  --ema_decay {args.pop('ema_decay', 0.99999)} \\\n",
            ])

        # --- Remaining unknown args ---
        rest = _format_cli_args(args)

        return (
            'echo "=== Stage 2: run_train ==="\n'
            "python -u -m mace.cli.run_train \\\n"
            f"  --name {self.new_model_name} \\\n"
            "  --multiheads_finetuning True \\\n"
            f"  --heads $(cat {heads_path}) \\\n"
            f"  --valid_fraction {valid_fraction} \\\n"
            f"  --foundation_model {self.model_path} \\\n"
            f"  --energy_weight {energy_weight} \\\n"
            f"  --forces_weight {forces_weight} \\\n"
            f"  --stress_weight {stress_weight} \\\n"
            + "".join(flags) +
            f"  --force_mh_ft_lr={fmhft} \\\n"
            f"  --lr {lr} \\\n"
            f"  --weight_decay {weight_decay} \\\n"
            f"  --device {device} \\\n"
            f"  --default_dtype {default_dtype} \\\n"
            f"  --max_num_epochs {max_num_epochs} \\\n"
            f"  --batch_size {batch_size} \\\n"
            f"  --compute_stress {compute_stress} \\\n"
            f"  --loss {loss} \\\n"
            f"  --seed {seed}"
            + (f" \\\n{rest}" if rest else "")
            + "  >> log_tuning"
        )

    # ------------------------------------------------------------------
    # Backup excludes
    # ------------------------------------------------------------------

    def backup_excludes(self) -> list[str]:
        return [
            "*.tar.gz",
            "gorun_*",
        ]
