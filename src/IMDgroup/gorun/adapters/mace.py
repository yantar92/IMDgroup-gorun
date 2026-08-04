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
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from sklearn.model_selection import train_test_split
from termcolor import colored

from IMDgroup.gorun.core.files import maybe_regenerate


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
# MaceMultiheadFinetuneAdapter
# ---------------------------------------------------------------------------


class MaceMultiheadFinetuneAdapter:
    """``gorun mace`` backend for multihead fine-tuning of a foundation model.

    Two-stage workflow (override with ``RUNFILE.sh`` or ``RUNFILE.py``):

    1. ``mace.cli.fine_tuning_select`` -- select replay configurations.
    2. ``mace.cli.run_train`` -- dual-head training.

    Data sources are specified via one of:

    - ``--data-path`` + ``--split-ratio``: single file (.xyz or .pkl), auto-split.
    - ``--train-data-path`` + ``--val-data-path``: pre-split files.

    Parameters beyond those exposed as CLI flags can be set via
    ``INCAR.toml`` (see :func:`apply_kwargs`).  Unknown keys in
    ``INCAR.toml`` are forwarded to ``run_train`` as ``--key value``
    flags, so future MACE CLI additions work without code changes.
    """

    name = "mace"
    log_file = "mace.out"

    #: Hardcoded defaults for every parameter the adapter knows about.
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
        # Training
        "batch_size": 6,
        "max_num_epochs": 100,
        "valid_fraction": 0.05,
        "lr": 0.0009,
        "weight_decay": 5e-9,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 10.0,
        # SWA
        "swa": True,
        "swa_lr": 0.0001,
        "start_swa": 40,
        "swa_energy_weight": 10.0,
        "swa_forces_weight": 10.0,
        "swa_stress_weight": 10.0,
        # EMA
        "ema": True,
        "ema_decay": 0.99999,
        # Multihead
        "force_mh_ft_lr": True,
        # Stage 1: fine_tuning_select
        "no_fine_tuning_select": False,
        "num_samples": 30000,
        "subselect": "fps",
        "filtering_type": "exclusive",
        "weight_pt": 1.0,
        "weight_ft": 10.0,
        # Hardware & precision
        "device": "cuda",
        "default_dtype": "float64",
        # Loss
        "compute_stress": True,
        "loss": "stress",
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
        # Training hyperparameters --------------------------------------
        batch_size: int = 6,
        max_num_epochs: int = 100,
        valid_fraction: float = 0.05,
        lr: float = 0.0009,
        weight_decay: float = 5e-9,
        energy_weight: float = 1.0,
        forces_weight: float = 10.0,
        stress_weight: float = 10.0,
        # SWA -----------------------------------------------------------
        swa: bool = True,
        swa_lr: float = 0.0001,
        start_swa: int = 40,
        swa_energy_weight: float = 10.0,
        swa_forces_weight: float = 10.0,
        swa_stress_weight: float = 10.0,
        # EMA -----------------------------------------------------------
        ema: bool = True,
        ema_decay: float = 0.99999,
        # Multihead -----------------------------------------------------
        force_mh_ft_lr: bool = True,
        # Stage 1: fine_tuning_select -----------------------------------
        no_fine_tuning_select: bool = False,
        num_samples: int = 30000,
        subselect: str = "fps",
        filtering_type: str = "exclusive",
        weight_pt: float = 1.0,
        weight_ft: float = 10.0,
        # Hardware & precision ------------------------------------------
        device: str = "cuda",
        default_dtype: str = "float64",
        # Loss ----------------------------------------------------------
        compute_stress: bool = True,
        loss: str = "stress",
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
        # Training
        self.batch_size = batch_size
        self.max_num_epochs = max_num_epochs
        self.valid_fraction = valid_fraction
        self.lr = lr
        self.weight_decay = weight_decay
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        # SWA
        self.swa = swa
        self.swa_lr = swa_lr
        self.start_swa = start_swa
        self.swa_energy_weight = swa_energy_weight
        self.swa_forces_weight = swa_forces_weight
        self.swa_stress_weight = swa_stress_weight
        # EMA
        self.ema = ema
        self.ema_decay = ema_decay
        # Multihead
        self.force_mh_ft_lr = force_mh_ft_lr
        # Stage 1
        self.no_fine_tuning_select = no_fine_tuning_select
        self.num_samples = num_samples
        self.subselect = subselect
        self.filtering_type = filtering_type
        self.weight_pt = weight_pt
        self.weight_ft = weight_ft
        # Hardware
        self.device = device
        self.default_dtype = default_dtype
        # Loss
        self.compute_stress = compute_stress
        self.loss = loss

        #: Extra CLI flags to forward to ``run_train`` as ``--key value``.
        self._passthrough_args: dict[str, object] = {}

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
                    "No data source specified.  Provide --data-path or --train-data-path/--val-data-path "
                    "via CLI or INCAR.toml, or place an existing train.xyz in the directory.",
                    "yellow",
                )
            )

    # ------------------------------------------------------------------
    # Extra kwargs (INCAR.toml passthrough)
    # ------------------------------------------------------------------

    def apply_kwargs(self, kwargs: dict[str, object]) -> None:
        """Apply a dict of key-value pairs to the adapter.

        Keys that match an adapter attribute override that attribute.
        Unknown keys are stored in ``_passthrough_args`` and forwarded
        to ``run_train`` as ``--key value``.
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                self._passthrough_args[key] = val

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
        # Also check if the final model exists (from a successful run)
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
            print("No data source specified (--data-path or --train-data-path/--val-data-path).")
            # Fallback: if train.xyz/val.xyz already exist, skip gracefully.
            # Otherwise, let the user know.
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
        # -- RUNFILE escape hatch --
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
        """Stage 1: ``fine_tuning_select``."""
        return (
            'echo "=== Stage 1: fine_tuning_select ==="\n'
            "python -u -m mace.cli.fine_tuning_select \\\n"
            f"  --configs_pt {self.replay_xyz} \\\n"
            f"  --configs_ft {path / 'train.xyz'} \\\n"
            f"  --num_samples {self.num_samples} \\\n"
            f"  --subselect {self.subselect} \\\n"
            f"  --model {self.model_path} \\\n"
            f"  --output {path / 'selected_configs.xyz'} \\\n"
            f"  --filtering_type {self.filtering_type} \\\n"
            "  --head_pt pt_head \\\n"
            "  --head_ft target_head \\\n"
            f"  --weight_pt {self.weight_pt} \\\n"
            f"  --device {self.device} \\\n"
            f"  --weight_ft {self.weight_ft} >> log_db"
        )

    def _passthrough_flags(self) -> str:
        """Format ``_passthrough_args`` as ``--key value \\`` lines."""
        if not self._passthrough_args:
            return ""
        lines: list[str] = []
        for key, val in self._passthrough_args.items():
            cli_key = key.replace("_", "-")
            if isinstance(val, bool):
                if val:
                    lines.append(f"  --{cli_key} \\")
            else:
                lines.append(f"  --{cli_key} {val} \\")
        return "\n".join(lines) + "\n" if lines else ""

    def _train_stage(self, path: Path) -> str:
        """Stage 2: ``run_train``."""
        heads_path = path / "heads.json"
        flags = []

        if self.swa:
            flags.extend([
                "  --swa \\",
                f"  --swa_lr {self.swa_lr} \\",
                f"  --start_swa {self.start_swa} \\",
                f"  --swa_energy_weight {self.swa_energy_weight} \\",
                f"  --swa_forces_weight {self.swa_forces_weight} \\",
                f"  --swa_stress_weight {self.swa_stress_weight} \\",
            ])
        if self.ema:
            flags.extend([
                "  --ema \\",
                f"  --ema_decay {self.ema_decay} \\",
            ])

        passthrough = self._passthrough_flags()
        return (
            'echo "=== Stage 2: run_train ==="\n'
            "python -u -m mace.cli.run_train \\\n"
            f"  --name {self.new_model_name} \\\n"
            "  --multiheads_finetuning True \\\n"
            f"  --heads $(cat {heads_path}) \\\n"
            f"  --valid_fraction {self.valid_fraction} \\\n"
            f"  --foundation_model {self.model_path} \\\n"
            f"  --energy_weight {self.energy_weight} \\\n"
            f"  --forces_weight {self.forces_weight} \\\n"
            f"  --stress_weight {self.stress_weight} \\\n"
            + "".join(flags) +
            f"  --force_mh_ft_lr={self.force_mh_ft_lr} \\\n"
            f"  --lr {self.lr} \\\n"
            f"  --weight_decay {self.weight_decay} \\\n"
            f"  --device {self.device} \\\n"
            f"  --default_dtype {self.default_dtype} \\\n"
            f"  --max_num_epochs {self.max_num_epochs} \\\n"
            f"  --batch_size {self.batch_size} \\\n"
            f"  --compute_stress {self.compute_stress} \\\n"
            f"  --loss {self.loss} \\\n"
            f"  --seed {self.seed}"
            + (f" \\\n{passthrough}" if passthrough else "")
            + "  >> log_tuning"
        )

    # ------------------------------------------------------------------
    # Backup excludes
    # ------------------------------------------------------------------

    def backup_excludes(self) -> list[str]:
        return [
            "*.tar.gz",
            "gorun_*",
            "*.xyz",
            "*.model",
            "*.model.staged",
            "checkpoints",
            "log_tuning",
            "log_db",
            "log.out",
            "log.err",
        ]
