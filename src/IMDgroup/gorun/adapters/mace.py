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
control) are exposed as CLI flags and stored at the top level of
``INCAR.toml``.  All other MACE training hyperparameters --
``batch_size``, ``lr``, ``device``, ``max_num_epochs``, ``swa``,
``ema``, etc. -- are forwarded directly to ``mace.cli.run_train`` and
``mace.cli.fine_tuning_select`` via ``INCAR.toml``.  Use TOML sections
to route arguments to the correct stage::

    # INCAR.toml
    model_path = "/path/to/foundation.model"
    replay_xyz = "/path/to/replay.xyz"
    data_path = "structures.xyz"
    device = "cpu"

    [fine_tuning_select]
    num_samples = 50000
    subselect = "fps"

    [run_train]
    batch_size = 12
    lr = 0.0009
    swa = true
    ema = true

Keys at the top level that are not gorun-specific (unknown to the
adapter) default to ``run_train``.  ``device`` and ``default_dtype``
are shared — gorun passes them to both stages.
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

from IMDgroup.gorun.adapters.gpu import GpuAdapter, _wrap_command
from IMDgroup.gorun.core.files import maybe_regenerate
from IMDgroup.gorun.core.incar import (
    read_incar_toml,
    write_incar_toml_sections,
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
    Boolean ``False``, ``None``, and empty-string values are silently
    skipped (they are defaults that should not produce CLI flags).
    """
    if not args:
        return ""
    lines: list[str] = []
    for key, val in args.items():
        cli_key = key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                lines.append(f"  --{cli_key} \\")
        elif val is not None and val != "":
            lines.append(f"  --{cli_key} {val} \\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MaceMultiheadFinetuneAdapter
# ---------------------------------------------------------------------------


class MaceMultiheadFinetuneAdapter(GpuAdapter):
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

    #: Hardcoded defaults for all parameters.
    #: Top-level scalar keys are gorun-specific or shared between
    #: stages.  Dict-valued keys represent TOML ``[section]`` tables:
    #: their sub-keys are forwarded directly to the corresponding
    #: MACE CLI stage.
    DEFAULTS: dict[str, object] = {
        # Top-level — gorun-specific (data sources, output, workflow)
        "model_path": "",
        "replay_xyz": "",
        "data_path": "",
        "train_data_path": "",
        "val_data_path": "",
        "split_ratio": 0.20,
        "new_model_name": "finetuned_model.model",
        "seed": 1,
        "e0s": "",
        "no_fine_tuning_select": False,
        # Top-level — shared between both stages
        "device": "cuda",
        "default_dtype": "float64",
        # [fine_tuning_select]
        "fine_tuning_select": {
            "num_samples": 30000,
            "subselect": "fps",
            "filtering_type": "exclusive",
            "weight_pt": 1.0,
            "weight_ft": 10.0,
            # Other MACE CLI flags users may set
            "descriptors": "",
            "disallow_random_padding": False,
            "filter_atomic_numbers_pt": "",
        },
        # [run_train]
        "run_train": {
            "valid_fraction": 0.05,
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "stress_weight": 10.0,
            "lr": 0.0009,
            "weight_decay": 5e-9,
            "max_num_epochs": 100,
            "batch_size": 6,
            "compute_stress": True,
            "loss": "stress",
            "force_mh_ft_lr": True,
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
        },
    }

    #: Parameters that must be non-empty for a valid run.
    REQUIRED: frozenset[str] = frozenset({"model_path", "replay_xyz"})

    @classmethod
    def _build_key_section(cls) -> dict[str, str | None]:
        """Map every adapter-owned key to its TOML section.

        Returns a dict mapping key → section name.  Dict-valued
        keys in ``DEFAULTS`` are section tables; their sub-keys
        map to that section name.  Scalar-valued keys map to
        ``None`` (top-level placement).

        The structure of ``DEFAULTS`` is the single source of truth
        for section membership.
        """
        mapping: dict[str, str | None] = {}
        for key, val in cls.DEFAULTS.items():
            if isinstance(val, dict):
                for sub_key in val:
                    mapping[sub_key] = key
            else:
                mapping[key] = None
        return mapping

    # pylint: disable=too-many-locals,too-many-branches
    def __init__(
        self, *,
        # Data sources
        model_path: str | None = None,
        replay_xyz: str | None = None,
        data_path: str | None = None,
        train_data_path: str | None = None,
        val_data_path: str | None = None,
        split_ratio: float | None = None,
        # Output
        new_model_name: str | None = None,
        seed: int | None = None,
        e0s: str | None = None,
        # Workflow
        no_fine_tuning_select: bool | None = None,
        # INCAR.toml overrides
        incar_overrides: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        """Build adapter from explicit kwargs, INCAR.toml, and --incar overrides.

        Merge order (lower to higher priority):
        adapter DEFAULTS < INCAR.toml < --incar overrides < explicit CLI kwargs.

        INCAR.toml is read from the working directory.  The merge
        is structural: ``[fine_tuning_select]`` and ``[run_train]``
        tables from INCAR.toml are merged into the corresponding
        section dicts in DEFAULTS.  Unknown top-level scalars in
        INCAR.toml default to ``run_train`` (backward compatibility
        with INCAR.toml files that predate section grouping).

        *incar_overrides* maps section name (or ``None`` for
        auto-routing) to ``{key: val_str}``.  Auto-routed keys
        (``None`` section) are placed based on DEFAULTS structure:
        keys known in a section go there; everything else defaults
        to ``run_train``.  Values are strings from the CLI.

        When INCAR.toml does not exist and no CLI args provide the
        required parameters, a reference INCAR.toml is written and
        the process exits.
        """
        raw_toml = read_incar_toml()
        if incar_overrides is None:
            incar_overrides = {}

        # --- Call GpuAdapter init ---
        super().__init__()

        # --- Extract section defaults from DEFAULTS ---
        select_defaults: dict[str, object] = dict(
            self.DEFAULTS.get("fine_tuning_select", {}))
        train_defaults: dict[str, object] = dict(
            self.DEFAULTS.get("run_train", {}))

        # Build sets of known keys for auto-routing.
        known_select_keys: set[str] = set(select_defaults.keys())
        known_top_keys: set[str] = {
            k for k, v in self.DEFAULTS.items()
            if not isinstance(v, dict)
        }

        # --- Merge sections: DEFAULTS < TOML section < incar_overrides section ---
        merged_select: dict[str, object] = dict(select_defaults)
        merged_train: dict[str, object] = dict(train_defaults)

        for key, val in raw_toml.items():
            if isinstance(val, dict):
                if key == "fine_tuning_select":
                    merged_select.update(val)
                elif key == "run_train":
                    merged_train.update(val)

        section_overrides = incar_overrides.get("fine_tuning_select", {})
        if section_overrides:
            merged_select.update(section_overrides)
        section_overrides = incar_overrides.get("run_train", {})
        if section_overrides:
            merged_train.update(section_overrides)

        # --- Merge top-level: DEFAULTS < TOML scalars < auto-routed incar ---
        merged_top: dict[str, object] = {
            k: v for k, v in self.DEFAULTS.items()
            if not isinstance(v, dict)
        }

        for key, val in raw_toml.items():
            if not isinstance(val, dict):
                if key in known_top_keys:
                    merged_top[key] = val
                elif key in known_select_keys:
                    merged_select[key] = val
                else:
                    # Unknown top-level keys default to run_train.
                    merged_train[key] = val

        auto_overrides = incar_overrides.get(None, {})
        for key, val in auto_overrides.items():
            if key in known_select_keys:
                merged_select[key] = val
            elif key in known_top_keys:
                merged_top[key] = val
            else:
                merged_train[key] = val

        # --- CLI kwargs override top-level ---
        cli_kwargs: dict[str, object] = {}
        if model_path is not None:
            cli_kwargs["model_path"] = model_path
        if replay_xyz is not None:
            cli_kwargs["replay_xyz"] = replay_xyz
        if data_path is not None:
            cli_kwargs["data_path"] = data_path
        if train_data_path is not None:
            cli_kwargs["train_data_path"] = train_data_path
        if val_data_path is not None:
            cli_kwargs["val_data_path"] = val_data_path
        if split_ratio is not None:
            cli_kwargs["split_ratio"] = split_ratio
        if new_model_name is not None:
            cli_kwargs["new_model_name"] = new_model_name
        if seed is not None:
            cli_kwargs["seed"] = seed
        if e0s is not None:
            cli_kwargs["e0s"] = e0s
        if no_fine_tuning_select is not None:
            cli_kwargs["no_fine_tuning_select"] = no_fine_tuning_select
        merged_top.update(cli_kwargs)

        # --- Copy shared keys into both stage dicts ---
        for shared_key in ("device", "default_dtype"):
            if shared_key not in merged_select:
                merged_select[shared_key] = merged_top[shared_key]
            if shared_key not in merged_train:
                merged_train[shared_key] = merged_top[shared_key]

        # --- Populate instance attributes from merged top-level ---
        self.model_path: str = merged_top.get("model_path", "")
        self.replay_xyz: str = merged_top.get("replay_xyz", "")
        self.data_path: str = merged_top.get("data_path", "")
        self.train_data_path: str = merged_top.get("train_data_path", "")
        self.val_data_path: str = merged_top.get("val_data_path", "")
        self.split_ratio: float = merged_top.get("split_ratio", 0.20)
        self.new_model_name: str = merged_top.get(
            "new_model_name", "finetuned_model.model")
        self.seed: int = merged_top.get("seed", 1)
        self.e0s: str = merged_top.get("e0s", "")
        self.no_fine_tuning_select: bool = merged_top.get(
            "no_fine_tuning_select", False)

        #: Extra CLI flags forwarded to ``fine_tuning_select``.
        self._select_args: dict[str, object] = merged_select

        #: Extra CLI flags forwarded to ``run_train``.
        self._train_args: dict[str, object] = merged_train

        #: Raw INCAR.toml content, stored for write-back in
        #: ``prepare_inputs``.
        self._raw_toml: dict[str, object] = raw_toml

        #: Flat merged parameter dict for INCAR.toml write-back.
        self._merged_kwargs: dict[str, object] = {}
        self._merged_kwargs.update(merged_top)
        self._merged_kwargs.update(merged_select)
        self._merged_kwargs.update(merged_train)

        # --- Bootstrap ---
        has_cli_input = bool(cli_kwargs) or bool(incar_overrides)
        if not raw_toml and not has_cli_input:
            missing_required = [
                attr for attr in self.REQUIRED
                if not self._merged_kwargs.get(attr)
            ]
            if missing_required:
                write_incar_toml_sections(
                    self._merged_kwargs,
                    key_section=self._build_key_section(),
                    raw_existing=None,
                    defaults=self.DEFAULTS,
                    always_include=self.REQUIRED,
                )
                print(colored(
                    "Edit INCAR.toml to set the required parameters: "
                    + ", ".join(missing_required),
                    "cyan",
                ))
                sys.exit(0)

        self.validate()

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
        if not self.data_path and not self.train_data_path:
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

    def setup_commands(self, server_config: dict) -> str:
        """Read MACE environment, falling back to ``[gpu]`` config.

        Looks for ``server_config["mace"]["setup"]`` first.  If that
        section is missing or empty, falls back to
        ``server_config["gpu"]["setup"]``.  This allows sharing a
        common GPU environment across multiple adapters.
        """
        mace_setup = server_config.get("mace", {}).get("setup", "")
        if mace_setup:
            return mace_setup
        return server_config.get("gpu", {}).get("setup", "")

    # ------------------------------------------------------------------
    # Directory inspection
    # ------------------------------------------------------------------

    def is_valid_input(self, path: Path) -> bool:
        """Return True when at least one data source file exists."""
        sources = [
            p for p in (self.data_path, self.train_data_path, self.val_data_path)
            if p
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

        # Write back merged INCAR.toml.  Deferred here (rather than
        # in the constructor) so that the file is not touched when
        # dispatch_run exits early due to gorun_ready or RUNNING.
        if self._merged_kwargs:
            write_incar_toml_sections(
                self._merged_kwargs,
                key_section=self._build_key_section(),
                raw_existing=self._raw_toml,
                defaults=self.DEFAULTS,
            )

        train_xyz = path / "train.xyz"
        val_xyz = path / "val.xyz"

        # -- train.xyz / val.xyz --
        sources = (None if force else self._get_train_val_source_paths())
        maybe_regenerate(
            path, ["train.xyz", "val.xyz"], keep,
            regenerate=lambda: self._generate_train_val(path),
            older_than=sources,
        )

        # -- heads.json --
        maybe_regenerate(
            path, "heads.json", keep,
            regenerate=lambda: self._write_heads_json(path),
            older_than=[train_xyz, val_xyz],
        )

    def _get_train_val_source_paths(self) -> list[Path]:
        """Return paths of source files used to generate train/val."""
        paths: list[Path] = []
        if self.data_path:
            p = Path(self.data_path)
            if p.is_file():
                paths.append(p)
        if self.train_data_path:
            p = Path(self.train_data_path)
            if p.is_file():
                paths.append(p)
        if self.val_data_path:
            p = Path(self.val_data_path)
            if p.is_file():
                paths.append(p)
        return paths

    def _generate_train_val(self, path: Path) -> None:
        """Read structures from data source(s), strip constraints,
        split if needed, and write ``train.xyz`` and ``val.xyz``.
        """
        if self.train_data_path and self.val_data_path:
            train = self._read_and_clean(self.train_data_path)
            val = self._read_and_clean(self.val_data_path)
        elif self.data_path:
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
            train_xyz = path / "train.xyz"
            need_select = maybe_regenerate(
                path, "selected_configs.xyz", keep=None,
                regenerate=lambda: None,
                older_than=[train_xyz],
            )
            if need_select:
                parts.append(_wrap_command(self._select_stage(path)))
        parts.append(_wrap_command(self._train_stage(path)))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Stage commands (private)
    # ------------------------------------------------------------------

    def _select_stage(self, path: Path) -> str:
        """Stage 1: ``fine_tuning_select``.

        Only select-stage keys are forwarded from ``_select_args``.
        Train-only flags never reach this stage.
        """
        args = dict(self._select_args)
        sd = self.DEFAULTS["fine_tuning_select"]

        num_samples = args.pop("num_samples", sd["num_samples"])
        subselect = args.pop("subselect", sd["subselect"])
        filtering_type = args.pop("filtering_type", sd["filtering_type"])
        weight_pt = args.pop("weight_pt", sd["weight_pt"])
        weight_ft = args.pop("weight_ft", sd["weight_ft"])
        device = args.pop("device", self.DEFAULTS["device"])
        default_dtype = args.pop("default_dtype", self.DEFAULTS["default_dtype"])

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
            f"  --weight_ft {weight_ft} \\\n"
            f"  --device {device} \\\n"
            f"  --default_dtype {default_dtype}"
            + (f" \\\n{rest}" if rest else "")
            + " >> log_db"
        )

    def _train_stage(self, path: Path) -> str:
        """Stage 2: ``run_train``.

        Only train-stage keys are forwarded from ``_train_args``.
        SWA and EMA are handled as boolean toggle groups (sub-args
        only emitted when the toggle is on).
        """
        args = dict(self._train_args)
        heads_path = path / "heads.json"

        # --- Pop known training parameters with defaults ---
        td = self.DEFAULTS["run_train"]
        valid_fraction = args.pop("valid_fraction", td["valid_fraction"])
        energy_weight = args.pop("energy_weight", td["energy_weight"])
        forces_weight = args.pop("forces_weight", td["forces_weight"])
        stress_weight = args.pop("stress_weight", td["stress_weight"])
        lr = args.pop("lr", td["lr"])
        weight_decay = args.pop("weight_decay", td["weight_decay"])
        device = args.pop("device", self.DEFAULTS["device"])
        default_dtype = args.pop("default_dtype", self.DEFAULTS["default_dtype"])
        max_num_epochs = args.pop("max_num_epochs", td["max_num_epochs"])
        batch_size = args.pop("batch_size", td["batch_size"])
        compute_stress = args.pop("compute_stress", td["compute_stress"])
        loss = args.pop("loss", td["loss"])
        seed = args.pop("seed", self.seed)
        fmhft = args.pop("force_mh_ft_lr", td["force_mh_ft_lr"])

        # --- SWA group ---
        flags: list[str] = []
        if args.pop("swa", td["swa"]):
            flags.extend([
                "  --swa \\\n",
                f"  --swa_lr {args.pop('swa_lr', td['swa_lr'])} \\\n",
                f"  --start_swa {args.pop('start_swa', td['start_swa'])} \\\n",
                f"  --swa_energy_weight {args.pop('swa_energy_weight', td['swa_energy_weight'])} \\\n",
                f"  --swa_forces_weight {args.pop('swa_forces_weight', td['swa_forces_weight'])} \\\n",
                f"  --swa_stress_weight {args.pop('swa_stress_weight', td['swa_stress_weight'])} \\\n",
            ])

        # --- EMA group ---
        if args.pop("ema", td["ema"]):
            flags.extend([
                "  --ema \\\n",
                f"  --ema_decay {args.pop('ema_decay', td['ema_decay'])} \\\n",
            ])

        # --- Remaining unknown args ---
        rest = _format_cli_args(args)

        return (
            'echo "=== Stage 2: run_train ==="\n'
            "python -u -m mace.cli.run_train \\\n"
            f"  --name {self.new_model_name} \\\n"
            "  --multiheads_finetuning True \\\n"
            f'  --heads "$(cat {heads_path})" \\\n'
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
        return []
