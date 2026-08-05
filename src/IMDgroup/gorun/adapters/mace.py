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

from IMDgroup.gorun.core.files import maybe_regenerate
from IMDgroup.gorun.core.incar import (
    read_incar_toml,
    write_incar_toml_sections,
)


# ---------------------------------------------------------------------------
# Wrapper helper
# ---------------------------------------------------------------------------

def _wrap_command(command: str) -> str:
    """Wrap *command* in a heredoc passed to ``GORUN_WRAPPER``.

    ``${GORUN_WRAPPER:-bash}`` ensures the heredoc is always piped to
    a shell: the configured wrapper when ``GORUN_WRAPPER`` is set, or
    plain ``bash`` otherwise.  This avoids duplicating *command* in
    an ``if/else`` branch.

    The heredoc is quoted (``<<'EOF'``) so the outer shell does not
    expand ``${GORUN_INNER_SETUP}`` before passing it through; the
    inner shell expands it.  Both variables are expected to be exported
    by the config's ``setup`` commands.
    """
    return (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        f"{command}\n"
        "EOF"
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

    #: Hardcoded defaults for all parameters.
    #: Used for INCAR.toml bootstrapping, sparse write-back, and as
    #: the fallback when a passthrough key is absent from INCAR.toml.
    DEFAULTS: dict[str, object] = {
        # Data sources
        "model_path": "",
        "replay_xyz": "",
        "data_path": "",
        "train_data_path": "",
        "val_data_path": "",
        "split_ratio": 0.20,
        # Output
        "new_model_name": "finetuned_model.model",
        "seed": 1,
        "e0s": "",
        # Workflow
        "no_fine_tuning_select": False,
        # fine_tuning_select
        "num_samples": 30000,
        "subselect": "fps",
        "filtering_type": "exclusive",
        "weight_pt": 1.0,
        "weight_ft": 10.0,
        # run_train — core
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 10.0,
        "lr": 0.0009,
        "weight_decay": 5e-9,
        "device": "cuda",
        "default_dtype": "float64",
        "max_num_epochs": 100,
        "batch_size": 6,
        "compute_stress": True,
        "loss": "stress",
        "force_mh_ft_lr": True,
        # run_train — SWA
        "swa": True,
        "swa_lr": 0.0001,
        "start_swa": 40,
        "swa_energy_weight": 10.0,
        "swa_forces_weight": 10.0,
        "swa_stress_weight": 10.0,
        # run_train — EMA
        "ema": True,
        "ema_decay": 0.99999,
    }

    #: Parameters that must be non-empty for a valid run.
    REQUIRED: frozenset[str] = frozenset({"model_path", "replay_xyz"})

    #: Keys that belong exclusively to ``fine_tuning_select``.
    #: ``run_train`` does not accept these; putting them in
    #: ``INCAR.toml`` routes them to ``_select_args`` only.
    _SELECT_ONLY_KEYS: frozenset[str] = frozenset({
        # Explicitly handled by _select_stage
        'num_samples', 'subselect', 'filtering_type',
        'weight_pt', 'weight_ft',
        # Other fine_tuning_select flags users might set in INCAR.toml
        'descriptors', 'disallow_random_padding',
        'filter_atomic_numbers_pt',
    })

    #: Keys accepted by both ``fine_tuning_select`` and ``run_train``.
    _SHARED_STAGE_KEYS: frozenset[str] = frozenset({
        'device', 'default_dtype',
    })

    #: Gorun-specific keys that are not forwarded to either MACE CLI.
    _TOP_LEVEL_KEYS: frozenset[str] = frozenset({
        'model_path', 'replay_xyz',
        'data_path', 'train_data_path', 'val_data_path', 'split_ratio',
        'new_model_name', 'seed', 'e0s', 'no_fine_tuning_select',
    })

    #: TOML section name for ``fine_tuning_select`` arguments.
    _SELECT_SECTION: str = 'fine_tuning_select'

    #: TOML section name for ``run_train`` arguments.
    _TRAIN_SECTION: str = 'run_train'

    @classmethod
    def _build_key_section(cls) -> dict[str, str | None]:
        """Map every adapter-owned key to its TOML section.

        Returns a dict mapping key → section name.  Keys in
        ``_TOP_LEVEL_KEYS`` and ``_SHARED_STAGE_KEYS`` map to
        ``None`` (top-level placement).  ``_SELECT_ONLY_KEYS``
        map to ``_SELECT_SECTION``.  Everything else in
        ``DEFAULTS`` maps to ``_TRAIN_SECTION``.
        """
        mapping: dict[str, str | None] = {}
        for key in cls.DEFAULTS:
            if key in cls._TOP_LEVEL_KEYS or key in cls._SHARED_STAGE_KEYS:
                mapping[key] = None
            elif key in cls._SELECT_ONLY_KEYS:
                mapping[key] = cls._SELECT_SECTION
            else:
                mapping[key] = cls._TRAIN_SECTION
        return mapping

    # pylint: disable=too-many-locals
    def __init__(
        self, *,
        # Data sources
        model_path: str,
        replay_xyz: str,
        data_path: str = "",
        train_data_path: str = "",
        val_data_path: str = "",
        split_ratio: float = 0.20,
        # Output
        new_model_name: str = "finetuned_model.model",
        seed: int = 1,
        e0s: str = "",
        # Workflow
        no_fine_tuning_select: bool = False,
        # Passthrough
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

        #: Extra CLI flags forwarded to ``fine_tuning_select``
        #: as ``--key value``.  Subset of *kwargs*: select-only
        #: keys plus shared keys like ``device``.
        self._select_args: dict[str, object] = {
            k: v for k, v in kwargs.items()
            if k in self._SELECT_ONLY_KEYS
            or k in self._SHARED_STAGE_KEYS
        }

        #: Extra CLI flags forwarded to ``run_train`` as
        #: ``--key value``.  Everything in *kwargs* except
        #: select-only keys (unknown TOML keys land here).
        self._train_args: dict[str, object] = {
            k: v for k, v in kwargs.items()
            if k not in self._SELECT_ONLY_KEYS
        }

        #: Raw INCAR.toml content read at construction time, stored
        #: for later write-back in ``prepare_inputs``.  Empty when
        #: the adapter is constructed directly (not via
        #: ``from_cli_and_toml``).
        self._raw_toml: dict[str, object] = {}
        #: Merged parameter values (defaults < TOML < CLI), stored
        #: for later write-back in ``prepare_inputs``.
        self._merged_kwargs: dict[str, object] = {}

    # Construction from CLI + INCAR.toml
    @classmethod
    def from_cli_and_toml(cls, args) -> "MaceMultiheadFinetuneAdapter":
        """Build adapter from CLI args and INCAR.toml.

        Merge order: adapter defaults -> INCAR.toml -> --incar -> explicit CLI flags.

        INCAR.toml may use ``[fine_tuning_select]`` and
        ``[run_train]`` sections.  Keys from those tables are
        flattened into the adapter's flat parameter dict before
        splitting into stage-specific args via ``_SELECT_ONLY_KEYS``.

        The ``--incar`` flag accepts ``KEY:VAL`` pairs.  Dot notation
        (``SECTION.KEY:VAL``) strips the section prefix; the bare key
        is merged into the flat parameter dict.  Unknown TOML keys
        are forwarded as passthrough ``--key value`` flags to
        ``run_train`` (unless they match ``_SELECT_ONLY_KEYS``, in
        which case they go to ``fine_tuning_select``).

        INCAR.toml is not written here; write-back is deferred to
        ``prepare_inputs`` so the file is not touched when the
        pipeline exits early (e.g. ``gorun_ready`` already present).

        When ``INCAR.toml`` does not exist and required parameters are
        missing, a reference file is written and the program exits.
        """
        raw_toml = read_incar_toml()
        defaults = cls.DEFAULTS

        # --incar overrides: KEY:VAL or SECTION.KEY:VAL
        incar_overrides: dict[str, object] = {}
        if getattr(args, 'incar', None):
            for item in args.incar:
                if ':' not in item:
                    print(colored(
                        f"Invalid --incar value '{item}'.  "
                        "Expected KEY:VAL format.",
                        "red",
                    ))
                    sys.exit(1)
                key_path, val_str = item.split(':', 1)
                key = key_path.rsplit('.', 1)[-1] if '.' in key_path else key_path
                incar_overrides[key] = val_str

        # Flatten TOML sections into a flat overlay dict.
        toml_overlay: dict[str, object] = {}
        for key, val in raw_toml.items():
            if isinstance(val, dict):
                toml_overlay.update(val)
            else:
                toml_overlay[key] = val

        # Build init kwargs: defaults < TOML < --incar < CLI
        init_kwargs: dict[str, object] = dict(defaults)
        for key in defaults:
            if key in toml_overlay:
                init_kwargs[key] = toml_overlay[key]
            if key in incar_overrides:
                init_kwargs[key] = incar_overrides[key]
            if hasattr(args, key):
                init_kwargs[key] = getattr(args, key)

        # Unknown TOML keys
        for key, val in toml_overlay.items():
            if key in defaults:
                continue
            init_kwargs[key] = val

        # Unknown --incar keys (after known keys, so they override TOML but not CLI)
        for key, val in incar_overrides.items():
            if key in defaults:
                continue
            init_kwargs[key] = val

        adapter = cls(**init_kwargs)

        # Store for later write-back in prepare_inputs, so that
        # INCAR.toml is not touched when gorun_ready is present and
        # the pipeline exits early.
        adapter._raw_toml = raw_toml
        adapter._merged_kwargs = init_kwargs

        # Bootstrap: no INCAR.toml, no MACE CLI args, required params still empty.
        # Write a reference INCAR.toml and exit.
        cli_mace_args = (
            any(hasattr(args, key) for key in defaults)
            or bool(incar_overrides)
        )
        if not raw_toml and not cli_mace_args:
            missing_required = [attr for attr in adapter.REQUIRED
                                if not init_kwargs.get(attr)]
            if missing_required:
                write_incar_toml_sections(
                    init_kwargs,
                    key_section=cls._build_key_section(),
                    raw_existing=None,
                    defaults=defaults,
                    always_include=adapter.REQUIRED,
                )
                print(colored(
                    "Edit INCAR.toml to set the required parameters: "
                    + ", ".join(missing_required),
                    "cyan",
                ))
                sys.exit(0)

        adapter.validate()

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
        # in from_cli_and_toml) so that the file is not touched when
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
        d = self.DEFAULTS

        num_samples = args.pop("num_samples", d["num_samples"])
        subselect = args.pop("subselect", d["subselect"])
        filtering_type = args.pop("filtering_type", d["filtering_type"])
        weight_pt = args.pop("weight_pt", d["weight_pt"])
        weight_ft = args.pop("weight_ft", d["weight_ft"])
        device = args.pop("device", d["device"])
        default_dtype = args.pop("default_dtype", d["default_dtype"])

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
        d = self.DEFAULTS
        valid_fraction = args.pop("valid_fraction", d["valid_fraction"])
        energy_weight = args.pop("energy_weight", d["energy_weight"])
        forces_weight = args.pop("forces_weight", d["forces_weight"])
        stress_weight = args.pop("stress_weight", d["stress_weight"])
        lr = args.pop("lr", d["lr"])
        weight_decay = args.pop("weight_decay", d["weight_decay"])
        device = args.pop("device", d["device"])
        default_dtype = args.pop("default_dtype", d["default_dtype"])
        max_num_epochs = args.pop("max_num_epochs", d["max_num_epochs"])
        batch_size = args.pop("batch_size", d["batch_size"])
        compute_stress = args.pop("compute_stress", d["compute_stress"])
        loss = args.pop("loss", d["loss"])
        seed = args.pop("seed", self.seed)
        fmhft = args.pop("force_mh_ft_lr", d["force_mh_ft_lr"])

        # --- SWA group ---
        flags: list[str] = []
        if args.pop("swa", d["swa"]):
            flags.extend([
                "  --swa \\\n",
                f"  --swa_lr {args.pop('swa_lr', d['swa_lr'])} \\\n",
                f"  --start_swa {args.pop('start_swa', d['start_swa'])} \\\n",
                f"  --swa_energy_weight {args.pop('swa_energy_weight', d['swa_energy_weight'])} \\\n",
                f"  --swa_forces_weight {args.pop('swa_forces_weight', d['swa_forces_weight'])} \\\n",
                f"  --swa_stress_weight {args.pop('swa_stress_weight', d['swa_stress_weight'])} \\\n",
            ])

        # --- EMA group ---
        if args.pop("ema", d["ema"]):
            flags.extend([
                "  --ema \\\n",
                f"  --ema_decay {args.pop('ema_decay', d['ema_decay'])} \\\n",
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
