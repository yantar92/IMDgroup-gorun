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

"""MACE fine-tuning adapter for gorun."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from sklearn.model_selection import train_test_split

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
# MaceFinetuneAdapter
# ---------------------------------------------------------------------------


class MaceFinetuneAdapter:
    """``gorun mace`` backend for fine-tuning a foundation model."""

    name = "mace"
    log_file = "mace.out"

    def __init__(
        self, *,
        model_path: str,
        replay_xyz: str,
        pkl_path: str = "result.pkl",
        new_model_name: str = "finetuned_model.model",
        seed: int = 1,
        e0s: str = "",
        batch_size: int = 6,
        max_num_epochs: int = 100,
        num_samples: int = 30000,
        no_fine_tuning_select: bool = False,
    ) -> None:
        self.model_path = model_path
        self.replay_xyz = replay_xyz
        self.pkl_path = pkl_path
        self.new_model_name = new_model_name
        self.seed = seed
        self.e0s = e0s
        self.batch_size = batch_size
        self.max_num_epochs = max_num_epochs
        self.num_samples = num_samples
        self.no_fine_tuning_select = no_fine_tuning_select

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        """Check that the foundation model and replay data exist."""
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
        pkl = path / self.pkl_path
        return pkl.is_file()

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

        # train.xyz and val.xyz are generated together.  We cannot
        # use maybe_regenerate per-file without double-running.
        train_exists = (path / "train.xyz").is_file()
        val_exists = (path / "val.xyz").is_file()
        keep_train = "train.xyz" in keep and train_exists
        keep_val = "val.xyz" in keep and val_exists

        if keep_train and keep_val:
            print("Keeping existing train.xyz")
            print("Keeping existing val.xyz")
        else:
            if not keep_train and train_exists:
                print("Regenerating train.xyz")
            if not keep_val and val_exists:
                print("Regenerating val.xyz")
            for fname in ("train.xyz", "val.xyz"):
                if fname in keep and not (path / fname).exists():
                    print(
                        f"--keep {fname} requested but {fname} does not exist.  "
                        "Generating."
                    )
            self._split_and_write_xyz(path)

        # heads.json
        maybe_regenerate(
            path, "heads.json", keep,
            regenerate=lambda: self._write_heads_json(path),
        )

    def _split_and_write_xyz(self, path: Path) -> None:
        """Read pkl, clean constraints, split 80-20, write xyz files."""
        pkl_path = path / self.pkl_path
        print(f"Reading {pkl_path} ...")
        df = pd.read_pickle(str(pkl_path))
        structures = list(df.structure)
        print(f"  {len(structures)} structures loaded")

        # Clean constraints
        n_constrained = 0
        for struc in structures:
            if len(struc.constraints) != 0:
                n_constrained += 1
                forces_raw = struc.get_forces(apply_constraint=False)
                energy_raw = struc.get_potential_energy()
                stress_raw = struc.get_stress(apply_constraint=False)
                del struc.constraints
                struc.calc = SinglePointCalculator(
                    struc,
                    energy=energy_raw,
                    forces=forces_raw,
                    stress=stress_raw,
                )
        if n_constrained:
            print(f"  cleaned constraints from {n_constrained} structures")

        # Split
        train, val = train_test_split(
            structures, test_size=0.20, random_state=self.seed,
        )
        print(f"  train: {len(train)}  val: {len(val)}")

        # Write
        train_xyz = path / "train.xyz"
        val_xyz = path / "val.xyz"
        write(str(train_xyz), train, format="extxyz")
        write(str(val_xyz), val, format="extxyz")
        print(f"  wrote {train_xyz}")
        print(f"  wrote {val_xyz}")

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
        with open(heads_json, "w") as f:
            json.dump(heads, f, indent=2)
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

        See ``_wrap_command`` for the generated shell pattern.
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

        # -- Canned fine-tuning stages --
        parts = []

        if not self.no_fine_tuning_select:
            parts.append(_wrap_command(self._select_stage(path)))

        parts.append(_wrap_command(self._train_stage(path)))
        return "\n\n".join(parts)

    def _select_stage(self, path: Path) -> str:
        """Stage 1 command: fine_tuning_select."""
        return (
            'echo "=== Stage 1: fine_tuning_select ==="\n'
            "python -u -m mace.cli.fine_tuning_select \\\n"
            f"  --configs_pt {self.replay_xyz} \\\n"
            f"  --configs_ft {path / 'train.xyz'} \\\n"
            f"  --num_samples {self.num_samples} \\\n"
            "  --subselect fps \\\n"
            f"  --model {self.model_path} \\\n"
            f"  --output {path / 'selected_configs.xyz'} \\\n"
            "  --filtering_type exclusive \\\n"
            "  --head_pt pt_head \\\n"
            "  --head_ft target_head \\\n"
            "  --weight_pt 1.0 \\\n"
            "  --device cuda \\\n"
            "  --weight_ft 10.0 >> log_db"
        )

    def _train_stage(self, path: Path) -> str:
        """Stage 2 command: run_train."""
        heads_path = path / 'heads.json'
        return (
            'echo "=== Stage 2: run_train ==="\n'
            "python -u -m mace.cli.run_train \\\n"
            f"  --name {self.new_model_name} \\\n"
            "  --multiheads_finetuning True \\\n"
            f"  --heads $(cat {heads_path}) \\\n"
            "  --valid_fraction 0.05 \\\n"
            f"  --foundation_model {self.model_path} \\\n"
            "  --energy_weight 1.0 \\\n"
            "  --forces_weight 10.0 \\\n"
            "  --stress_weight 10.0 \\\n"
            "  --swa \\\n"
            "  --swa_lr 0.0001 \\\n"
            "  --start_swa 40 \\\n"
            "  --swa_energy_weight 10.0 \\\n"
            "  --swa_forces_weight 10.0 \\\n"
            "  --swa_stress_weight 10.0 \\\n"
            "  --force_mh_ft_lr=True \\\n"
            "  --lr 0.0009 \\\n"
            "  --weight_decay 5e-9 \\\n"
            "  --ema \\\n"
            "  --ema_decay 0.99999 \\\n"
            "  --device cuda \\\n"
            "  --default_dtype float64 \\\n"
            f"  --max_num_epochs {self.max_num_epochs} \\\n"
            f"  --batch_size {self.batch_size} \\\n"
            "  --compute_stress True \\\n"
            "  --loss stress \\\n"
            f"  --seed {self.seed} >> log_tuning"
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
