"""Tests for IMDgroup.gorun.slurm (phase 1: sbatch_script only)."""

from __future__ import annotations

from IMDgroup.gorun.slurm import sbatch_script


def test_sbatch_script_empty_args() -> None:
    """With no args, the script carries only the fixed signal line."""
    assert sbatch_script("#!/usr/bin/env bash", {}, "echo hi") == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        "\n"
        "echo hi\n"
    )


def test_sbatch_script_single_arg() -> None:
    """A single sbatch arg is emitted with its key and quoted value."""
    assert sbatch_script(
        "#!/usr/bin/env bash", {"partition": "cpu"}, "echo hi"
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        "echo hi\n"
    )


def test_sbatch_script_multiple_args_preserve_order() -> None:
    """Args are emitted in dict insertion order, before the script."""
    assert sbatch_script(
        "#!/usr/bin/env bash",
        {"partition": "cpu", "time": "01:00:00"},
        "run.sh",
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        '#SBATCH --time="01:00:00"\n'
        "run.sh\n"
    )
