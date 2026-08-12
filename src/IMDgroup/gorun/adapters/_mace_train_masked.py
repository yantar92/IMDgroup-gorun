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

"""Wrapper around ``mace.cli.run_train`` that replaces the
energy/forces/stress loss with a mask-aware variant.

When ``ref.atom_mask`` is present on a batch (per-atom float tensor,
1 = supervised, 0 = masked), masked atoms' site energies and atomic
stresses are detached from autograd, preventing direct energy/stress
training on those atoms while preserving indirect training through
neighbour message passing.  When ``ref.atom_mask`` is absent (e.g.
pt_head replay data), the standard loss is used unchanged.

Usage:  python -u -m IMDgroup.gorun.adapters._mace_train_masked [ARGS...]

Does *not* require a custom ``--loss`` value -- keep using
``--loss stress``.  The ``configure_model`` stress check and SWA
loss selection both work because the original class is replaced
in-place.
"""

from __future__ import annotations

import logging

import torch

import mace.modules.loss as mace_loss
from mace.tools.scatter import scatter_sum

logger = logging.getLogger(__name__)

#: Threshold above which a mask entry is considered "supervised"
#: (contributes to the loss).  Mask values are 0 (masked) or 1
#: (supervised), but a float comparison avoids strict equality.
MASK_THRESHOLD: float = 0.5


## Masked loss class (replaces WeightedEnergyForcesStressLoss)

class MaskedEnergyForcesStressLoss(
    mace_loss.WeightedEnergyForcesStressLoss,
):
    """Standard E+F+S loss with per-atom masking when ``ref.atom_mask``
    is present on the batch."""

    def forward(
        self,
        ref,
        pred,
        ddp: bool | None = None,
    ) -> torch.Tensor:
        mask = getattr(ref, 'atom_mask', None)
        if mask is None:
            return super().forward(ref, pred, ddp)

        return (
            self.energy_weight * _masked_energy(ref, pred, mask, ddp)
            + self.forces_weight * _masked_forces(ref, pred, mask, ddp)
            + self.stress_weight * _masked_stress(ref, pred, mask, ddp)
        )


## Replace the class in-place so that get_loss_fn and get_swa
## pick up the masked variant automatically.

mace_loss.WeightedEnergyForcesStressLoss = MaskedEnergyForcesStressLoss


## Masked helpers

def _masked_energy(
    ref,
    pred,
    mask: torch.Tensor,
    ddp: bool | None,
) -> torch.Tensor:
    """Compute masked energy loss.

    Node energies of masked atoms are detached from autograd
    so they contribute to the per-structure total (and thus
    indirectly to training via message passing) but their
    site-energy gradient is zero.
    """
    supervised_float = (
        (mask > MASK_THRESHOLD)
        .to(dtype=pred['node_energy'].dtype)
    )
    node_energy = pred['node_energy']
    hybrid_node_energy = (
        node_energy * supervised_float
        + node_energy.detach() * (1.0 - supervised_float)
    )

    predicted_energy = scatter_sum(
        hybrid_node_energy,
        ref['batch'],
        dim=0,
        dim_size=ref.num_graphs,
    )
    natoms_per_config = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square(
            (ref['energy'] - predicted_energy) / natoms_per_config,
        )
    )
    return mace_loss.reduce_loss(raw_loss, ddp)


def _masked_forces(
    ref,
    pred,
    mask: torch.Tensor,
    ddp: bool | None,
) -> torch.Tensor:
    """Compute masked forces loss.

    Masked atoms get zero gradient for the force residual --
    only supervised atoms drive the force loss.
    """
    mask_2d = mask.view(-1, 1)
    natoms_per_config = ref.ptr[1:] - ref.ptr[:-1]
    config_weight = torch.repeat_interleave(
        ref.weight, natoms_per_config,
    ).unsqueeze(-1)
    config_forces_weight = torch.repeat_interleave(
        ref.forces_weight, natoms_per_config,
    ).unsqueeze(-1)
    raw_loss = (
        config_weight
        * config_forces_weight
        * mask_2d
        * torch.square(ref['forces'] - pred['forces'])
    )
    return mace_loss.reduce_loss(raw_loss, ddp)


def _masked_stress(
    ref,
    pred,
    mask: torch.Tensor,
    ddp: bool | None,
) -> torch.Tensor:
    """Compute masked stress loss.

    When atomic stresses are available, masked atoms' stress
    contributions are detached.  Falls back to the standard
    per-structure stress loss when atomic stresses are absent.
    """
    if (
        'atomic_stresses' not in pred
        or pred['atomic_stresses'] is None
    ):
        return mace_loss.weighted_mean_squared_stress(ref, pred, ddp)

    supervised_float = (
        (mask > MASK_THRESHOLD)
        .to(dtype=pred['atomic_stresses'].dtype)
        .view(-1, 1, 1)
    )
    atomic_stresses = pred['atomic_stresses']
    hybrid_atomic_stresses = (
        atomic_stresses * supervised_float
        + atomic_stresses.detach() * (1.0 - supervised_float)
    )

    predicted_stress = scatter_sum(
        hybrid_atomic_stresses,
        ref['batch'],
        dim=0,
        dim_size=ref.num_graphs,
    )
    config_weight = ref.weight.view(-1, 1, 1)
    config_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        config_weight
        * config_stress_weight
        * torch.square(ref['stress'] - predicted_stress)
    )
    return mace_loss.reduce_loss(raw_loss, ddp)


## Delegate to MACE

if __name__ == '__main__':
    from mace.cli.run_train import main
    main()
