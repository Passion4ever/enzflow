"""Atom-level decoder: token + atom representations to velocity field.

Takes the updated token_repr from the Pairformer trunk plus the
skip-connected atom_repr from the encoder, and produces per-atom
velocity predictions v_theta [B, N, 14, 3].

The final linear layer is zero-initialized so that the initial
velocity prediction is approximately zero (identity flow at init).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from enzflow.model.atom_encoder import CrossResidueBlock


class AtomDecoder(nn.Module):
    """Decode token + atom representations into atom14 velocities.

    Pipeline:
        1. Project token_repr to atom-level and broadcast to 14 atoms
        2. Add skip-connected atom_repr from encoder
        3. Intra-residue attention blocks
        4. Project to 3D velocity (zero-init)
        5. Mask virtual atoms to zero velocity
    """

    def __init__(
        self,
        d_token: int = 256,
        d_atom: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.d_atom = d_atom

        self.token_to_atom = nn.Linear(d_token, d_atom)
        self.blocks = nn.ModuleList([CrossResidueBlock(d_atom) for _ in range(n_layers)])
        self.to_velocity = nn.Linear(d_atom, 3)

        # Zero-init velocity projection: initial v_theta ~ 0
        nn.init.zeros_(self.to_velocity.weight)
        nn.init.zeros_(self.to_velocity.bias)

    def forward(
        self,
        token_repr: Tensor,
        atom_repr: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            token_repr: ``[B, N, d_token]`` -- from Pairformer trunk.
            atom_repr: ``[B, N, 14, d_atom]`` -- skip from encoder.
            atom_mask: ``[B, N, 14]`` -- bool, True for real atoms.

        Returns:
            ``v_theta [B, N, 14, 3]`` -- predicted velocity field.
        """
        B, N = token_repr.shape[:2]

        # Broadcast token -> atom level
        token_broadcast = self.token_to_atom(token_repr)  # [B, N, d_atom]
        token_broadcast = token_broadcast.unsqueeze(2).expand(-1, -1, 14, -1)

        # Skip connection
        atom_feat = atom_repr + token_broadcast  # [B, N, 14, d_atom]

        # Cross-residue attention
        for block in self.blocks:
            atom_feat = block(atom_feat, atom_mask)

        # Project to velocity
        v = self.to_velocity(atom_feat)  # [B, N, 14, 3]

        # Zero out virtual atoms
        v = v * atom_mask.float().unsqueeze(-1)

        return v
