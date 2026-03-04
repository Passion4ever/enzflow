"""Atom-level encoder: atom14 coordinates to token and atom representations.

Converts per-residue atom14 coordinates into:
- ``token_repr [B, N, d_token]``: one vector per residue (for Pairformer)
- ``atom_repr [B, N, 14, d_atom]``: per-atom features (skip-connected to decoder)

IntraResidueBlock is defined here and shared with the AtomDecoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from enzflow.data.residue_constants import (
    NUM_RES_TYPES_WITH_MASK,
    get_atom14_element_indices,
)

# CA is always at atom14 index 1
_CA_INDEX = 1
# Elements: C=0, N=1, O=2, S=3, unknown=4
_NUM_ELEMENTS = 5


class IntraResidueBlock(nn.Module):
    """Pre-LN transformer layer operating over the 14 atoms within a residue.

    Shared between AtomEncoder and AtomDecoder. The sequence length is always 14
    (the atom14 slots within one residue).
    """

    def __init__(self, d_atom: int, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_atom)
        self.attn = nn.MultiheadAttention(
            d_atom, n_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_atom)
        self.ffn = nn.Sequential(
            nn.Linear(d_atom, d_atom * ffn_mult),
            nn.GELU(),
            nn.Linear(d_atom * ffn_mult, d_atom),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``[B*N, 14, d_atom]``

        Returns:
            ``[B*N, 14, d_atom]``
        """
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class AtomEncoder(nn.Module):
    """Encode atom14 coordinates into token and atom representations.

    Input embedding per atom (concatenated then projected):
    - coords: 3D
    - mask_float: 1D
    - position_embed: ``Embedding(14, d_atom)`` -- atom position within residue
    - element_embed: ``Embedding(5, d_atom)`` -- C/N/O/S/unknown

    After intra-residue attention, CA features are projected to token_repr
    and augmented with aatype and motif embeddings.
    """

    def __init__(
        self,
        d_token: int = 256,
        d_atom: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.d_token = d_token
        self.d_atom = d_atom

        # Atom input embeddings
        self.position_embed = nn.Embedding(14, d_atom)
        self.element_embed = nn.Embedding(_NUM_ELEMENTS, d_atom)
        self.input_proj = nn.Linear(d_atom * 2 + 4, d_atom)

        # Intra-residue attention blocks
        self.blocks = nn.ModuleList([IntraResidueBlock(d_atom) for _ in range(n_layers)])

        # Aggregation: CA feature -> token repr
        self.ca_proj = nn.Linear(d_atom, d_token)
        self.aatype_embed = nn.Embedding(NUM_RES_TYPES_WITH_MASK, d_token)
        self.motif_embed = nn.Embedding(2, d_token)

        # Register element index buffer [20, 14]
        self.register_buffer("elem_indices", get_atom14_element_indices())

    def forward(
        self,
        coords: Tensor,
        atom_mask: Tensor,
        aatype: Tensor,
        motif_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            coords: ``[B, N, 14, 3]`` -- atom14 coordinates (x_t).
            atom_mask: ``[B, N, 14]`` -- bool, True for real atoms.
            aatype: ``[B, N]`` -- long, 0-19 or 20 (MASK token).
            motif_mask: ``[B, N]`` -- bool, True for motif residues.

        Returns:
            ``(token_repr, atom_repr)`` where
            ``token_repr [B, N, d_token]`` and ``atom_repr [B, N, 14, d_atom]``.
        """
        B, N = coords.shape[:2]

        # --- Per-atom input features ---
        mask_float = atom_mask.float().unsqueeze(-1)  # [B, N, 14, 1]

        # Position embedding: [14] -> [1, 1, 14, d_atom]
        pos_idx = torch.arange(14, device=coords.device)
        pos_emb = self.position_embed(pos_idx).unsqueeze(0).unsqueeze(0)  # [1, 1, 14, d]

        # Element embedding: lookup from aatype
        # aatype can be 20 (MASK) -> clamp to [0, 19] for table lookup
        aa_clamped = aatype.clamp(max=19)  # [B, N]
        elem_idx = self.elem_indices[aa_clamped]  # [B, N, 14]
        # For MASK tokens (aatype==20), map all elements to unknown (idx=4)
        is_mask = (aatype == 20).unsqueeze(-1).expand_as(elem_idx)  # [B, N, 14]
        elem_idx = elem_idx.masked_fill(is_mask, _NUM_ELEMENTS - 1)
        elem_emb = self.element_embed(elem_idx)  # [B, N, 14, d_atom]

        # Concatenate: [coords(3) + mask(1) + pos_emb(d) + elem_emb(d)] -> project
        atom_input = torch.cat([coords, mask_float, pos_emb.expand(B, N, -1, -1), elem_emb], dim=-1)
        atom_repr = self.input_proj(atom_input)  # [B, N, 14, d_atom]

        # --- Intra-residue attention ---
        atom_repr = atom_repr.reshape(B * N, 14, self.d_atom)
        for block in self.blocks:
            atom_repr = block(atom_repr)
        atom_repr = atom_repr.reshape(B, N, 14, self.d_atom)

        # --- Aggregate to token level ---
        ca_feat = atom_repr[:, :, _CA_INDEX, :]  # [B, N, d_atom]
        token_repr = self.ca_proj(ca_feat)  # [B, N, d_token]
        token_repr = token_repr + self.aatype_embed(aatype)
        token_repr = token_repr + self.motif_embed(motif_mask.long())

        return token_repr, atom_repr
