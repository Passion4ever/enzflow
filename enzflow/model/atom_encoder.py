"""Atom-level encoder: atom14 coordinates to token and atom representations.

Converts per-residue atom14 coordinates into:
- ``token_repr [B, N, d_token]``: one vector per residue (for Pairformer)
- ``atom_repr [B, N, 14, d_atom]``: per-atom features (skip-connected to decoder)

IntraResidueBlock is the original residue-internal-only attention.
CrossResidueBlock extends it with a sliding window over neighboring residues
(i-1, i, i+1) so that atoms across residue boundaries can directly interact,
enabling the model to learn peptide bond geometry and backbone dihedral angles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

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


class CrossResidueBlock(nn.Module):
    """Transformer layer with cross-residue sliding-window attention.

    For residue i, query (14 atoms) attends to atoms from residues
    i-1, i, i+1 (up to 42 atoms). This enables direct atomic interaction
    across residue boundaries for peptide bonds and backbone dihedral
    angles (phi/psi/omega).

    A learned residue-offset embedding (prev/curr/next) is added to key/value
    so the model can distinguish atoms from neighboring residues.
    """

    def __init__(self, d_atom: int, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        self.d_atom = d_atom
        self.n_heads = n_heads
        self.d_head = d_atom // n_heads

        self.norm1 = nn.LayerNorm(d_atom)

        self.q_proj = nn.Linear(d_atom, d_atom, bias=False)
        self.k_proj = nn.Linear(d_atom, d_atom, bias=False)
        self.v_proj = nn.Linear(d_atom, d_atom, bias=False)
        self.out_proj = nn.Linear(d_atom, d_atom, bias=False)

        # Residue offset embedding: 0 = prev (i-1), 1 = current (i), 2 = next (i+1)
        self.offset_embed = nn.Embedding(3, d_atom)

        self.norm2 = nn.LayerNorm(d_atom)
        self.ffn = nn.Sequential(
            nn.Linear(d_atom, d_atom * ffn_mult),
            nn.GELU(),
            nn.Linear(d_atom * ffn_mult, d_atom),
        )

    def forward(self, x: Tensor, atom_mask: Tensor) -> Tensor:
        """Forward pass with cross-residue attention.

        Args:
            x: ``[B, N, 14, d_atom]``
            atom_mask: ``[B, N, 14]`` -- bool, True for real atoms.

        Returns:
            ``[B, N, 14, d_atom]``
        """
        B, N, A, D = x.shape  # A = 14
        H = self.n_heads
        Dh = self.d_head

        # --- Pre-LN ---
        h = self.norm1(x)  # [B, N, 14, D]

        # --- Build context window from normalized features ---
        # Pad residue dimension (dim=1) by 1 on each side with zeros
        h_pad = F.pad(h, (0, 0, 0, 0, 1, 1))  # [B, N+2, 14, D]
        context = torch.cat(
            [h_pad[:, :N], h, h_pad[:, 2:N + 2]], dim=2,
        )  # [B, N, 42, D]

        # Add residue offset embedding so model knows which residue each atom is from
        offset_idx = torch.arange(3, device=x.device).repeat_interleave(A)  # [42]
        context = context + self.offset_embed(offset_idx)  # broadcast [42, D]

        # --- Build context mask ---
        mask_f = atom_mask.float()  # [B, N, 14]
        mask_pad = F.pad(mask_f, (0, 0, 1, 1), value=0.0)  # [B, N+2, 14]
        context_mask = torch.cat(
            [mask_pad[:, :N], mask_f, mask_pad[:, 2:N + 2]], dim=2,
        )  # [B, N, 42]

        # --- Cross-attention: Q from current residue, KV from window ---
        q = self.q_proj(h.reshape(B * N, A, D))
        kv = context.reshape(B * N, 3 * A, D)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = q.reshape(B * N, A, H, Dh).permute(0, 2, 1, 3)    # [B*N, H, 14, Dh]
        k = k.reshape(B * N, 3 * A, H, Dh).permute(0, 2, 1, 3)  # [B*N, H, 42, Dh]
        v = v.reshape(B * N, 3 * A, H, Dh).permute(0, 2, 1, 3)  # [B*N, H, 42, Dh]

        scale = Dh ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*N, H, 14, 42]

        # Apply context mask: [B, N, 42] -> [B*N, 1, 1, 42]
        ctx_mask = context_mask.reshape(B * N, 1, 1, 3 * A)
        attn = attn.masked_fill(ctx_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows with 0
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        out = torch.matmul(attn, v)  # [B*N, H, 14, Dh]
        out = out.permute(0, 2, 1, 3).reshape(B * N, A, D)
        out = self.out_proj(out).reshape(B, N, A, D)

        # Residual
        x = x + out

        # --- FFN ---
        h = self.norm2(x)
        x = x + self.ffn(h)

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

        # Cross-residue attention blocks (sliding window over i-1, i, i+1)
        self.blocks = nn.ModuleList([CrossResidueBlock(d_atom) for _ in range(n_layers)])

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

        # --- Cross-residue attention ---
        for block in self.blocks:
            atom_repr = block(atom_repr, atom_mask)

        # --- Aggregate to token level ---
        ca_feat = atom_repr[:, :, _CA_INDEX, :]  # [B, N, d_atom]
        token_repr = self.ca_proj(ca_feat)  # [B, N, d_token]
        token_repr = token_repr + self.aatype_embed(aatype)
        token_repr = token_repr + self.motif_embed(motif_mask.long())

        return token_repr, atom_repr
