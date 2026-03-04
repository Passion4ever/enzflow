"""Pair representation initialization from token features and geometry.

Combines four sources of pair information:
    1. Relative sequence position embedding
    2. CA-CA distance (RBF encoded)
    3. CA-CA unit direction vectors
    4. Outer product of projected token representations

All geometric features are computed from x_t (noisy coords) to avoid
information leakage during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from enzflow.data.featurizer import rbf_encode

_CA_INDEX = 1
_REL_POS_BINS = 65  # 2 * 32 + 1
_REL_POS_CLAMP = 32


class PairRepresentationInit(nn.Module):
    """Initialize pair representation from token features and coordinates.

    Produces ``pair_repr [B, N, N, d_pair]`` by summing four components.
    """

    def __init__(
        self,
        d_token: int = 256,
        d_pair: int = 128,
        num_rbf: int = 16,
        d_proj: int = 32,
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf

        # 1. Relative position
        self.rel_pos_embed = nn.Embedding(_REL_POS_BINS, d_pair)

        # 2. CA distance RBF
        self.dist_proj = nn.Linear(num_rbf, d_pair)

        # 3. CA direction
        self.dir_proj = nn.Linear(3, d_pair)

        # 4. Outer product of token features
        self.d_proj = d_proj
        self.proj_left = nn.Linear(d_token, d_proj)
        self.proj_right = nn.Linear(d_token, d_proj)
        self.outer_proj = nn.Linear(d_proj, d_pair)

    def forward(
        self,
        token_repr: Tensor,
        coords: Tensor,
        residue_index: Tensor,
        seq_mask: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            token_repr: ``[B, N, d_token]``
            coords: ``[B, N, 14, 3]`` -- atom14 coords (x_t).
            residue_index: ``[B, N]`` -- residue sequence numbers.
            seq_mask: ``[B, N]`` -- bool, True for real residues.

        Returns:
            ``pair_repr [B, N, N, d_pair]``
        """
        # --- 1. Relative sequence position ---
        # [B, N, 1] - [B, 1, N] -> [B, N, N]
        rel_pos = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        rel_pos = rel_pos.clamp(-_REL_POS_CLAMP, _REL_POS_CLAMP) + _REL_POS_CLAMP
        pair = self.rel_pos_embed(rel_pos)  # [B, N, N, d_pair]

        # --- 2. CA-CA distance (RBF) ---
        ca_coords = coords[:, :, _CA_INDEX, :]  # [B, N, 3]
        ca_diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)  # [B, N, N, 3]
        ca_dist = ca_diff.norm(dim=-1)  # [B, N, N]
        ca_rbf = rbf_encode(ca_dist, num_rbf=self.num_rbf)  # [B, N, N, num_rbf]
        pair = pair + self.dist_proj(ca_rbf)

        # --- 3. CA-CA direction ---
        ca_unit = ca_diff / (ca_dist.unsqueeze(-1) + 1e-8)  # [B, N, N, 3]
        # Zero self-pairs
        eye = torch.eye(ca_unit.shape[1], device=ca_unit.device, dtype=torch.bool)
        ca_unit = ca_unit.masked_fill(eye.unsqueeze(0).unsqueeze(-1), 0.0)
        pair = pair + self.dir_proj(ca_unit)

        # --- 4. Outer product ---
        left = self.proj_left(token_repr)   # [B, N, d_proj]
        right = self.proj_right(token_repr)  # [B, N, d_proj]
        # [B, N, 1, d_proj] * [B, 1, N, d_proj] -> [B, N, N, d_proj]
        outer = left.unsqueeze(2) * right.unsqueeze(1)
        pair = pair + self.outer_proj(outer)

        # Mask padding positions
        pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)  # [B, N, N]
        pair = pair * pair_mask.unsqueeze(-1).float()

        return pair
