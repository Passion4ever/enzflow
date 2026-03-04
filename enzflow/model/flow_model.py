"""All-atom flow model: the full architecture for Rectified Flow Matching.

Integrates all sub-modules into a single forward pass:
    x_t [B,N,14,3] -> AtomEncoder -> PairInit -> PairformerBlock x n_trunk
    -> AtomDecoder -> v_theta [B,N,14,3]

Conditioning: sinusoidal time embedding + bottleneck EC embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from enzflow.model.atom_decoder import AtomDecoder
from enzflow.model.atom_encoder import AtomEncoder
from enzflow.model.pair_repr import PairRepresentationInit
from enzflow.model.pairformer import PairformerBlock


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t in [0, 1]."""

    def __init__(self, d_embed: int) -> None:
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: ``[B]`` -- timestep values in [0, 1].

        Returns:
            ``[B, d_embed]``
        """
        half = self.d_embed // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        # t: [B] -> [B, 1] * [half] -> [B, half]
        args = t.unsqueeze(-1) * freq
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AllAtomFlowModel(nn.Module):
    """Full all-atom flow matching model.

    Architecture:
        1. Conditioning: time MLP + EC bottleneck -> cond [B, d_cond]
        2. AtomEncoder: x_t -> token_repr + atom_repr
        3. PairInit: token_repr + coords -> pair_repr
        4. Pairformer trunk: n_trunk blocks
        5. AtomDecoder: token_repr + atom_repr (skip) -> v_theta
    """

    def __init__(
        self,
        d_token: int = 256,
        d_pair: int = 128,
        d_atom: int = 128,
        d_cond: int = 256,
        n_trunk: int = 12,
        n_atom_layers: int = 3,
        n_heads: int = 8,
        d_ec_input: int = 1024,
    ) -> None:
        super().__init__()

        # --- Conditioning ---
        self.time_embed = SinusoidalTimeEmbedding(d_cond)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        # EC bottleneck: 1024 -> 128 -> 64 -> d_cond
        self.ec_bottleneck = nn.Sequential(
            nn.Linear(d_ec_input, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, d_cond),
        )

        # --- Encoder ---
        self.atom_encoder = AtomEncoder(
            d_token=d_token, d_atom=d_atom, n_layers=n_atom_layers,
        )

        # --- Pair init ---
        self.pair_init = PairRepresentationInit(
            d_token=d_token, d_pair=d_pair,
        )

        # --- Pairformer trunk ---
        self.trunk = nn.ModuleList([
            PairformerBlock(
                d_token=d_token, d_pair=d_pair, d_cond=d_cond, n_heads=n_heads,
            )
            for _ in range(n_trunk)
        ])

        # --- Final norm ---
        self.final_norm = nn.LayerNorm(d_token)

        # --- Decoder ---
        self.atom_decoder = AtomDecoder(
            d_token=d_token, d_atom=d_atom, n_layers=n_atom_layers,
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        atom_mask: Tensor,
        aatype: Tensor,
        residue_index: Tensor,
        ec_embed: Tensor,
        motif_mask: Tensor,
        seq_mask: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x_t: ``[B, N, 14, 3]`` -- noisy atom14 coordinates.
            t: ``[B]`` -- timestep in [0, 1].
            atom_mask: ``[B, N, 14]`` -- bool, True for real atoms.
            aatype: ``[B, N]`` -- amino acid type (0-19 or 20=MASK).
            residue_index: ``[B, N]`` -- residue sequence numbers.
            ec_embed: ``[B, d_ec_input]`` -- EC number embedding.
            motif_mask: ``[B, N]`` -- bool, True for motif residues.
            seq_mask: ``[B, N]`` -- bool, True for real residues.

        Returns:
            ``v_theta [B, N, 14, 3]`` -- predicted velocity field.
        """
        # 1. Conditioning
        cond = self.time_mlp(self.time_embed(t)) + self.ec_bottleneck(ec_embed)

        # 2. Encode
        token_repr, atom_repr = self.atom_encoder(x_t, atom_mask, aatype, motif_mask)

        # 3. Pair init
        pair_repr = self.pair_init(token_repr, x_t, residue_index, seq_mask)

        # 4. Trunk
        for block in self.trunk:
            token_repr, pair_repr = block(token_repr, pair_repr, cond, seq_mask)

        # 5. Final norm
        token_repr = self.final_norm(token_repr)

        # 6. Decode
        v_theta = self.atom_decoder(token_repr, atom_repr, atom_mask)

        return v_theta
