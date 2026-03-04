"""Adaptive Layer Normalization with zero-init gating (AdaLN-Zero).

Used in every PairformerBlock to inject time/EC conditioning into the
token representation. At initialization, gate=0 so each block acts as
an identity -- a standard trick for training deep residual networks.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class AdaLNZero(nn.Module):
    """AdaLN-Zero: condition-dependent normalization with zero-initialized gate.

    Given condition vector ``cond [B, d_cond]`` and input ``x [B, N, d_model]``,
    produces normalized-and-shifted ``h`` and a per-channel gate ``alpha``.

    The last linear layer is zero-initialized so that at the start of training
    ``alpha = 0`` (gate closed) and ``h = LayerNorm(x)`` (no shift/scale).
    """

    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, 3 * d_model),
        )
        # Zero-init the last linear layer
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: ``[B, N, d_model]``
            cond: ``[B, d_cond]``

        Returns:
            ``(h, alpha)`` where ``h [B, N, d_model]`` is the normalized
            and modulated input, ``alpha [B, 1, d_model]`` is the gate.
        """
        gamma, beta, alpha = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        h = self.norm(x) * (1 + gamma) + beta
        return h, alpha
