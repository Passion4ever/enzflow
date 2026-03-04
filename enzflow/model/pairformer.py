"""Pairformer block: the core transformer layer of the model.

Each block updates both the token and pair representations:
- Token track: AdaLN -> PairBiasedAttention -> AdaLN -> SwiGLU FFN
- Pair track: PairTransition (simple MLP)

Key design choices:
- QK LayerNorm prevents attention logit explosion in deep stacks
- All output projections are zero-initialized
- AdaLN gates start at 0 (identity at init)
- No triangle updates (Proteina showed they are not necessary)
- No attention dropout (flow matching models typically skip it)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from enzflow.model.adaln import AdaLNZero


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with zero-initialized output.

    SwiGLU: SiLU(xW1a) * xW1b -> W2 (Shazeer 2020).
    """

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            # Common heuristic: 8/3 * d_model rounded to multiple of 8
            d_ff = int(8 / 3 * d_model + 7) // 8 * 8
        self.w1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class PairBiasedAttention(nn.Module):
    """Multi-head attention with pair representation bias and QK LayerNorm.

    Attention logits = Q @ K^T / sqrt(d_head) + pair_bias
    where pair_bias comes from projecting pair_repr to n_heads scalars.
    QK LayerNorm (per-head) prevents logit explosion in deep stacks.
    """

    def __init__(self, d_model: int, d_pair: int, n_heads: int = 8) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK LayerNorm (per-head)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

        # Pair bias: [B, N, N, d_pair] -> [B, n_heads, N, N]
        self.pair_bias_proj = nn.Linear(d_pair, n_heads, bias=False)

        # Zero-init output
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self, x: Tensor, pair_repr: Tensor, seq_mask: Tensor
    ) -> Tensor:
        """Forward pass.

        Args:
            x: ``[B, N, d_model]`` -- query/key/value source.
            pair_repr: ``[B, N, N, d_pair]`` -- pair features for bias.
            seq_mask: ``[B, N]`` -- bool, True for real residues.

        Returns:
            ``[B, N, d_model]``
        """
        B, N, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Project Q, K, V and reshape to [B, H, N, D]
        q = self.q_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)

        # QK LayerNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention logits
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add pair bias
        pair_bias = self.pair_bias_proj(pair_repr)  # [B, N, N, H]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        attn = attn + pair_bias

        # Key padding mask: prevent attending to padding positions
        # seq_mask [B, N] -> [B, 1, 1, N]
        key_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows with 0
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        out = torch.matmul(attn, v)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * D)
        return self.out_proj(out)


class PairTransition(nn.Module):
    """Simple MLP transition for pair representation."""

    def __init__(self, d_pair: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair * mult),
            nn.SiLU(),
            nn.Linear(d_pair * mult, d_pair),
        )

    def forward(self, pair_repr: Tensor) -> Tensor:
        return self.net(pair_repr)


class PairformerBlock(nn.Module):
    """Single Pairformer block updating both token and pair representations.

    Token track: AdaLN -> Attention -> residual -> AdaLN -> FFN -> residual
    Pair track: PairTransition -> residual
    """

    def __init__(
        self,
        d_token: int = 256,
        d_pair: int = 128,
        d_cond: int = 256,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        # Token track
        self.adaln_attn = AdaLNZero(d_token, d_cond)
        self.attention = PairBiasedAttention(d_token, d_pair, n_heads)
        self.adaln_ffn = AdaLNZero(d_token, d_cond)
        self.ffn = SwiGLUFFN(d_token)

        # Pair track
        self.pair_transition = PairTransition(d_pair)

    def forward(
        self,
        token_repr: Tensor,
        pair_repr: Tensor,
        cond: Tensor,
        seq_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            token_repr: ``[B, N, d_token]``
            pair_repr: ``[B, N, N, d_pair]``
            cond: ``[B, d_cond]`` -- time + EC condition vector.
            seq_mask: ``[B, N]`` -- bool, True for real residues.

        Returns:
            ``(token_repr, pair_repr)`` -- updated representations.
        """
        # Token: attention
        h, gate1 = self.adaln_attn(token_repr, cond)
        token_repr = token_repr + gate1 * self.attention(h, pair_repr, seq_mask)

        # Token: FFN
        h, gate2 = self.adaln_ffn(token_repr, cond)
        token_repr = token_repr + gate2 * self.ffn(h)

        # Pair: transition
        pair_repr = pair_repr + self.pair_transition(pair_repr)

        return token_repr, pair_repr
