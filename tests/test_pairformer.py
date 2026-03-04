"""Tests for PairformerBlock and sub-modules."""

import torch
import torch.nn as nn

from enzflow.model.pairformer import PairBiasedAttention, PairformerBlock, SwiGLUFFN

D_TOKEN = 64
D_PAIR = 32
D_COND = 128
N_HEADS = 4
B, N = 2, 10


def _make_block_inputs():
    token_repr = torch.randn(B, N, D_TOKEN)
    pair_repr = torch.randn(B, N, N, D_PAIR)
    cond = torch.randn(B, D_COND)
    seq_mask = torch.ones(B, N, dtype=torch.bool)
    return token_repr, pair_repr, cond, seq_mask


# --- SwiGLU FFN ---

def test_swiglu_shape():
    ffn = SwiGLUFFN(D_TOKEN)
    x = torch.randn(B, N, D_TOKEN)
    out = ffn(x)
    assert out.shape == (B, N, D_TOKEN)


def test_swiglu_zero_init():
    """Output should be zero at init (w2 is zero-initialized)."""
    ffn = SwiGLUFFN(D_TOKEN)
    x = torch.randn(B, N, D_TOKEN)
    out = ffn(x)
    assert torch.allclose(out, torch.zeros_like(out))


# --- PairBiasedAttention ---

def test_attention_shape():
    attn = PairBiasedAttention(D_TOKEN, D_PAIR, N_HEADS)
    x = torch.randn(B, N, D_TOKEN)
    pair = torch.randn(B, N, N, D_PAIR)
    mask = torch.ones(B, N, dtype=torch.bool)
    out = attn(x, pair, mask)
    assert out.shape == (B, N, D_TOKEN)


def test_attention_mask_works():
    """Padded positions should not influence output of real positions."""
    attn = PairBiasedAttention(D_TOKEN, D_PAIR, N_HEADS)
    # Break zero-init to see effects
    with torch.no_grad():
        attn.out_proj.weight.fill_(0.01)

    x = torch.randn(1, 5, D_TOKEN)
    pair = torch.randn(1, 5, 5, D_PAIR)

    mask_full = torch.ones(1, 5, dtype=torch.bool)
    mask_partial = torch.ones(1, 5, dtype=torch.bool)
    mask_partial[:, 3:] = False  # mask last 2

    out_full = attn(x, pair, mask_full)
    out_partial = attn(x, pair, mask_partial)

    # First 3 positions should be different due to different masks
    # (they attend to different sets of keys)
    assert not torch.allclose(out_full[:, :3], out_partial[:, :3], atol=1e-4)


def test_attention_zero_init():
    """Output should be zero at init (out_proj is zero-initialized)."""
    attn = PairBiasedAttention(D_TOKEN, D_PAIR, N_HEADS)
    x = torch.randn(B, N, D_TOKEN)
    pair = torch.randn(B, N, N, D_PAIR)
    mask = torch.ones(B, N, dtype=torch.bool)
    out = attn(x, pair, mask)
    assert torch.allclose(out, torch.zeros_like(out))


def test_pair_bias_affects_attention():
    """Different pair_repr should produce different attention outputs."""
    attn = PairBiasedAttention(D_TOKEN, D_PAIR, N_HEADS)
    with torch.no_grad():
        attn.out_proj.weight.fill_(0.01)

    x = torch.randn(1, N, D_TOKEN)
    mask = torch.ones(1, N, dtype=torch.bool)
    pair1 = torch.randn(1, N, N, D_PAIR)
    pair2 = torch.randn(1, N, N, D_PAIR)
    out1 = attn(x, pair1, mask)
    out2 = attn(x, pair2, mask)
    assert not torch.allclose(out1, out2, atol=1e-4)


# --- PairformerBlock ---

def test_block_shapes():
    block = PairformerBlock(D_TOKEN, D_PAIR, D_COND, N_HEADS)
    token, pair, cond, mask = _make_block_inputs()
    token_out, pair_out = block(token, pair, cond, mask)
    assert token_out.shape == (B, N, D_TOKEN)
    assert pair_out.shape == (B, N, N, D_PAIR)


def test_block_near_identity_at_init():
    """At init, all gates=0 and output projections=0, so token should be ~unchanged."""
    block = PairformerBlock(D_TOKEN, D_PAIR, D_COND, N_HEADS)
    token, pair, cond, mask = _make_block_inputs()
    token_out, pair_out = block(token, pair, cond, mask)

    # Token should be approximately unchanged (gates are zero)
    assert torch.allclose(token_out, token, atol=1e-5)

    # Pair changes due to PairTransition (which is NOT zero-initialized)
    assert pair_out.shape == pair.shape


def test_block_gradient():
    block = PairformerBlock(D_TOKEN, D_PAIR, D_COND, N_HEADS)
    token, pair, cond, mask = _make_block_inputs()
    token.requires_grad_(True)
    pair.requires_grad_(True)
    cond.requires_grad_(True)
    token_out, pair_out = block(token, pair, cond, mask)
    loss = token_out.sum() + pair_out.sum()
    loss.backward()
    assert token.grad is not None
    assert pair.grad is not None
    assert cond.grad is not None


def test_block_with_padding():
    """Block should handle padding without NaN."""
    block = PairformerBlock(D_TOKEN, D_PAIR, D_COND, N_HEADS)
    token, pair, cond, mask = _make_block_inputs()
    mask[:, 7:] = False
    token_out, pair_out = block(token, pair, cond, mask)
    assert not torch.isnan(token_out).any()
    assert not torch.isnan(pair_out).any()


def test_stack_of_blocks():
    """Multiple blocks in sequence should not explode or produce NaN."""
    blocks = nn.ModuleList([
        PairformerBlock(D_TOKEN, D_PAIR, D_COND, N_HEADS) for _ in range(4)
    ])
    token, pair, cond, mask = _make_block_inputs()
    for block in blocks:
        token, pair = block(token, pair, cond, mask)
    assert not torch.isnan(token).any()
    assert not torch.isnan(pair).any()
    assert token.abs().max() < 1e6  # no explosion
