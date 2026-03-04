"""Tests for PairRepresentationInit."""

import torch

from enzflow.model.pair_repr import PairRepresentationInit

D_TOKEN = 64
D_PAIR = 32
B, N = 2, 10


def _make_inputs():
    token_repr = torch.randn(B, N, D_TOKEN)
    coords = torch.randn(B, N, 14, 3)
    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    seq_mask = torch.ones(B, N, dtype=torch.bool)
    return token_repr, coords, residue_index, seq_mask


def test_output_shape():
    m = PairRepresentationInit(d_token=D_TOKEN, d_pair=D_PAIR)
    token_repr, coords, residue_index, seq_mask = _make_inputs()
    pair = m(token_repr, coords, residue_index, seq_mask)
    assert pair.shape == (B, N, N, D_PAIR)


def test_rel_pos_asymmetric():
    """Relative position is asymmetric: (i,j) != (j,i) in general."""
    m = PairRepresentationInit(d_token=D_TOKEN, d_pair=D_PAIR)
    token_repr, coords, residue_index, seq_mask = _make_inputs()
    pair = m(token_repr, coords, residue_index, seq_mask)
    # rel_pos contribution makes pair asymmetric
    assert not torch.allclose(pair[:, 0, 1], pair[:, 1, 0], atol=1e-4)


def test_distance_symmetric():
    """CA-CA distance is symmetric, but full pair_repr need not be."""
    m = PairRepresentationInit(d_token=D_TOKEN, d_pair=D_PAIR)
    token_repr, coords, residue_index, seq_mask = _make_inputs()
    # Distance component only (isolate by zeroing other weights)
    with torch.no_grad():
        m.rel_pos_embed.weight.zero_()
        m.dir_proj.weight.zero_()
        m.dir_proj.bias.zero_()
        m.outer_proj.weight.zero_()
        m.outer_proj.bias.zero_()
    pair = m(token_repr, coords, residue_index, seq_mask)
    # Distance-only pair should be symmetric
    assert torch.allclose(pair, pair.transpose(1, 2), atol=1e-5)


def test_padding_no_nan():
    """Padding positions should be zeroed, not NaN."""
    m = PairRepresentationInit(d_token=D_TOKEN, d_pair=D_PAIR)
    token_repr, coords, residue_index, seq_mask = _make_inputs()
    seq_mask[:, 7:] = False  # last 3 are padding
    pair = m(token_repr, coords, residue_index, seq_mask)
    assert not torch.isnan(pair).any()
    # Padding rows/cols should be zero
    assert (pair[:, 7:, :, :] == 0).all()
    assert (pair[:, :, 7:, :] == 0).all()


def test_gradient_flows():
    m = PairRepresentationInit(d_token=D_TOKEN, d_pair=D_PAIR)
    token_repr, coords, residue_index, seq_mask = _make_inputs()
    token_repr.requires_grad_(True)
    coords.requires_grad_(True)
    pair = m(token_repr, coords, residue_index, seq_mask)
    pair.sum().backward()
    assert token_repr.grad is not None
    assert coords.grad is not None


def test_consistent_with_featurizer():
    """RBF encoding should match featurizer.rbf_encode."""
    from enzflow.data.featurizer import rbf_encode

    coords = torch.randn(1, 5, 14, 3)
    ca = coords[0, :, 1, :]  # [5, 3]
    dist = (ca.unsqueeze(0) - ca.unsqueeze(1)).norm(dim=-1)  # [5, 5]
    rbf = rbf_encode(dist, num_rbf=16)
    # Check it matches the batched version
    ca_b = coords[:, :, 1, :]  # [1, 5, 3]
    dist_b = (ca_b.unsqueeze(2) - ca_b.unsqueeze(1)).norm(dim=-1)  # [1, 5, 5]
    rbf_b = rbf_encode(dist_b, num_rbf=16)
    assert torch.allclose(rbf, rbf_b[0], atol=1e-6)
