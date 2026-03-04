"""Tests for AtomEncoder."""

import torch

from enzflow.model.atom_encoder import AtomEncoder, IntraResidueBlock

D_TOKEN = 64
D_ATOM = 32
B, N = 2, 10


def _make_inputs(batch_size=B, seq_len=N):
    coords = torch.randn(batch_size, seq_len, 14, 3)
    atom_mask = torch.ones(batch_size, seq_len, 14, dtype=torch.bool)
    # GLY has only 4 atoms
    atom_mask[:, :, 4:] = False
    aatype = torch.randint(0, 20, (batch_size, seq_len))
    motif_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    motif_mask[:, :3] = True
    return coords, atom_mask, aatype, motif_mask


def test_intra_residue_block_shape():
    block = IntraResidueBlock(D_ATOM)
    x = torch.randn(B * N, 14, D_ATOM)
    out = block(x)
    assert out.shape == (B * N, 14, D_ATOM)


def test_encoder_output_shapes():
    enc = AtomEncoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=2)
    coords, atom_mask, aatype, motif_mask = _make_inputs()
    token_repr, atom_repr = enc(coords, atom_mask, aatype, motif_mask)
    assert token_repr.shape == (B, N, D_TOKEN)
    assert atom_repr.shape == (B, N, 14, D_ATOM)


def test_mask_token_no_error():
    """aatype=20 (MASK) should not cause index errors."""
    enc = AtomEncoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    coords, atom_mask, _, motif_mask = _make_inputs()
    aatype = torch.full((B, N), 20, dtype=torch.long)  # all MASK
    token_repr, atom_repr = enc(coords, atom_mask, aatype, motif_mask)
    assert not torch.isnan(token_repr).any()
    assert not torch.isnan(atom_repr).any()


def test_virtual_atoms_no_nan():
    """Masked-out (virtual) atoms should not produce NaN."""
    enc = AtomEncoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    coords, atom_mask, aatype, motif_mask = _make_inputs()
    # Make most atoms virtual
    atom_mask[:, :, 4:] = False
    coords[:, :, 4:] = 0.0
    token_repr, atom_repr = enc(coords, atom_mask, aatype, motif_mask)
    assert not torch.isnan(token_repr).any()
    assert not torch.isnan(atom_repr).any()


def test_gradient_flows():
    enc = AtomEncoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    coords, atom_mask, aatype, motif_mask = _make_inputs()
    coords.requires_grad_(True)
    token_repr, atom_repr = enc(coords, atom_mask, aatype, motif_mask)
    loss = token_repr.sum() + atom_repr.sum()
    loss.backward()
    assert coords.grad is not None
    assert not torch.isnan(coords.grad).any()


def test_different_lengths():
    """Encoder should handle different sequence lengths."""
    enc = AtomEncoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    for seq_len in [5, 20, 50]:
        coords, atom_mask, aatype, motif_mask = _make_inputs(batch_size=1, seq_len=seq_len)
        token_repr, atom_repr = enc(coords, atom_mask, aatype, motif_mask)
        assert token_repr.shape == (1, seq_len, D_TOKEN)
        assert atom_repr.shape == (1, seq_len, 14, D_ATOM)
