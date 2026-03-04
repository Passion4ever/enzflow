"""Tests for AtomDecoder."""

import torch

from enzflow.model.atom_decoder import AtomDecoder

D_TOKEN = 64
D_ATOM = 32
B, N = 2, 10


def _make_inputs():
    token_repr = torch.randn(B, N, D_TOKEN)
    atom_repr = torch.randn(B, N, 14, D_ATOM)
    atom_mask = torch.ones(B, N, 14, dtype=torch.bool)
    atom_mask[:, :, 5:] = False  # only 5 real atoms
    return token_repr, atom_repr, atom_mask


def test_output_shape():
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=2)
    token_repr, atom_repr, atom_mask = _make_inputs()
    v = dec(token_repr, atom_repr, atom_mask)
    assert v.shape == (B, N, 14, 3)


def test_virtual_atoms_zero():
    """Velocity at masked (virtual) atom positions must be exactly 0."""
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=2)
    token_repr, atom_repr, atom_mask = _make_inputs()
    v = dec(token_repr, atom_repr, atom_mask)
    virtual = ~atom_mask
    assert (v[virtual.unsqueeze(-1).expand_as(v)] == 0).all()


def test_zero_init_velocity_near_zero():
    """At initialization, velocity should be approximately zero."""
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    token_repr, atom_repr, atom_mask = _make_inputs()
    v = dec(token_repr, atom_repr, atom_mask)
    assert v.abs().max() < 1e-5


def test_skip_connection_matters():
    """atom_repr (skip) should affect the output."""
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    # Break zero-init to see the effect
    with torch.no_grad():
        dec.to_velocity.weight.fill_(0.01)
    token_repr = torch.randn(1, N, D_TOKEN)
    atom_mask = torch.ones(1, N, 14, dtype=torch.bool)
    atom_repr1 = torch.randn(1, N, 14, D_ATOM)
    atom_repr2 = torch.randn(1, N, 14, D_ATOM)
    v1 = dec(token_repr, atom_repr1, atom_mask)
    v2 = dec(token_repr, atom_repr2, atom_mask)
    assert not torch.allclose(v1, v2, atol=1e-4)


def test_gradient_flows_to_encoder():
    """Gradients should flow back through atom_repr to encoder."""
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=1)
    token_repr = torch.randn(B, N, D_TOKEN, requires_grad=True)
    atom_repr = torch.randn(B, N, 14, D_ATOM, requires_grad=True)
    atom_mask = torch.ones(B, N, 14, dtype=torch.bool)
    v = dec(token_repr, atom_repr, atom_mask)
    v.sum().backward()
    assert token_repr.grad is not None
    assert atom_repr.grad is not None


def test_no_nan():
    dec = AtomDecoder(d_token=D_TOKEN, d_atom=D_ATOM, n_layers=2)
    token_repr, atom_repr, atom_mask = _make_inputs()
    v = dec(token_repr, atom_repr, atom_mask)
    assert not torch.isnan(v).any()
