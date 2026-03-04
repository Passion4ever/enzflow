"""Tests for AdaLN-Zero module."""

import torch

from enzflow.model.adaln import AdaLNZero

D_MODEL = 64
D_COND = 128
B, N = 2, 10


def _make_inputs():
    x = torch.randn(B, N, D_MODEL)
    cond = torch.randn(B, D_COND)
    return x, cond


def test_output_shapes():
    m = AdaLNZero(D_MODEL, D_COND)
    x, cond = _make_inputs()
    h, alpha = m(x, cond)
    assert h.shape == (B, N, D_MODEL)
    assert alpha.shape == (B, 1, D_MODEL)


def test_zero_init_gate_is_zero():
    """At init, the last linear is zero so alpha should be exactly 0."""
    m = AdaLNZero(D_MODEL, D_COND)
    x, cond = _make_inputs()
    _, alpha = m(x, cond)
    assert torch.allclose(alpha, torch.zeros_like(alpha))


def test_zero_init_h_equals_layernorm():
    """At init, gamma=0 and beta=0 so h = LayerNorm(x)."""
    m = AdaLNZero(D_MODEL, D_COND)
    x, cond = _make_inputs()
    h, _ = m(x, cond)
    expected = torch.nn.functional.layer_norm(x, [D_MODEL])
    assert torch.allclose(h, expected, atol=1e-6)


def test_gradient_flows():
    m = AdaLNZero(D_MODEL, D_COND)
    x, cond = _make_inputs()
    x.requires_grad_(True)
    cond.requires_grad_(True)
    h, alpha = m(x, cond)
    loss = (h * alpha).sum()
    loss.backward()
    assert x.grad is not None
    assert cond.grad is not None


def test_condition_affects_output():
    """Different cond vectors should produce different outputs (after training)."""
    m = AdaLNZero(D_MODEL, D_COND)
    # Manually set non-zero weights in last linear to break zero-init
    with torch.no_grad():
        m.proj[2].weight.fill_(0.1)
    x = torch.randn(1, N, D_MODEL)
    cond1 = torch.randn(1, D_COND)
    cond2 = torch.randn(1, D_COND)
    h1, _ = m(x, cond1)
    h2, _ = m(x, cond2)
    assert not torch.allclose(h1, h2, atol=1e-4)
