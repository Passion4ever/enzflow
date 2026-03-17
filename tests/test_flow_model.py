"""Tests for AllAtomFlowModel."""

import torch

from enzflow.model.flow_model import AllAtomFlowModel, SinusoidalTimeEmbedding

# Small config for testing
CFG = dict(
    d_token=64, d_pair=32, d_atom=32, d_cond=64,
    n_trunk=2, n_atom_layers=1, n_heads=4, d_ec_input=128,
)
B, N = 2, 10


def _make_batch(batch_size=B, seq_len=N, d_ec=128):
    return {
        "x_t": torch.randn(batch_size, seq_len, 14, 3),
        "t": torch.rand(batch_size),
        "atom_mask": torch.ones(batch_size, seq_len, 14, dtype=torch.bool),
        "aatype": torch.randint(0, 20, (batch_size, seq_len)),
        "residue_index": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        "ec_embed": torch.randn(batch_size, d_ec),
        "motif_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "seq_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


# --- SinusoidalTimeEmbedding ---

def test_time_embedding_shape():
    emb = SinusoidalTimeEmbedding(64)
    t = torch.rand(4)
    out = emb(t)
    assert out.shape == (4, 64)


def test_time_embedding_different_t():
    emb = SinusoidalTimeEmbedding(64)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    assert not torch.allclose(emb(t1), emb(t2))


# --- AllAtomFlowModel ---

def test_forward_shape():
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    v, seq_logits = model(**batch)
    assert v.shape == (B, N, 14, 3)
    assert seq_logits.shape == (B, N, 20)


def test_simulated_batch_dict():
    """Simulate a batch dict similar to what collate_fn produces."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    # Add partial masking
    batch["atom_mask"][:, :, 5:] = False
    batch["aatype"][:, 5:] = 20  # MASK
    batch["motif_mask"][:, :3] = True
    batch["seq_mask"][:, 8:] = False
    v, seq_logits = model(**batch)
    assert v.shape == (B, N, 14, 3)
    assert seq_logits.shape == (B, N, 20)
    assert not torch.isnan(v).any()
    assert not torch.isnan(seq_logits).any()


def test_backward_no_nan():
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    batch["x_t"].requires_grad_(True)
    v, seq_logits = model(**batch)
    loss = v.sum() + seq_logits.sum()
    loss.backward()
    assert batch["x_t"].grad is not None
    assert not torch.isnan(batch["x_t"].grad).any()


def test_param_count():
    """Check parameter count is in expected range for small config."""
    model = AllAtomFlowModel(**CFG)
    n_params = sum(p.numel() for p in model.parameters())
    # Small config: should be < 5M
    assert 100_000 < n_params < 5_000_000, f"Unexpected param count: {n_params}"


def test_virtual_atoms_zero():
    """Virtual atom velocities must be exactly zero."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    batch["atom_mask"][:, :, 5:] = False
    v, _ = model(**batch)
    virtual = ~batch["atom_mask"]
    assert (v[virtual.unsqueeze(-1).expand_as(v)] == 0).all()


def test_unconditional_ec_zero():
    """Model should work with ec_embed=0 (unconditional)."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    batch["ec_embed"].zero_()
    v, seq_logits = model(**batch)
    assert not torch.isnan(v).any()
    assert not torch.isnan(seq_logits).any()


def test_all_mask_aatype():
    """All aatype=20 (MASK) should not cause errors."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    batch["aatype"].fill_(20)
    v, seq_logits = model(**batch)
    assert not torch.isnan(v).any()
    assert not torch.isnan(seq_logits).any()


def test_motif_mode():
    """With motif conditioning enabled, model should still work."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    batch["motif_mask"][:, :5] = True
    batch["aatype"][:, 5:] = 20
    v, seq_logits = model(**batch)
    assert not torch.isnan(v).any()
    assert not torch.isnan(seq_logits).any()


def test_initial_velocity_near_zero():
    """At init, v_theta should be approximately zero (zero-init decoder)."""
    model = AllAtomFlowModel(**CFG)
    batch = _make_batch()
    v, _ = model(**batch)
    assert v.abs().max() < 1e-4, f"Max velocity at init: {v.abs().max()}"


def test_base_config_param_count():
    """Check parameter count for the base config (~28M)."""
    model = AllAtomFlowModel(
        d_token=256, d_pair=128, d_atom=128, d_cond=256,
        n_trunk=12, n_atom_layers=3, n_heads=8, d_ec_input=1024,
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Should be roughly 28M (allow 15-50M range)
    assert 15_000_000 < n_params < 50_000_000, f"Base config params: {n_params / 1e6:.1f}M"
