"""Tests for geometric feature computation."""

import torch
import pytest

from enzflow.data.featurizer import rbf_encode, compute_pair_features


class TestRBFEncode:
    """Tests for rbf_encode."""

    def test_output_shape(self):
        dist = torch.rand(10, 10)
        out = rbf_encode(dist, num_rbf=16)
        assert out.shape == (10, 10, 16)

    def test_output_shape_1d(self):
        dist = torch.rand(5)
        out = rbf_encode(dist, num_rbf=8)
        assert out.shape == (5, 8)

    def test_peak_at_center(self):
        """Distance equal to a kernel center should produce peak response."""
        d_min, d_max, num_rbf = 0.0, 20.0, 10
        mu = torch.linspace(d_min, d_max, num_rbf)

        # Test with a distance exactly at the 5th kernel center
        dist = mu[4].unsqueeze(0)  # distance = mu[4]
        out = rbf_encode(dist, d_min=d_min, d_max=d_max, num_rbf=num_rbf)

        # The 5th kernel should have the highest response
        assert out[0].argmax().item() == 4

    def test_values_in_zero_one(self):
        """RBF values should be in (0, 1]."""
        dist = torch.rand(100) * 22
        out = rbf_encode(dist, num_rbf=16)
        assert (out >= 0).all()
        assert (out <= 1.0 + 1e-6).all()

    def test_zero_distance(self):
        """Distance=0 should give max response at the first kernel."""
        out = rbf_encode(torch.tensor([0.0]), d_min=0.0, d_max=22.0, num_rbf=16)
        assert out[0].argmax().item() == 0


class TestComputePairFeatures:
    """Tests for compute_pair_features."""

    @pytest.fixture
    def sample_data(self):
        """10-residue protein with random coords."""
        torch.manual_seed(42)
        N = 10
        coords = torch.randn(N, 14, 3) * 5
        residue_index = torch.arange(1, N + 1)
        return coords, residue_index

    def test_output_keys(self, sample_data):
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        assert set(feats.keys()) == {"rel_pos", "ca_dist_rbf", "ca_unit_vec"}

    def test_rel_pos_shape(self, sample_data):
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        N = coords.shape[0]
        assert feats["rel_pos"].shape == (N, N)

    def test_rel_pos_range(self, sample_data):
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx, rel_pos_clamp=32)
        # After clamping and shifting, values should be in [0, 64]
        assert feats["rel_pos"].min() >= 0
        assert feats["rel_pos"].max() <= 64

    def test_rel_pos_diagonal_is_clamp(self, sample_data):
        """Self-pair: j - i = 0, after shift = clamp value."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx, rel_pos_clamp=32)
        # Diagonal should be 32 (0 + clamp)
        diag = feats["rel_pos"].diagonal()
        assert (diag == 32).all()

    def test_rel_pos_antisymmetric(self, sample_data):
        """rel_pos[i,j] + rel_pos[j,i] should equal 2 * clamp."""
        coords, res_idx = sample_data
        clamp = 32
        feats = compute_pair_features(coords, res_idx, rel_pos_clamp=clamp)
        rp = feats["rel_pos"]
        # For non-clamped entries: rp[i,j] + rp[j,i] = (j-i+c) + (i-j+c) = 2c
        assert (rp + rp.T == 2 * clamp).all()

    def test_rel_pos_with_gap(self):
        """Non-contiguous residue indices (missing residues)."""
        N = 5
        coords = torch.randn(N, 14, 3)
        res_idx = torch.tensor([1, 2, 3, 10, 11])  # gap at 4-9
        feats = compute_pair_features(coords, res_idx, rel_pos_clamp=5)
        # res 0 to res 3: 10 - 1 = 9, clamped to 5, shifted to 10
        assert feats["rel_pos"][0, 3].item() == 10

    def test_ca_dist_rbf_shape(self, sample_data):
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx, num_rbf=16)
        N = coords.shape[0]
        assert feats["ca_dist_rbf"].shape == (N, N, 16)

    def test_ca_dist_symmetric(self, sample_data):
        """CA distances should be symmetric: d(i,j) = d(j,i)."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        rbf = feats["ca_dist_rbf"]
        torch.testing.assert_close(rbf, rbf.transpose(0, 1))

    def test_ca_dist_self_zero(self, sample_data):
        """Self-distance should be 0, giving peak at first RBF kernel."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx, num_rbf=16)
        diag_rbf = feats["ca_dist_rbf"].diagonal(dim1=0, dim2=1)  # [16, N]
        # First kernel (d_min=0) should have highest response for self-pairs
        assert (diag_rbf.argmax(dim=0) == 0).all()

    def test_ca_unit_vec_shape(self, sample_data):
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        N = coords.shape[0]
        assert feats["ca_unit_vec"].shape == (N, N, 3)

    def test_ca_unit_vec_norm(self, sample_data):
        """Off-diagonal unit vectors should have norm ~1."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        uv = feats["ca_unit_vec"]
        N = coords.shape[0]
        norms = uv.norm(dim=-1)

        # Create off-diagonal mask
        off_diag = ~torch.eye(N, dtype=torch.bool)
        off_norms = norms[off_diag]
        torch.testing.assert_close(
            off_norms, torch.ones_like(off_norms), atol=1e-4, rtol=1e-4,
        )

    def test_ca_unit_vec_diagonal_zero(self, sample_data):
        """Self-pair direction vectors should be zero."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        diag = feats["ca_unit_vec"].diagonal(dim1=0, dim2=1)  # [3, N]
        assert (diag == 0).all()

    def test_ca_unit_vec_antisymmetric(self, sample_data):
        """Direction i->j should be opposite to j->i."""
        coords, res_idx = sample_data
        feats = compute_pair_features(coords, res_idx)
        uv = feats["ca_unit_vec"]
        torch.testing.assert_close(uv, -uv.transpose(0, 1), atol=1e-5, rtol=1e-5)
