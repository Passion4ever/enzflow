"""Tests for SE(3) data augmentation."""

import torch
import pytest

from enzflow.data.transforms import random_rotation_matrix, random_se3_augmentation


class TestRandomRotationMatrix:
    """Tests for random_rotation_matrix."""

    def test_shape(self):
        R = random_rotation_matrix()
        assert R.shape == (3, 3)

    def test_orthogonal(self):
        """R^T @ R should be identity."""
        R = random_rotation_matrix()
        eye = R.T @ R
        torch.testing.assert_close(eye, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_proper_rotation(self):
        """det(R) should be +1, not -1."""
        for _ in range(50):
            R = random_rotation_matrix()
            det = torch.det(R)
            torch.testing.assert_close(det, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_different_each_call(self):
        """Two calls should produce different rotations."""
        R1 = random_rotation_matrix()
        R2 = random_rotation_matrix()
        assert not torch.allclose(R1, R2)

    def test_uniform_distribution(self):
        """Columns of R should have zero mean over many samples (uniformity)."""
        rotations = torch.stack([random_rotation_matrix() for _ in range(5000)])
        # Mean of all rotations should be close to zero matrix
        mean_R = rotations.mean(dim=0)
        assert mean_R.abs().max() < 0.1, f"Mean rotation not near zero: {mean_R}"

    def test_device(self):
        R = random_rotation_matrix(device=torch.device("cpu"))
        assert R.device == torch.device("cpu")


class TestRandomSE3Augmentation:
    """Tests for random_se3_augmentation."""

    @pytest.fixture
    def sample_protein(self):
        """A small protein with 10 residues, 5 atoms each (rest masked)."""
        torch.manual_seed(42)
        N = 10
        coords = torch.randn(N, 14, 3) * 10 + 50  # offset from origin
        mask = torch.zeros(N, 14, dtype=torch.bool)
        mask[:, :5] = True  # first 5 atoms are real
        return coords, mask

    def test_output_shape(self, sample_protein):
        coords, mask = sample_protein
        out = random_se3_augmentation(coords, mask)
        assert out.shape == coords.shape

    def test_input_not_modified(self, sample_protein):
        coords, mask = sample_protein
        coords_copy = coords.clone()
        random_se3_augmentation(coords, mask)
        torch.testing.assert_close(coords, coords_copy)

    def test_centered(self, sample_protein):
        """After augmentation, CA centroid should be near origin."""
        coords, mask = sample_protein
        out = random_se3_augmentation(coords, mask)
        ca_centroid = out[:, 1, :].mean(dim=0)  # CA = index 1
        assert ca_centroid.abs().max() < 1e-4, f"CA centroid not at origin: {ca_centroid}"

    def test_distances_preserved(self, sample_protein):
        """SE(3) + normalization: pairwise distances scale by 1/COORD_SCALE."""
        from enzflow.data.transforms import COORD_SCALE
        coords, mask = sample_protein
        ca_before = coords[:, 1, :]
        dist_before = torch.cdist(ca_before.unsqueeze(0), ca_before.unsqueeze(0)).squeeze(0)

        out = random_se3_augmentation(coords, mask)
        ca_after = out[:, 1, :]
        dist_after = torch.cdist(ca_after.unsqueeze(0), ca_after.unsqueeze(0)).squeeze(0)

        torch.testing.assert_close(dist_before / COORD_SCALE, dist_after, atol=1e-4, rtol=1e-4)

    def test_bond_lengths_preserved(self, sample_protein):
        """Intra-residue distances scale by 1/COORD_SCALE after augmentation."""
        from enzflow.data.transforms import COORD_SCALE
        coords, mask = sample_protein
        # Distance between atom 0 (N) and atom 1 (CA) for each residue
        bond_before = (coords[:, 0, :] - coords[:, 1, :]).norm(dim=-1)

        out = random_se3_augmentation(coords, mask)
        bond_after = (out[:, 0, :] - out[:, 1, :]).norm(dim=-1)

        torch.testing.assert_close(bond_before / COORD_SCALE, bond_after, atol=1e-4, rtol=1e-4)

    def test_different_each_call(self, sample_protein):
        """Two augmentations of the same protein should differ."""
        coords, mask = sample_protein
        out1 = random_se3_augmentation(coords, mask)
        out2 = random_se3_augmentation(coords, mask)
        assert not torch.allclose(out1, out2)

    def test_without_mask(self, sample_protein):
        """Should work without mask (all CAs assumed present)."""
        coords, _ = sample_protein
        out = random_se3_augmentation(coords)
        assert out.shape == coords.shape
        ca_centroid = out[:, 1, :].mean(dim=0)
        assert ca_centroid.abs().max() < 1e-4

    def test_with_missing_ca(self):
        """Handle edge case: some CA atoms are masked out."""
        torch.manual_seed(0)
        N = 10
        coords = torch.randn(N, 14, 3) * 10 + 50
        mask = torch.zeros(N, 14, dtype=torch.bool)
        mask[:, :4] = True
        # Mask out CA for residues 8 and 9
        mask[8, 1] = False
        mask[9, 1] = False

        out = random_se3_augmentation(coords, mask)
        assert out.shape == coords.shape

        # Centroid should be computed from the 8 valid CAs
        ca_valid = out[:8, 1, :]
        centroid = ca_valid.mean(dim=0)
        assert centroid.abs().max() < 0.5  # approximate, due to excluded residues
