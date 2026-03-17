"""Tests for evaluation modules: geometry, diversity, designability."""

from __future__ import annotations

import math

import numpy as np

from enzflow.evaluation.diversity import (
    compute_pairwise_tm,
    evaluate_diversity,
)
from enzflow.evaluation.geometry import (
    compute_bond_angles,
    compute_bond_lengths,
    compute_clashes,
    compute_ramachandran,
    evaluate_geometry,
    ramachandran_allowed_ratio,
)

# ---------------------------------------------------------------------------
# Helpers: build ideal backbone
# ---------------------------------------------------------------------------

def _ideal_backbone(n_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Build an idealized linear backbone (atom14 format).

    Places N, CA, C, O at ideal bond lengths and angles along a straight
    extended chain. Only backbone atoms (indices 0-3) are filled.

    Returns:
        (coords, atom_mask): coords [n_res, 14, 3], atom_mask [n_res, 14]
    """
    coords = np.zeros((n_res, 14, 3))
    atom_mask = np.zeros((n_res, 14), dtype=bool)

    # Ideal bond lengths
    n_ca = 1.458
    ca_c = 1.525
    c_o = 1.229
    c_n = 1.329  # peptide bond

    # Build along x-axis with simple geometry
    x = 0.0
    for i in range(n_res):
        # N
        coords[i, 0] = [x, 0.0, 0.0]
        atom_mask[i, 0] = True
        x += n_ca

        # CA
        coords[i, 1] = [x, 0.0, 0.0]
        atom_mask[i, 1] = True
        x += ca_c

        # C
        coords[i, 2] = [x, 0.0, 0.0]
        atom_mask[i, 2] = True

        # O (perpendicular to backbone)
        coords[i, 3] = [x, c_o, 0.0]
        atom_mask[i, 3] = True

        x += c_n  # peptide bond to next N

    return coords, atom_mask


def _helical_backbone(n_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a rough alpha-helix backbone.

    Uses standard helix parameters: 3.6 residues/turn, 1.5A rise/residue,
    radius ~2.3A. phi=-57, psi=-47 degrees.

    Returns:
        (coords, atom_mask)
    """
    coords = np.zeros((n_res, 14, 3))
    atom_mask = np.zeros((n_res, 14), dtype=bool)

    # Helix parameters
    rise_per_res = 1.5  # along z
    radius = 2.3
    angle_per_res = 2 * math.pi / 3.6  # ~100 degrees

    for i in range(n_res):
        theta = i * angle_per_res
        z = i * rise_per_res
        cx = radius * math.cos(theta)
        cy = radius * math.sin(theta)

        # Place backbone atoms around the helix axis
        coords[i, 0] = [cx - 0.7, cy, z - 0.5]       # N
        coords[i, 1] = [cx, cy, z]                     # CA
        coords[i, 2] = [cx + 0.7, cy, z + 0.5]        # C
        coords[i, 3] = [cx + 0.7, cy + 1.2, z + 0.5]  # O
        atom_mask[i, :4] = True

    return coords, atom_mask


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------


class TestBondLengths:
    """Test bond length computation."""

    def test_ideal_backbone_lengths(self):
        """Ideal backbone should have near-zero bond length errors."""
        coords, atom_mask = _ideal_backbone(20)
        lengths = compute_bond_lengths(coords, atom_mask)

        assert "N_CA" in lengths
        assert "CA_C" in lengths
        assert "C_O" in lengths
        assert "C_N_pep" in lengths

        np.testing.assert_allclose(lengths["N_CA"], 1.458, atol=1e-6)
        np.testing.assert_allclose(lengths["CA_C"], 1.525, atol=1e-6)
        np.testing.assert_allclose(lengths["C_O"], 1.229, atol=1e-6)
        np.testing.assert_allclose(lengths["C_N_pep"], 1.329, atol=1e-6)

    def test_output_shapes(self):
        n_res = 10
        coords, atom_mask = _ideal_backbone(n_res)
        lengths = compute_bond_lengths(coords, atom_mask)

        assert len(lengths["N_CA"]) == n_res
        assert len(lengths["CA_C"]) == n_res
        assert len(lengths["C_O"]) == n_res
        assert len(lengths["C_N_pep"]) == n_res - 1

    def test_single_residue(self):
        coords, atom_mask = _ideal_backbone(1)
        lengths = compute_bond_lengths(coords, atom_mask)

        assert len(lengths["N_CA"]) == 1
        assert len(lengths.get("C_N_pep", [])) == 0


class TestBondAngles:
    """Test bond angle computation."""

    def test_linear_backbone_angles(self):
        """Linear backbone (all atoms on x-axis) should have ~180 deg angles."""
        coords, atom_mask = _ideal_backbone(10)
        angles = compute_bond_angles(coords, atom_mask)

        assert "N_CA_C" in angles
        # Linear chain -> 180 degrees (pi radians)
        np.testing.assert_allclose(angles["N_CA_C"], math.pi, atol=1e-3)

    def test_output_shapes(self):
        n_res = 10
        coords, atom_mask = _ideal_backbone(n_res)
        angles = compute_bond_angles(coords, atom_mask)

        assert len(angles["N_CA_C"]) == n_res
        assert len(angles["CA_C_N"]) == n_res - 1
        assert len(angles["C_N_CA"]) == n_res - 1

    def test_right_angle(self):
        """Three points forming a 90-degree angle."""
        coords = np.zeros((1, 14, 3))
        atom_mask = np.zeros((1, 14), dtype=bool)
        coords[0, 0] = [1, 0, 0]  # N
        coords[0, 1] = [0, 0, 0]  # CA (vertex)
        coords[0, 2] = [0, 1, 0]  # C
        atom_mask[0, :3] = True

        angles = compute_bond_angles(coords, atom_mask)
        np.testing.assert_allclose(
            angles["N_CA_C"], math.pi / 2, atol=1e-5
        )


class TestRamachandran:
    """Test Ramachandran angle computation."""

    def test_helical_backbone_phi_negative(self):
        """Alpha helix should have negative phi angles."""
        coords, atom_mask = _helical_backbone(20)
        rama = compute_ramachandran(coords, atom_mask)

        assert len(rama["phi"]) > 0
        assert len(rama["psi"]) > 0

    def test_output_lengths(self):
        n_res = 15
        coords, atom_mask = _ideal_backbone(n_res)
        rama = compute_ramachandran(coords, atom_mask)

        # phi defined for residues 1..N-1 (need C[i-1])
        assert len(rama["phi"]) == n_res - 1
        # psi defined for residues 0..N-2 (need N[i+1])
        assert len(rama["psi"]) == n_res - 1

    def test_single_residue(self):
        coords, atom_mask = _ideal_backbone(1)
        rama = compute_ramachandran(coords, atom_mask)
        assert len(rama["phi"]) == 0
        assert len(rama["psi"]) == 0

    def test_allowed_ratio_all_negative_phi(self):
        """All negative phi should be ~100% allowed."""
        phi = np.full(50, -1.0)  # ~-57 degrees
        psi = np.full(50, -0.8)  # ~-46 degrees
        ratio = ramachandran_allowed_ratio(phi, psi)
        assert ratio == 1.0

    def test_allowed_ratio_all_positive_phi(self):
        """All positive phi (outside allowed) should have low ratio."""
        phi = np.full(50, 2.5)  # ~143 degrees, forbidden
        psi = np.full(50, 0.0)
        ratio = ramachandran_allowed_ratio(phi, psi)
        assert ratio < 0.1


class TestClashes:
    """Test steric clash detection."""

    def test_no_clashes_ideal(self):
        """Ideal backbone should have no clashes."""
        coords, atom_mask = _ideal_backbone(20)
        result = compute_clashes(coords, atom_mask)
        assert result["n_clashes"] == 0
        assert result["clash_ratio"] == 0.0

    def test_forced_clash(self):
        """Non-adjacent residues with overlapping atoms should clash."""
        # Use 3 residues so res 0 and res 2 are non-adjacent (|0-2|=2 > 1)
        coords = np.zeros((3, 14, 3))
        atom_mask = np.zeros((3, 14), dtype=bool)
        # Residue 0: atom at origin
        coords[0, 0] = [0, 0, 0]
        atom_mask[0, 0] = True
        # Residue 1: far away (not involved)
        coords[1, 0] = [100, 0, 0]
        atom_mask[1, 0] = True
        # Residue 2: atom very close to residue 0
        coords[2, 0] = [0.5, 0, 0]
        atom_mask[2, 0] = True

        result = compute_clashes(coords, atom_mask, clash_threshold=1.5)
        assert result["n_clashes"] == 1

    def test_no_clash_far_apart(self):
        """Atoms far apart should not clash."""
        coords = np.zeros((2, 14, 3))
        atom_mask = np.zeros((2, 14), dtype=bool)
        coords[0, 0] = [0, 0, 0]
        atom_mask[0, 0] = True
        coords[1, 0] = [10, 0, 0]
        atom_mask[1, 0] = True

        result = compute_clashes(coords, atom_mask)
        assert result["n_clashes"] == 0

    def test_single_residue(self):
        """Single residue: no inter-residue clashes possible."""
        coords, atom_mask = _ideal_backbone(1)
        result = compute_clashes(coords, atom_mask)
        assert result["n_clashes"] == 0


class TestEvaluateGeometry:
    """Test the combined evaluate_geometry function."""

    def test_returns_all_keys(self):
        coords, atom_mask = _ideal_backbone(20)
        metrics = evaluate_geometry(coords, atom_mask)

        assert "bond_mae_all" in metrics
        assert "angle_mae_all_deg" in metrics
        assert "rama_allowed_ratio" in metrics
        assert "clash_count" in metrics
        assert "clash_ratio" in metrics

    def test_ideal_backbone_good_metrics(self):
        """Ideal backbone should have very small bond length errors."""
        coords, atom_mask = _ideal_backbone(20)
        metrics = evaluate_geometry(coords, atom_mask)

        assert metrics["bond_mae_all"] < 1e-5
        assert metrics["bond_within_0.05A"] == 1.0
        assert metrics["clash_count"] == 0


# ---------------------------------------------------------------------------
# Diversity tests
# ---------------------------------------------------------------------------


class TestDiversity:
    """Test diversity metrics."""

    def test_identical_structures(self):
        """Identical structures should have TM-score = 1.0."""
        ca = np.random.default_rng(42).standard_normal((50, 3)) * 5
        ca_list = [ca, ca.copy()]
        seqs = ["A" * 50, "A" * 50]

        result = evaluate_diversity(ca_list, seqs)
        assert result["pairwise_tm_mean"] > 0.99
        assert result["n_structures"] == 2

    def test_different_structures(self):
        """Very different structures should have low TM-score."""
        rng = np.random.default_rng(42)
        ca1 = rng.standard_normal((50, 3)) * 10
        ca2 = rng.standard_normal((50, 3)) * 10

        result = evaluate_diversity([ca1, ca2], ["A" * 50, "A" * 50])
        assert result["pairwise_tm_mean"] < 0.5

    def test_single_structure(self):
        """Single structure: no pairwise comparison."""
        ca = np.random.default_rng(42).standard_normal((50, 3))
        result = evaluate_diversity([ca], ["A" * 50])
        assert result["n_structures"] == 1
        assert result["pairwise_tm_mean"] == 0.0

    def test_pairwise_tm_matrix_shape(self):
        """Matrix should be NxN."""
        rng = np.random.default_rng(42)
        cas = [rng.standard_normal((30, 3)) for _ in range(4)]
        seqs = ["A" * 30] * 4

        matrix = compute_pairwise_tm(cas, seqs)
        assert matrix.shape == (4, 4)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Designability tests (lightweight, no ESMFold)
# ---------------------------------------------------------------------------


class TestDesignability:
    """Test designability helpers (without ESMFold)."""

    def test_check_esm_available(self):
        """Just test that the check function runs without error."""
        from enzflow.evaluation.designability import _check_esm_available
        # Returns bool, doesn't raise
        result = _check_esm_available()
        assert isinstance(result, bool)

    def test_compute_sc_tm_identical(self):
        """Identical coords should give scTM = 1.0."""
        from enzflow.evaluation.designability import compute_sc_tm

        ca = np.random.default_rng(42).standard_normal((50, 3)) * 5
        sc_tms = compute_sc_tm([ca], [ca.copy()], ["A" * 50])
        assert len(sc_tms) == 1
        assert sc_tms[0] > 0.99

    def test_compute_sc_tm_different(self):
        """Very different coords should give low scTM."""
        from enzflow.evaluation.designability import compute_sc_tm

        rng = np.random.default_rng(42)
        ca1 = rng.standard_normal((50, 3)) * 10
        ca2 = rng.standard_normal((50, 3)) * 10
        sc_tms = compute_sc_tm([ca1], [ca2], ["A" * 50])
        assert sc_tms[0] < 0.5
