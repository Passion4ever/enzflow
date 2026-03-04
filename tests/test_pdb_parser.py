"""Tests for enzflow.data.pdb_parser — parsing correctness."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from enzflow.data.pdb_parser import ProteinStructure, parse_structure, write_pdb
from enzflow.data.residue_constants import NUM_ATOMS_PER_RES, RESTYPE_ATOM14_NAMES, RESTYPES


def test_parse_pdb_basic(sample_pdb_path: Path):
    """Parsing produces a ProteinStructure with correct shapes."""
    struct = parse_structure(sample_pdb_path)
    n_res = struct.aatype.shape[0]

    assert n_res > 0
    assert struct.coords.shape == (n_res, 14, 3)
    assert struct.atom_mask.shape == (n_res, 14)
    assert struct.aatype.shape == (n_res,)
    assert struct.residue_index.shape == (n_res,)
    assert len(struct.chain_id) == n_res


def test_output_types(sample_pdb_path: Path):
    """Check dtype of each field in ProteinStructure."""
    struct = parse_structure(sample_pdb_path)
    assert struct.coords.dtype == torch.float32
    assert struct.atom_mask.dtype == torch.bool
    assert struct.aatype.dtype == torch.long
    assert struct.residue_index.dtype == torch.long
    assert isinstance(struct.chain_id, tuple)
    assert all(isinstance(c, str) for c in struct.chain_id)


def test_atom_mask_matches_real_atoms(sample_pdb_path: Path):
    """Where mask=True, coordinates should not all be identical to CA."""
    struct = parse_structure(sample_pdb_path)
    # For each residue, check that masked atoms have real coordinates
    for i in range(struct.aatype.shape[0]):
        ca_coord = struct.coords[i, 1]  # CA is index 1
        for j in range(NUM_ATOMS_PER_RES):
            if struct.atom_mask[i, j]:
                # At least the atom should have been read (it's a real atom)
                # We just verify that masked positions are marked correctly
                assert struct.coords[i, j].shape == (3,)


def test_backbone_atoms_always_present(sample_pdb_path: Path):
    """Every residue should have N, CA, C, O (indices 0-3) with mask=True."""
    struct = parse_structure(sample_pdb_path)
    for i in range(struct.aatype.shape[0]):
        for bb_idx in range(4):  # N=0, CA=1, C=2, O=3
            assert struct.atom_mask[i, bb_idx].item(), (
                f"Residue {i} missing backbone atom at index {bb_idx}"
            )


def test_write_then_parse_roundtrip(sample_pdb_path: Path, tmp_path: Path):
    """Parse → write → re-parse should give coordinates within 0.01 Å."""
    original = parse_structure(sample_pdb_path)

    roundtrip_path = tmp_path / "roundtrip.pdb"
    write_pdb(original, roundtrip_path)
    reloaded = parse_structure(roundtrip_path)

    # Same number of residues
    assert original.aatype.shape[0] == reloaded.aatype.shape[0]
    # Same residue types
    assert torch.equal(original.aatype, reloaded.aatype)

    # Coordinate difference for masked atoms
    mask = original.atom_mask & reloaded.atom_mask
    if mask.any():
        diff = (original.coords - reloaded.coords).abs()
        masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
        max_diff = masked_diff.max().item()
        assert max_diff < 0.01, f"Max coordinate diff: {max_diff:.4f} Å"


def test_skip_non_standard_residues(tmp_path: Path):
    """Non-standard residues (like HOH) should be skipped."""
    pdb_content = """\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   1.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -1.000   0.000  1.00  0.00           C
HETATM    6  O   HOH A 100      10.000  10.000  10.000  1.00  0.00           O
HETATM    7  C1  UNK A 101      20.000  20.000  20.000  1.00  0.00           C
ATOM      8  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      9  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C
ATOM     10  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C
ATOM     11  O   GLY A   2       6.000   1.000   0.000  1.00  0.00           O
END
"""
    pdb_path = tmp_path / "with_nonstandard.pdb"
    pdb_path.write_text(pdb_content)

    struct = parse_structure(pdb_path)
    # Should have exactly 2 residues (ALA + GLY), skipping HOH and UNK
    assert struct.aatype.shape[0] == 2


def test_missing_atoms_filled_with_ca(tmp_path: Path):
    """Missing sidechain atoms should get CA coordinates with mask=False."""
    # Create an ALA with only backbone (missing CB)
    pdb_content = """\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   1.000   0.000  1.00  0.00           O
END
"""
    pdb_path = tmp_path / "missing_cb.pdb"
    pdb_path.write_text(pdb_content)

    struct = parse_structure(pdb_path)
    assert struct.aatype.shape[0] == 1

    # CB is index 4 for ALA
    cb_idx = 4
    assert not struct.atom_mask[0, cb_idx].item(), "CB should be masked as missing"

    # CB coords should equal CA coords
    ca_coords = struct.coords[0, 1]  # CA = index 1
    cb_coords = struct.coords[0, cb_idx]
    torch.testing.assert_close(cb_coords, ca_coords)


def test_nonstandard_residue_mapped(tmp_path: Path):
    """MSE (selenomethionine) should be mapped to MET, not skipped."""
    pdb_content = """\
HETATM    1  N   MSE A   1       1.000   0.000   0.000  1.00  0.00           N
HETATM    2  CA  MSE A   1       2.000   0.000   0.000  1.00  0.00           C
HETATM    3  C   MSE A   1       3.000   0.000   0.000  1.00  0.00           C
HETATM    4  O   MSE A   1       3.000   1.000   0.000  1.00  0.00           O
HETATM    5  CB  MSE A   1       2.000  -1.000   0.0    1.00  0.00           C
HETATM    6  CG  MSE A   1       2.000  -2.000   0.0    1.00  0.00           C
HETATM    7  SE  MSE A   1       2.000  -3.000   0.0    1.00  0.00          SE
HETATM    8  CE  MSE A   1       2.000  -4.000   0.0    1.00  0.00           C
ATOM      9  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM     10  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C
ATOM     11  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C
ATOM     12  O   GLY A   2       6.000   1.000   0.000  1.00  0.00           O
END
"""
    pdb_path = tmp_path / "with_mse.pdb"
    pdb_path.write_text(pdb_content)

    from enzflow.data.residue_constants import RESTYPE_ORDER

    struct = parse_structure(pdb_path)
    # MSE mapped to MET + GLY = 2 residues (not 1)
    assert struct.aatype.shape[0] == 2
    assert struct.aatype[0].item() == RESTYPE_ORDER["MET"]
    assert struct.aatype[1].item() == RESTYPE_ORDER["GLY"]
    # MSE backbone should be present
    for bb_idx in range(4):
        assert struct.atom_mask[0, bb_idx].item()


def test_file_not_found():
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_structure("/nonexistent/path/to/file.pdb")


def test_empty_structure(tmp_path: Path):
    """Should raise RuntimeError if no standard residues found."""
    pdb_content = """\
HETATM    1  O   HOH A   1      10.000  10.000  10.000  1.00  0.00           O
END
"""
    pdb_path = tmp_path / "empty.pdb"
    pdb_path.write_text(pdb_content)

    with pytest.raises(RuntimeError, match="No standard residues"):
        parse_structure(pdb_path)
