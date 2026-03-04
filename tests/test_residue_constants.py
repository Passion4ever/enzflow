"""Tests for enzflow.data.residue_constants — constant table integrity."""

from __future__ import annotations

import torch

from enzflow.data.residue_constants import (
    ATOM37_NAMES,
    ATOM37_ORDER,
    ELEMENTS,
    ELEMENT_ORDER,
    NUM_ATOMS_PER_RES,
    NUM_RES_TYPES,
    RESTYPES,
    RESTYPE_ATOM14_NAMES,
    RESTYPE_ORDER,
    get_atom14_element_indices,
    get_atom14_mask,
    get_atom14_to_atom37,
    get_atom37_to_atom14,
    validate_constants,
)


def test_all_20_amino_acids_defined():
    """All 20 standard amino acids are present."""
    assert len(RESTYPES) == 20
    assert len(RESTYPE_ATOM14_NAMES) == 20
    assert len(RESTYPE_ORDER) == 20
    # Check all 20 are unique
    assert len(set(RESTYPES)) == 20


def test_backbone_atoms_consistent():
    """Every residue has N, CA, C, O as the first 4 atom14 slots."""
    for resname in RESTYPES:
        atoms = RESTYPE_ATOM14_NAMES[resname]
        assert atoms[0] == "N", f"{resname}[0] = {atoms[0]}"
        assert atoms[1] == "CA", f"{resname}[1] = {atoms[1]}"
        assert atoms[2] == "C", f"{resname}[2] = {atoms[2]}"
        assert atoms[3] == "O", f"{resname}[3] = {atoms[3]}"


def test_atom14_mask_shape_and_values():
    """ATOM14_MASK has shape [20, 14] with values in {0, 1}."""
    mask = get_atom14_mask()
    assert mask.shape == (NUM_RES_TYPES, NUM_ATOMS_PER_RES)
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_gly_has_4_atoms():
    """GLY should have exactly 4 atoms (N, CA, C, O)."""
    gly_idx = RESTYPE_ORDER["GLY"]
    mask = get_atom14_mask()
    assert mask[gly_idx].sum().item() == 4


def test_trp_has_14_atoms():
    """TRP is the only residue that fills all 14 atom14 slots."""
    trp_idx = RESTYPE_ORDER["TRP"]
    mask = get_atom14_mask()
    assert mask[trp_idx].sum().item() == 14
    # No other residue should have 14
    for i in range(NUM_RES_TYPES):
        if i != trp_idx:
            assert mask[i].sum().item() < 14, f"{RESTYPES[i]} also has 14 atoms?"


def test_atom14_to_atom37_invertible():
    """atom14 → atom37 → atom14 round-trip is identity for real atoms."""
    a14_to_a37 = get_atom14_to_atom37()
    a37_to_a14 = get_atom37_to_atom14()
    mask = get_atom14_mask()

    for i in range(NUM_RES_TYPES):
        for j in range(NUM_ATOMS_PER_RES):
            if mask[i, j] > 0:
                a37_idx = a14_to_a37[i, j].item()
                roundtrip = a37_to_a14[i, a37_idx].item()
                assert roundtrip == j, (
                    f"{RESTYPES[i]} atom14[{j}] → atom37[{a37_idx}] → atom14[{roundtrip}]"
                )


def test_met_sidechain_correct():
    """MET sidechain should be CB, CG, SD, CE (not LYS's CD, CE, NZ)."""
    met_atoms = RESTYPE_ATOM14_NAMES["MET"]
    real_atoms = [a for a in met_atoms if a != ""]
    assert "SD" in real_atoms, "MET must have SD (sulfur)"
    assert "CE" in real_atoms, "MET must have CE"
    assert "NZ" not in real_atoms, "MET must NOT have NZ (that's LYS)"
    assert "CD" not in real_atoms, "MET must NOT have CD"


def test_element_mapping_consistent():
    """Element type matches the first letter of each atom name."""
    elem_indices = get_atom14_element_indices()
    mask = get_atom14_mask()

    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "" and mask[i, j] > 0:
                first_alpha = ""
                for ch in aname:
                    if ch.isalpha():
                        first_alpha = ch
                        break
                expected_idx = ELEMENT_ORDER[first_alpha]
                actual_idx = elem_indices[i, j].item()
                assert actual_idx == expected_idx, (
                    f"{resname}.{aname}: expected element {ELEMENTS[expected_idx]}, "
                    f"got {ELEMENTS[actual_idx]}"
                )


def test_no_duplicate_atom_names():
    """No duplicate atom names within a single residue."""
    for resname in RESTYPES:
        atoms = [a for a in RESTYPE_ATOM14_NAMES[resname] if a != ""]
        assert len(atoms) == len(set(atoms)), (
            f"{resname} has duplicate atoms: {atoms}"
        )


def test_validate_constants_passes():
    """The built-in validate_constants() should pass without errors."""
    validate_constants()


def test_atom37_has_37_entries():
    """ATOM37_NAMES should have exactly 37 unique entries."""
    assert len(ATOM37_NAMES) == 37
    assert len(ATOM37_ORDER) == 37
    assert len(set(ATOM37_NAMES)) == 37


def test_all_atom14_names_in_atom37():
    """Every non-empty atom14 name must exist in ATOM37_NAMES."""
    for resname in RESTYPES:
        for aname in RESTYPE_ATOM14_NAMES[resname]:
            if aname != "":
                assert aname in ATOM37_ORDER, (
                    f"{resname}.{aname} not found in ATOM37_NAMES"
                )
