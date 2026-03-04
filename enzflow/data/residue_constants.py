"""Residue constants for atom14/atom37 representations.

This module defines all amino acid atom mappings that form the data foundation
for the entire enzflow project. All mappings must be 100% correct — downstream
modules depend on them.

Reference: OpenFold / AlphaFold2 residue_constants, Engh & Huber (2001).
"""

from __future__ import annotations

import functools
import math
from typing import Final

import torch

# ---------------------------------------------------------------------------
# 1. Basic constants
# ---------------------------------------------------------------------------

NUM_RES_TYPES: Final[int] = 20
NUM_RES_TYPES_WITH_MASK: Final[int] = 21
NUM_ATOMS_PER_RES: Final[int] = 14
NUM_ATOM37_TYPES: Final[int] = 37

# 20 standard amino acids — three-letter codes, alphabetical order by 1-letter
RESTYPES: Final[tuple[str, ...]] = (
    "ALA",  # A  0
    "ARG",  # R  1
    "ASN",  # N  2
    "ASP",  # D  3
    "CYS",  # C  4
    "GLN",  # Q  5
    "GLU",  # E  6
    "GLY",  # G  7
    "HIS",  # H  8
    "ILE",  # I  9
    "LEU",  # L 10
    "LYS",  # K 11
    "MET",  # M 12
    "PHE",  # F 13
    "PRO",  # P 14
    "SER",  # S 15
    "THR",  # T 16
    "TRP",  # W 17
    "TYR",  # Y 18
    "VAL",  # V 19
)

RESTYPE_ORDER: Final[dict[str, int]] = {name: i for i, name in enumerate(RESTYPES)}

RESTYPE_1TO3: Final[dict[str, str]] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

RESTYPE_3TO1: Final[dict[str, str]] = {v: k for k, v in RESTYPE_1TO3.items()}

# Non-standard residue → standard residue mapping
# These modified residues are chemically similar to their parent and should
# be treated as the standard form rather than silently discarded.
NONSTANDARD_RESTYPE_MAP: Final[dict[str, str]] = {
    "MSE": "MET",  # selenomethionine → methionine
    "HYP": "PRO",  # hydroxyproline → proline
    "TPO": "THR",  # phosphothreonine → threonine
    "SEP": "SER",  # phosphoserine → serine
    "PTR": "TYR",  # phosphotyrosine → tyrosine
    "CSO": "CYS",  # S-hydroxycysteine → cysteine
    "CSD": "CYS",  # S-cysteinesulfinic acid → cysteine
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiocysteine → cysteine
    "MLY": "LYS",  # N-dimethyl-lysine → lysine
    "M3L": "LYS",  # N-trimethyl-lysine → lysine
    "ALY": "LYS",  # N-acetyl-lysine → lysine
}

# ---------------------------------------------------------------------------
# 2. atom14 atom names per residue type
#    Indices 0-3 are always N, CA, C, O (backbone).
#    Empty string "" means the slot is unused.
# ---------------------------------------------------------------------------

RESTYPE_ATOM14_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""),
    "GLY": ("N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""),
    "TRP": (
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
        "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
    ),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""),
}

# ---------------------------------------------------------------------------
# 3. atom37 definitions
#    The canonical 37 heavy-atom types used in AlphaFold / OpenFold.
# ---------------------------------------------------------------------------

ATOM37_NAMES: Final[tuple[str, ...]] = (
    "N",    #  0
    "CA",   #  1
    "C",    #  2
    "CB",   #  3
    "O",    #  4
    "CG",   #  5
    "CG1",  #  6
    "CG2",  #  7
    "OG",   #  8
    "OG1",  #  9
    "SG",   # 10
    "CD",   # 11
    "CD1",  # 12
    "CD2",  # 13
    "ND1",  # 14
    "ND2",  # 15
    "OD1",  # 16
    "OD2",  # 17
    "SD",   # 18
    "CE",   # 19
    "CE1",  # 20
    "CE2",  # 21
    "CE3",  # 22
    "NE",   # 23
    "NE1",  # 24
    "NE2",  # 25
    "OE1",  # 26
    "OE2",  # 27
    "CH2",  # 28
    "NH1",  # 29
    "NH2",  # 30
    "OH",   # 31
    "CZ",   # 32
    "CZ2",  # 33
    "CZ3",  # 34
    "NZ",   # 35
    "OXT",  # 36
)

ATOM37_ORDER: Final[dict[str, int]] = {name: i for i, name in enumerate(ATOM37_NAMES)}

# ---------------------------------------------------------------------------
# 4. Element constants
# ---------------------------------------------------------------------------

ELEMENTS: Final[tuple[str, ...]] = ("C", "N", "O", "S")
ELEMENT_ORDER: Final[dict[str, int]] = {e: i for i, e in enumerate(ELEMENTS)}

# ---------------------------------------------------------------------------
# 5. Precomputed mapping tensors
# ---------------------------------------------------------------------------


def _atom_name_to_element(atom_name: str) -> str:
    """Extract element from PDB atom name (first non-digit character)."""
    for ch in atom_name:
        if ch.isalpha():
            return ch
    raise ValueError(f"Cannot determine element for atom name: {atom_name!r}")


@functools.lru_cache(maxsize=1)
def _build_atom14_mask() -> torch.Tensor:
    """ATOM14_MASK: Tensor[20, 14] — which atom14 slots have real atoms."""
    mask = torch.zeros(NUM_RES_TYPES, NUM_ATOMS_PER_RES, dtype=torch.float32)
    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "":
                mask[i, j] = 1.0
    return mask


@functools.lru_cache(maxsize=1)
def _build_atom14_to_atom37() -> torch.Tensor:
    """ATOM14_TO_ATOM37: Tensor[20, 14] — atom14 index → atom37 index.

    Unused slots map to 0 (but should be masked out).
    """
    mapping = torch.zeros(NUM_RES_TYPES, NUM_ATOMS_PER_RES, dtype=torch.long)
    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "":
                mapping[i, j] = ATOM37_ORDER[aname]
    return mapping


@functools.lru_cache(maxsize=1)
def _build_atom37_to_atom14() -> torch.Tensor:
    """ATOM37_TO_ATOM14: Tensor[20, 37] — atom37 index → atom14 index.

    Slots that don't exist in a given residue map to 0 (mask out).
    """
    mapping = torch.zeros(NUM_RES_TYPES, NUM_ATOM37_TYPES, dtype=torch.long)
    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "":
                a37_idx = ATOM37_ORDER[aname]
                mapping[i, a37_idx] = j
    return mapping


@functools.lru_cache(maxsize=1)
def _build_atom14_element_indices() -> torch.Tensor:
    """ATOM14_ELEMENT_INDICES: Tensor[20, 14] — element type per atom slot.

    C=0, N=1, O=2, S=3. Unused slots get 0.
    """
    indices = torch.zeros(NUM_RES_TYPES, NUM_ATOMS_PER_RES, dtype=torch.long)
    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "":
                elem = _atom_name_to_element(aname)
                indices[i, j] = ELEMENT_ORDER[elem]
    return indices


# Public accessors (read-only tensors via lru_cache)
def get_atom14_mask() -> torch.Tensor:
    """Return ATOM14_MASK: Tensor[20, 14]."""
    return _build_atom14_mask()


def get_atom14_to_atom37() -> torch.Tensor:
    """Return ATOM14_TO_ATOM37: Tensor[20, 14]."""
    return _build_atom14_to_atom37()


def get_atom37_to_atom14() -> torch.Tensor:
    """Return ATOM37_TO_ATOM14: Tensor[20, 37]."""
    return _build_atom37_to_atom14()


def get_atom14_element_indices() -> torch.Tensor:
    """Return ATOM14_ELEMENT_INDICES: Tensor[20, 14]."""
    return _build_atom14_element_indices()


# ---------------------------------------------------------------------------
# 6. Ideal bond lengths and angles (Engh & Huber, 2001)
# ---------------------------------------------------------------------------

# Bond lengths in Ångströms — (atom1, atom2): length
IDEAL_BOND_LENGTHS: Final[dict[tuple[str, str], float]] = {
    # Backbone
    ("N", "CA"): 1.458,
    ("CA", "C"): 1.525,
    ("C", "O"): 1.229,
    ("C", "N"): 1.329,  # peptide bond
    # CA-CB
    ("CA", "CB"): 1.530,
    # Common sidechain
    ("CB", "CG"): 1.530,
    ("CB", "CG1"): 1.530,
    ("CB", "CG2"): 1.521,
    ("CB", "OG"): 1.431,
    ("CB", "OG1"): 1.433,
    ("CB", "SG"): 1.808,
    ("CG", "CD"): 1.530,
    ("CG", "OD1"): 1.249,
    ("CG", "OD2"): 1.249,
    ("CG", "ND1"): 1.371,
    ("CG", "ND2"): 1.370,
    ("CG", "SD"): 1.810,
    ("CD", "NE"): 1.461,
    ("CD", "OE1"): 1.249,
    ("CD", "OE2"): 1.249,
    ("CD", "CE"): 1.530,
    ("CD", "NE2"): 1.327,
    ("SD", "CE"): 1.791,
    ("CE", "NZ"): 1.486,
    ("NE", "CZ"): 1.329,
    ("CZ", "NH1"): 1.326,
    ("CZ", "NH2"): 1.326,
    ("CZ", "OH"): 1.376,
    # Aromatic
    ("CG", "CD1"): 1.384,
    ("CG", "CD2"): 1.384,
    ("CD1", "CE1"): 1.384,
    ("CD2", "CE2"): 1.384,
    ("CE1", "CZ"): 1.384,
    ("CE2", "CZ"): 1.384,
    ("CD1", "NE1"): 1.374,
    ("CE2", "NE2"): 1.370,
    ("CE2", "CZ2"): 1.394,
    ("CD2", "CE3"): 1.398,
    ("CZ2", "CH2"): 1.368,
    ("CE3", "CZ3"): 1.382,
    ("CZ3", "CH2"): 1.400,
}

# Bond angles in radians — (atom1, atom2, atom3): angle
IDEAL_BOND_ANGLES: Final[dict[tuple[str, str, str], float]] = {
    # Backbone
    ("N", "CA", "C"): math.radians(111.2),
    ("CA", "C", "O"): math.radians(120.8),
    ("CA", "C", "N"): math.radians(116.2),  # peptide
    ("C", "N", "CA"): math.radians(121.7),  # peptide
    ("O", "C", "N"): math.radians(123.0),
    # CA-CB branch
    ("N", "CA", "CB"): math.radians(110.5),
    ("C", "CA", "CB"): math.radians(110.1),
    # Common sidechain
    ("CA", "CB", "CG"): math.radians(113.8),
    ("CA", "CB", "CG1"): math.radians(111.5),
    ("CA", "CB", "OG"): math.radians(111.1),
    ("CA", "CB", "OG1"): math.radians(109.2),
    ("CA", "CB", "SG"): math.radians(114.4),
    ("CB", "CG", "CD"): math.radians(113.0),
    ("CB", "CG", "CD1"): math.radians(120.0),
    ("CB", "CG", "CD2"): math.radians(120.0),
    ("CB", "CG", "OD1"): math.radians(119.2),
    ("CB", "CG", "ND2"): math.radians(116.5),
    ("CB", "CG", "SD"): math.radians(112.7),
    ("CG", "CD", "NE"): math.radians(112.0),
    ("CG", "CD", "OE1"): math.radians(119.0),
    ("CG", "CD", "NE2"): math.radians(116.0),
    ("CG", "CD", "CE"): math.radians(111.8),
    ("CG", "SD", "CE"): math.radians(100.9),
    ("CD", "NE", "CZ"): math.radians(124.8),
    ("CD", "CE", "NZ"): math.radians(111.7),
    ("NE", "CZ", "NH1"): math.radians(120.0),
    ("NE", "CZ", "NH2"): math.radians(120.0),
    ("NH1", "CZ", "NH2"): math.radians(120.0),
}

# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------

# Expected number of real (non-empty) atoms per residue in atom14
_EXPECTED_ATOM_COUNTS: Final[dict[str, int]] = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6,
    "GLN": 9, "GLU": 9, "GLY": 4, "HIS": 10, "ILE": 8,
    "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11, "PRO": 7,
    "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}


def validate_constants() -> None:
    """Run self-checks on all constant tables. Raises AssertionError on failure."""
    # Check we have exactly 20 amino acids
    assert len(RESTYPES) == NUM_RES_TYPES, f"Expected {NUM_RES_TYPES} restypes, got {len(RESTYPES)}"
    assert len(RESTYPE_ATOM14_NAMES) == NUM_RES_TYPES

    # Check 1-to-3 and 3-to-1 are consistent and complete
    assert len(RESTYPE_1TO3) == NUM_RES_TYPES
    assert len(RESTYPE_3TO1) == NUM_RES_TYPES
    for code1, code3 in RESTYPE_1TO3.items():
        assert RESTYPE_3TO1[code3] == code1

    # Check atom37 has 37 entries and no duplicates
    assert len(ATOM37_NAMES) == NUM_ATOM37_TYPES
    assert len(set(ATOM37_NAMES)) == NUM_ATOM37_TYPES

    for resname in RESTYPES:
        atoms = RESTYPE_ATOM14_NAMES[resname]
        assert len(atoms) == NUM_ATOMS_PER_RES, f"{resname}: expected {NUM_ATOMS_PER_RES} slots"

        # Backbone at positions 0-3
        assert atoms[0] == "N", f"{resname}: slot 0 should be N, got {atoms[0]}"
        assert atoms[1] == "CA", f"{resname}: slot 1 should be CA, got {atoms[1]}"
        assert atoms[2] == "C", f"{resname}: slot 2 should be C, got {atoms[2]}"
        assert atoms[3] == "O", f"{resname}: slot 3 should be O, got {atoms[3]}"

        # Atom count
        real_atoms = [a for a in atoms if a != ""]
        expected = _EXPECTED_ATOM_COUNTS[resname]
        assert len(real_atoms) == expected, (
            f"{resname}: expected {expected} atoms, got {len(real_atoms)}: {real_atoms}"
        )

        # No duplicate atom names within a residue
        assert len(real_atoms) == len(set(real_atoms)), (
            f"{resname}: duplicate atom names: {real_atoms}"
        )

        # Every atom name must be in ATOM37_NAMES
        for aname in real_atoms:
            assert aname in ATOM37_ORDER, f"{resname}: atom {aname!r} not in ATOM37_NAMES"

    # Check mapping invertibility: atom14 → atom37 → atom14
    a14_to_a37 = get_atom14_to_atom37()
    a37_to_a14 = get_atom37_to_atom14()
    mask = get_atom14_mask()
    for i in range(NUM_RES_TYPES):
        for j in range(NUM_ATOMS_PER_RES):
            if mask[i, j] > 0:
                a37_idx = a14_to_a37[i, j].item()
                roundtrip = a37_to_a14[i, a37_idx].item()
                assert roundtrip == j, (
                    f"{RESTYPES[i]} atom14[{j}] → atom37[{a37_idx}] → atom14[{roundtrip}] != {j}"
                )

    # Check element mapping consistency
    elem_indices = get_atom14_element_indices()
    for i, resname in enumerate(RESTYPES):
        for j, aname in enumerate(RESTYPE_ATOM14_NAMES[resname]):
            if aname != "":
                expected_elem = _atom_name_to_element(aname)
                actual_idx = elem_indices[i, j].item()
                assert ELEMENTS[actual_idx] == expected_elem, (
                    f"{resname}.{aname}: element should be {expected_elem}, "
                    f"got {ELEMENTS[actual_idx]}"
                )
