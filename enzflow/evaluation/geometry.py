"""Geometry quality metrics for generated protein structures.

Evaluates physical plausibility of generated atom14 coordinates:
    - Backbone bond lengths (N-CA, CA-C, C-O, C-N peptide)
    - Backbone bond angles (N-CA-C, CA-C-N, C-N-CA)
    - Ramachandran (phi/psi dihedral angles)
    - Steric clashes (non-bonded atom pairs too close)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from enzflow.data.residue_constants import (
    IDEAL_BOND_ANGLES,
    IDEAL_BOND_LENGTHS,
)

# Backbone atom indices in atom14: N=0, CA=1, C=2, O=3
_N, _CA, _C, _O = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Bond lengths
# ---------------------------------------------------------------------------


def compute_bond_lengths(
    coords: NDArray, atom_mask: NDArray
) -> dict[str, NDArray]:
    """Compute backbone bond lengths for each residue.

    Args:
        coords: ``[N_res, 14, 3]`` atom14 coordinates.
        atom_mask: ``[N_res, 14]`` bool mask.

    Returns:
        Dict with keys ``"N_CA", "CA_C", "C_O", "C_N_pep"``, each an
        array of measured distances. ``"C_N_pep"`` has length N_res-1.
    """
    N_res = coords.shape[0]
    result: dict[str, NDArray] = {}

    # Intra-residue bonds
    for name, (i, j) in [("N_CA", (_N, _CA)), ("CA_C", (_CA, _C)), ("C_O", (_C, _O))]:
        valid = atom_mask[:, i] & atom_mask[:, j]
        d = np.linalg.norm(coords[valid, i] - coords[valid, j], axis=-1)
        result[name] = d

    # Peptide bond: C[i] -- N[i+1]
    if N_res > 1:
        valid = atom_mask[:-1, _C] & atom_mask[1:, _N]
        d = np.linalg.norm(coords[:-1, _C] - coords[1:, _N], axis=-1)
        result["C_N_pep"] = d[valid]

    # CA-CA pseudo-bond: CA[i] -- CA[i+1]
    if N_res > 1:
        valid = atom_mask[:-1, _CA] & atom_mask[1:, _CA]
        d = np.linalg.norm(coords[:-1, _CA] - coords[1:, _CA], axis=-1)
        result["CA_CA"] = d[valid]

    return result


_IDEAL_BACKBONE_LENGTHS: dict[str, float] = {
    "N_CA": IDEAL_BOND_LENGTHS[("N", "CA")],
    "CA_C": IDEAL_BOND_LENGTHS[("CA", "C")],
    "C_O": IDEAL_BOND_LENGTHS[("C", "O")],
    "C_N_pep": IDEAL_BOND_LENGTHS[("C", "N")],
    "CA_CA": 3.80,
}


# ---------------------------------------------------------------------------
# Bond angles
# ---------------------------------------------------------------------------


def _angle(a: NDArray, b: NDArray, c: NDArray) -> NDArray:
    """Angle at vertex b formed by a-b-c, in radians."""
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def compute_bond_angles(
    coords: NDArray, atom_mask: NDArray
) -> dict[str, NDArray]:
    """Compute backbone bond angles.

    Args:
        coords: ``[N_res, 14, 3]``
        atom_mask: ``[N_res, 14]``

    Returns:
        Dict with ``"N_CA_C"`` (intra-residue), ``"CA_C_N"`` and
        ``"C_N_CA"`` (inter-residue, length N_res-1). Values in radians.
    """
    N_res = coords.shape[0]
    result: dict[str, NDArray] = {}

    # Intra-residue: N-CA-C
    valid = atom_mask[:, _N] & atom_mask[:, _CA] & atom_mask[:, _C]
    ang = _angle(coords[valid, _N], coords[valid, _CA], coords[valid, _C])
    result["N_CA_C"] = ang

    if N_res > 1:
        # CA[i]-C[i]-N[i+1]
        valid = atom_mask[:-1, _CA] & atom_mask[:-1, _C] & atom_mask[1:, _N]
        ang = _angle(coords[:-1, _CA][valid], coords[:-1, _C][valid], coords[1:, _N][valid])
        result["CA_C_N"] = ang

        # C[i]-N[i+1]-CA[i+1]
        valid = atom_mask[:-1, _C] & atom_mask[1:, _N] & atom_mask[1:, _CA]
        ang = _angle(coords[:-1, _C][valid], coords[1:, _N][valid], coords[1:, _CA][valid])
        result["C_N_CA"] = ang

    return result


_IDEAL_BACKBONE_ANGLES: dict[str, float] = {
    "N_CA_C": IDEAL_BOND_ANGLES[("N", "CA", "C")],
    "CA_C_N": IDEAL_BOND_ANGLES[("CA", "C", "N")],
    "C_N_CA": IDEAL_BOND_ANGLES[("C", "N", "CA")],
}


# ---------------------------------------------------------------------------
# Ramachandran (phi/psi)
# ---------------------------------------------------------------------------


def _dihedral(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray:
    """Dihedral angle defined by four points, in radians [-pi, pi].

    Uses the Praxeolitic formula (same sign convention as IUPAC/Biopython).
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c
    # Normalize b1 for projection
    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    # Remove b1 component from b0 and b2
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)


def compute_ramachandran(
    coords: NDArray, atom_mask: NDArray
) -> dict[str, NDArray]:
    """Compute phi/psi dihedral angles.

    phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])   -- defined for i>=1
    psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])    -- defined for i<=N-2

    Args:
        coords: ``[N_res, 14, 3]``
        atom_mask: ``[N_res, 14]``

    Returns:
        Dict with ``"phi"`` and ``"psi"`` arrays in radians.
    """
    N_res = coords.shape[0]
    result: dict[str, NDArray] = {}

    if N_res < 2:
        result["phi"] = np.array([])
        result["psi"] = np.array([])
        return result

    # phi[i]: C[i-1] - N[i] - CA[i] - C[i], for i = 1..N-1
    valid_phi = (
        atom_mask[:-1, _C] & atom_mask[1:, _N] & atom_mask[1:, _CA] & atom_mask[1:, _C]
    )
    phi = _dihedral(
        coords[:-1, _C][valid_phi],
        coords[1:, _N][valid_phi],
        coords[1:, _CA][valid_phi],
        coords[1:, _C][valid_phi],
    )
    result["phi"] = phi

    # psi[i]: N[i] - CA[i] - C[i] - N[i+1], for i = 0..N-2
    valid_psi = (
        atom_mask[:-1, _N] & atom_mask[:-1, _CA] & atom_mask[:-1, _C] & atom_mask[1:, _N]
    )
    psi = _dihedral(
        coords[:-1, _N][valid_psi],
        coords[:-1, _CA][valid_psi],
        coords[:-1, _C][valid_psi],
        coords[1:, _N][valid_psi],
    )
    result["psi"] = psi

    return result


def ramachandran_allowed_ratio(phi: NDArray, psi: NDArray) -> float:
    """Fraction of residues with phi/psi in broadly allowed regions.

    Uses a generous threshold: excludes only the top-left forbidden
    region (phi > 0, except for glycine which we cannot distinguish here).
    A more rigorous check would use Procheck-style regions, but this
    simple check catches obvious failures (e.g., positive phi).

    Returns:
        Fraction in [0, 1]. Higher is better.
    """
    if len(phi) == 0:
        return 0.0
    n = min(len(phi), len(psi))
    phi, psi = phi[:n], psi[:n]
    # Broadly allowed: phi < 0 (most residues) OR the small positive-phi
    # region around phi~60 (left-handed helix, rare but allowed)
    allowed = (phi < 0) | ((phi > 0.35) & (phi < 1.57))  # ~20-90 degrees
    return float(allowed.mean())


# ---------------------------------------------------------------------------
# Steric clashes
# ---------------------------------------------------------------------------


def compute_clashes(
    coords: NDArray,
    atom_mask: NDArray,
    clash_threshold: float = 1.5,
) -> dict[str, float]:
    """Count steric clashes between non-bonded atoms.

    Only considers inter-residue atom pairs (same-residue atoms are bonded
    or near-bonded). Uses a simple distance cutoff.

    Args:
        coords: ``[N_res, 14, 3]``
        atom_mask: ``[N_res, 14]``
        clash_threshold: Minimum allowed distance in Angstroms.

    Returns:
        Dict with ``"n_clashes"`` (int) and ``"clash_ratio"`` (fraction
        of inter-residue atom pairs that clash).
    """
    N_res = coords.shape[0]

    # Extract all real atom positions
    real_atoms = []
    res_ids = []
    for i in range(N_res):
        for j in range(14):
            if atom_mask[i, j]:
                real_atoms.append(coords[i, j])
                res_ids.append(i)

    if len(real_atoms) < 2:
        return {"n_clashes": 0, "clash_ratio": 0.0}

    atoms = np.array(real_atoms)  # [M, 3]
    res_ids = np.array(res_ids)  # [M]

    # Pairwise distances — use chunking if too many atoms
    M = len(atoms)
    if M > 5000:
        # Sample a subset for large proteins
        idx = np.random.default_rng(42).choice(M, 5000, replace=False)
        atoms = atoms[idx]
        res_ids = res_ids[idx]
        M = 5000

    # [M, 1, 3] - [1, M, 3] -> [M, M]
    diffs = atoms[:, None, :] - atoms[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # Mask: inter-residue pairs with |res_i - res_j| > 1, upper triangle.
    # Adjacent residues (|i-j|=1) share a peptide bond (C-N ~1.33A) so
    # they must be excluded from clash checks.
    res_diff = np.abs(res_ids[:, None] - res_ids[None, :])
    non_neighbor = res_diff > 1
    upper = np.triu(np.ones((M, M), dtype=bool), k=1)
    valid_pairs = non_neighbor & upper

    n_valid = valid_pairs.sum()
    if n_valid == 0:
        return {"n_clashes": 0, "clash_ratio": 0.0}

    clashing = (dists < clash_threshold) & valid_pairs
    n_clashes = int(clashing.sum())

    return {
        "n_clashes": n_clashes,
        "clash_ratio": n_clashes / n_valid,
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_geometry(
    coords: NDArray,
    atom_mask: NDArray,
) -> dict[str, float]:
    """Run all geometry checks on a single structure.

    Args:
        coords: ``[N_res, 14, 3]`` float numpy array.
        atom_mask: ``[N_res, 14]`` bool numpy array.

    Returns:
        Dict of metric_name -> value. Includes per-bond-type mean/std
        absolute errors, angle errors, Ramachandran stats, and clash info.
    """
    metrics: dict[str, float] = {}

    # --- Bond lengths ---
    bond_lengths = compute_bond_lengths(coords, atom_mask)
    all_bond_errors = []
    for name, dists in bond_lengths.items():
        if len(dists) == 0:
            continue
        ideal = _IDEAL_BACKBONE_LENGTHS[name]
        errors = np.abs(dists - ideal)
        metrics[f"bond_{name}_mean"] = float(dists.mean())
        metrics[f"bond_{name}_ideal"] = ideal
        metrics[f"bond_{name}_mae"] = float(errors.mean())
        # CA_CA is a pseudo-bond, exclude from aggregate bond_mae_all
        if name != "CA_CA":
            all_bond_errors.append(errors)
    if all_bond_errors:
        all_err = np.concatenate(all_bond_errors)
        metrics["bond_mae_all"] = float(all_err.mean())
        metrics["bond_within_0.05A"] = float((all_err < 0.05).mean())

    # --- Bond angles ---
    bond_angles = compute_bond_angles(coords, atom_mask)
    all_angle_errors = []
    for name, angles in bond_angles.items():
        if len(angles) == 0:
            continue
        ideal = _IDEAL_BACKBONE_ANGLES[name]
        errors = np.abs(angles - ideal)
        metrics[f"angle_{name}_mean_deg"] = float(np.degrees(angles.mean()))
        metrics[f"angle_{name}_ideal"] = float(np.degrees(ideal))
        metrics[f"angle_{name}_mae_deg"] = float(np.degrees(errors.mean()))
        all_angle_errors.append(errors)
    if all_angle_errors:
        all_err = np.concatenate(all_angle_errors)
        metrics["angle_mae_all_deg"] = float(np.degrees(all_err.mean()))
        metrics["angle_within_5deg"] = float((all_err < np.radians(5)).mean())

    # --- Ramachandran ---
    rama = compute_ramachandran(coords, atom_mask)
    phi, psi = rama["phi"], rama["psi"]
    metrics["rama_n_residues"] = float(min(len(phi), len(psi)))
    metrics["rama_allowed_ratio"] = ramachandran_allowed_ratio(phi, psi)

    # --- Clashes ---
    clash_info = compute_clashes(coords, atom_mask)
    metrics["clash_count"] = float(clash_info["n_clashes"])
    metrics["clash_ratio"] = clash_info["clash_ratio"]

    return metrics
