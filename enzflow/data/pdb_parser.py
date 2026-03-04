"""PDB / mmCIF parser → atom14 tensor representation.

Uses Gemmi for structure I/O. Produces a :class:`ProteinStructure` dataclass
with standardised atom14 coordinates, masks, and metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gemmi
import torch
from torch import Tensor

from enzflow.data.residue_constants import (
    NONSTANDARD_RESTYPE_MAP,
    RESTYPE_ATOM14_NAMES,
    RESTYPE_ORDER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinStructure:
    """Parsed protein structure in atom14 representation.

    Attributes:
        coords: ``FloatTensor[N_res, 14, 3]`` — atom14 coordinates.
        atom_mask: ``BoolTensor[N_res, 14]`` — ``True`` where a real atom exists.
        aatype: ``LongTensor[N_res]`` — amino-acid type index (0–19).
        residue_index: ``LongTensor[N_res]`` — PDB residue sequence number.
        chain_id: ``tuple[str, ...]`` — chain identifier per residue.
    """

    coords: Tensor
    atom_mask: Tensor
    aatype: Tensor
    residue_index: Tensor
    chain_id: tuple[str, ...]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _select_atom(residue: gemmi.Residue, atom_name: str) -> gemmi.Atom | None:
    """Pick the best atom for *atom_name* from a residue, handling altloc.

    Priority: altloc '' > 'A' > highest occupancy.
    """
    candidates: list[gemmi.Atom] = []
    for atom in residue:
        if atom.name == atom_name:
            candidates.append(atom)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer blank altloc, then 'A', then highest occupancy
    for target_altloc in ("", "\x00", "A"):
        for atom in candidates:
            if atom.altloc == target_altloc:
                return atom

    # Fallback: highest occupancy
    return max(candidates, key=lambda a: a.occ)


def parse_structure(path: str | Path) -> ProteinStructure:
    """Parse a PDB or mmCIF file into a :class:`ProteinStructure`.

    Only standard amino acids are kept. Non-standard residues (including water)
    are silently skipped with a debug log.

    Args:
        path: Path to a ``.pdb``, ``.ent``, ``.cif``, or ``.mmcif`` file.
            Gzipped files are supported.

    Returns:
        A :class:`ProteinStructure` with atom14 tensors.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If parsing fails or the structure is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    # Gemmi auto-detects PDB vs mmCIF
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as exc:
        raise RuntimeError(f"Gemmi failed to parse {path}: {exc}") from exc

    structure.setup_entities()
    model = structure[0]

    all_coords: list[list[list[float]]] = []
    all_masks: list[list[bool]] = []
    all_aatype: list[int] = []
    all_residue_index: list[int] = []
    all_chain_id: list[str] = []

    for chain in model:
        for residue in chain:
            resname = residue.name

            # Map known non-standard residues to their standard parent
            if resname in NONSTANDARD_RESTYPE_MAP:
                logger.debug(
                    "Mapping non-standard residue %s → %s: chain=%s seq=%d",
                    resname, NONSTANDARD_RESTYPE_MAP[resname],
                    chain.name, residue.seqid.num,
                )
                resname = NONSTANDARD_RESTYPE_MAP[resname]

            if resname not in RESTYPE_ORDER:
                logger.debug(
                    "Skipping unknown residue: %s %s %d",
                    chain.name, residue.name, residue.seqid.num,
                )
                continue

            # Check for CA — skip residue if missing
            ca_atom = _select_atom(residue, "CA")
            if ca_atom is None:
                logger.warning(
                    "Skipping residue without CA: chain=%s res=%s seq=%d",
                    chain.name, resname, residue.seqid.num,
                )
                continue

            ca_pos = [ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z]

            atom14_names = RESTYPE_ATOM14_NAMES[resname]
            res_coords: list[list[float]] = []
            res_mask: list[bool] = []

            for aname in atom14_names:
                if aname == "":
                    # Empty slot
                    res_coords.append([0.0, 0.0, 0.0])
                    res_mask.append(False)
                else:
                    atom = _select_atom(residue, aname)
                    if atom is not None:
                        res_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        res_mask.append(True)
                    else:
                        # Missing atom: fill with CA position, mask = False
                        logger.debug(
                            "Missing atom %s in %s %s %d — filling with CA coords",
                            aname, chain.name, resname, residue.seqid.num,
                        )
                        res_coords.append(ca_pos)
                        res_mask.append(False)

            all_coords.append(res_coords)
            all_masks.append(res_mask)
            all_aatype.append(RESTYPE_ORDER[resname])
            all_residue_index.append(residue.seqid.num)
            all_chain_id.append(chain.name)

    if not all_coords:
        raise RuntimeError(f"No standard residues found in {path}")

    return ProteinStructure(
        coords=torch.tensor(all_coords, dtype=torch.float32),
        atom_mask=torch.tensor(all_masks, dtype=torch.bool),
        aatype=torch.tensor(all_aatype, dtype=torch.long),
        residue_index=torch.tensor(all_residue_index, dtype=torch.long),
        chain_id=tuple(all_chain_id),
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_pdb(structure: ProteinStructure, path: str | Path) -> None:
    """Write a :class:`ProteinStructure` to a PDB file.

    Only atoms where ``atom_mask`` is ``True`` are written.

    Args:
        structure: The protein structure to write.
        path: Output PDB file path.
    """
    path = Path(path)
    gemmi_structure = gemmi.Structure()
    gemmi_model = gemmi.Model("1")

    from enzflow.data.residue_constants import RESTYPES

    # Group residues by chain
    chain_residues: dict[str, list[int]] = {}
    for idx, cid in enumerate(structure.chain_id):
        chain_residues.setdefault(cid, []).append(idx)

    for cid, res_indices in chain_residues.items():
        gemmi_chain = gemmi.Chain(cid)
        for res_idx in res_indices:
            resname = RESTYPES[structure.aatype[res_idx].item()]
            atom14_names = RESTYPE_ATOM14_NAMES[resname]

            gemmi_residue = gemmi.Residue()
            gemmi_residue.name = resname
            gemmi_residue.seqid = gemmi.SeqId(str(structure.residue_index[res_idx].item()))

            for atom_j in range(14):
                if not structure.atom_mask[res_idx, atom_j].item():
                    continue
                aname = atom14_names[atom_j]
                if aname == "":
                    continue

                gemmi_atom = gemmi.Atom()
                gemmi_atom.name = aname
                coords = structure.coords[res_idx, atom_j]
                gemmi_atom.pos = gemmi.Position(
                    coords[0].item(), coords[1].item(), coords[2].item()
                )
                gemmi_atom.occ = 1.0
                gemmi_atom.b_iso = 0.0
                # Set element
                for ch in aname:
                    if ch.isalpha():
                        gemmi_atom.element = gemmi.Element(ch)
                        break

                gemmi_residue.add_atom(gemmi_atom)

            gemmi_chain.add_residue(gemmi_residue)
        gemmi_model.add_chain(gemmi_chain)

    gemmi_structure.add_model(gemmi_model)
    gemmi_structure.write_pdb(str(path))
