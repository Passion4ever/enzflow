"""Shared pytest fixtures for enzflow tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Fix random seed for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def sample_pdb_path(tmp_path: Path) -> Path:
    """Provide the path to a small test PDB file (Trp-cage, 1L2Y-like).

    If a real PDB file exists at ``tests/data/1l2y.pdb``, use it.
    Otherwise, generate a minimal synthetic PDB for testing.
    """
    real_pdb = Path(__file__).parent / "data" / "1l2y.pdb"
    if real_pdb.exists():
        return real_pdb

    # Generate a minimal synthetic PDB with a few residues
    pdb_content = _make_synthetic_pdb()
    pdb_path = tmp_path / "synthetic.pdb"
    pdb_path.write_text(pdb_content)
    return pdb_path


def _make_synthetic_pdb() -> str:
    """Create a minimal PDB string with 5 standard residues.

    Includes ALA, GLY, MET, TRP, CYS to test various atom counts.
    Backbone atoms (N, CA, C, O) plus sidechain atoms.
    """
    lines = ["HEADER    SYNTHETIC TEST STRUCTURE"]
    atom_num = 1
    residues = [
        ("ALA", 1, [
            ("N", 1.0, 0.0, 0.0),
            ("CA", 2.0, 0.0, 0.0),
            ("C", 3.0, 0.0, 0.0),
            ("O", 3.0, 1.0, 0.0),
            ("CB", 2.0, -1.0, 0.5),
        ]),
        ("GLY", 2, [
            ("N", 4.0, 0.0, 0.0),
            ("CA", 5.0, 0.0, 0.0),
            ("C", 6.0, 0.0, 0.0),
            ("O", 6.0, 1.0, 0.0),
        ]),
        ("MET", 3, [
            ("N", 7.0, 0.0, 0.0),
            ("CA", 8.0, 0.0, 0.0),
            ("C", 9.0, 0.0, 0.0),
            ("O", 9.0, 1.0, 0.0),
            ("CB", 8.0, -1.0, 0.0),
            ("CG", 8.0, -2.0, 0.0),
            ("SD", 8.0, -3.0, 0.0),
            ("CE", 8.0, -4.0, 0.0),
        ]),
        ("TRP", 4, [
            ("N", 10.0, 0.0, 0.0),
            ("CA", 11.0, 0.0, 0.0),
            ("C", 12.0, 0.0, 0.0),
            ("O", 12.0, 1.0, 0.0),
            ("CB", 11.0, -1.0, 0.0),
            ("CG", 11.0, -2.0, 0.0),
            ("CD1", 11.5, -3.0, 0.0),
            ("CD2", 10.5, -3.0, 0.0),
            ("NE1", 11.5, -4.0, 0.0),
            ("CE2", 10.5, -4.0, 0.0),
            ("CE3", 10.0, -3.5, 0.5),
            ("CZ2", 10.0, -5.0, 0.0),
            ("CZ3", 9.5, -4.5, 0.5),
            ("CH2", 9.5, -5.5, 0.0),
        ]),
        ("CYS", 5, [
            ("N", 13.0, 0.0, 0.0),
            ("CA", 14.0, 0.0, 0.0),
            ("C", 15.0, 0.0, 0.0),
            ("O", 15.0, 1.0, 0.0),
            ("CB", 14.0, -1.0, 0.0),
            ("SG", 14.0, -2.0, 0.0),
        ]),
    ]

    for resname, resnum, atoms in residues:
        for aname, x, y, z in atoms:
            # Determine element from first alpha char
            elem = ""
            for ch in aname:
                if ch.isalpha():
                    elem = ch
                    break
            line = (
                f"ATOM  {atom_num:5d}  {aname:<4s}{resname:>3s} A{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2s}"
            )
            lines.append(line)
            atom_num += 1

    lines.append("END")
    return "\n".join(lines) + "\n"
