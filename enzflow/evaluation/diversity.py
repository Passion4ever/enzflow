"""Structural diversity metrics for generated protein ensembles.

Computes pairwise TM-scores between all generated structures to assess
whether the model produces diverse folds or collapses to a few modes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tmtools import tm_align


def compute_pairwise_tm(
    ca_coords_list: list[NDArray],
    sequences: list[str],
) -> NDArray:
    """Compute pairwise TM-scores between all structures.

    TM-score is normalized by the length of the second structure
    (target normalization). The matrix is NOT symmetric because
    TM-score normalization depends on target length.

    Args:
        ca_coords_list: List of CA coordinate arrays, each ``[N_i, 3]``.
        sequences: List of one-letter amino acid sequences.

    Returns:
        ``[N, N]`` matrix of TM-scores. ``tm_matrix[i][j]`` is the
        TM-score of aligning structure i to structure j (normalized
        by length of j).
    """
    N = len(ca_coords_list)
    tm_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                tm_matrix[i, j] = 1.0
                continue
            result = tm_align(
                ca_coords_list[i],
                ca_coords_list[j],
                sequences[i],
                sequences[j],
            )
            # tm_norm_chain2 = TM-score normalized by target (chain2) length
            tm_matrix[i, j] = result.tm_norm_chain2

    return tm_matrix


def evaluate_diversity(
    ca_coords_list: list[NDArray],
    sequences: list[str],
) -> dict[str, float]:
    """Compute diversity metrics for a set of generated structures.

    Args:
        ca_coords_list: List of CA coordinate arrays, each ``[N_i, 3]``.
        sequences: List of one-letter amino acid sequences.

    Returns:
        Dict with diversity metrics:
        - ``pairwise_tm_mean``: mean off-diagonal TM-score (lower = more diverse)
        - ``pairwise_tm_std``: std of off-diagonal TM-scores
        - ``pairwise_tm_median``: median off-diagonal TM-score
        - ``n_structures``: number of structures compared
    """
    N = len(ca_coords_list)
    if N < 2:
        return {
            "pairwise_tm_mean": 0.0,
            "pairwise_tm_std": 0.0,
            "pairwise_tm_median": 0.0,
            "n_structures": N,
        }

    tm_matrix = compute_pairwise_tm(ca_coords_list, sequences)

    # Extract off-diagonal entries
    mask = ~np.eye(N, dtype=bool)
    off_diag = tm_matrix[mask]

    return {
        "pairwise_tm_mean": float(off_diag.mean()),
        "pairwise_tm_std": float(off_diag.std()),
        "pairwise_tm_median": float(np.median(off_diag)),
        "n_structures": N,
    }
