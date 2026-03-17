"""Designability (self-consistency) evaluation.

Measures whether generated sequences fold back into the generated structures:
    1. Take generated structure + codesigned sequence
    2. Fold the sequence with ESMFold
    3. Compute scTM = TMscore(generated_backbone, predicted_backbone)
    4. scTM > 0.5 means the design is "designable"

ESMFold is optional and heavy (~16GB GPU memory). This module gracefully
handles the case where ESMFold is not installed.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_CA_INDEX = 1


def _check_esm_available() -> bool:
    """Check if ESMFold is available."""
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        return False


def fold_sequences_esmfold(
    sequences: list[str],
    device: str = "cuda",
) -> list[NDArray]:
    """Fold sequences using ESMFold.

    Args:
        sequences: List of amino acid sequences (one-letter codes).
        device: Torch device string.

    Returns:
        List of CA coordinate arrays, each ``[N_res, 3]``.

    Raises:
        ImportError: If ``fair-esm`` is not installed.
    """
    import esm
    import torch

    logger.info("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)

    ca_coords_list = []
    for i, seq in enumerate(sequences):
        with torch.no_grad():
            output = model.infer(seq)
        # output["positions"] shape: [1, N, 37, 3]  (atom37 format)
        # atom37 index 1 = CA
        positions = output["positions"][0].cpu().numpy()
        ca = positions[:, _CA_INDEX, :]  # [N, 3]
        ca_coords_list.append(ca)

        if (i + 1) % 10 == 0:
            logger.info("  Folded %d/%d sequences", i + 1, len(sequences))

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return ca_coords_list


def compute_sc_tm(
    gen_ca_coords: list[NDArray],
    pred_ca_coords: list[NDArray],
    sequences: list[str],
) -> NDArray:
    """Compute self-consistency TM-scores.

    Args:
        gen_ca_coords: CA coords from generated structures, each ``[N, 3]``.
        pred_ca_coords: CA coords from ESMFold predictions, each ``[N, 3]``.
        sequences: Corresponding sequences.

    Returns:
        Array of scTM scores, one per structure.
    """
    from tmtools import tm_align

    sc_tms = []
    for gen_ca, pred_ca, seq in zip(gen_ca_coords, pred_ca_coords, sequences, strict=True):
        result = tm_align(gen_ca, pred_ca, seq, seq)
        sc_tms.append(result.tm_norm_chain1)

    return np.array(sc_tms)


def evaluate_designability(
    gen_ca_coords: list[NDArray],
    sequences: list[str],
    device: str = "cuda",
) -> dict[str, float]:
    """Run full designability evaluation.

    Args:
        gen_ca_coords: CA coords from generated structures.
        sequences: Codesigned sequences.
        device: Device for ESMFold.

    Returns:
        Dict with designability metrics. Returns empty dict if ESMFold
        is not installed.
    """
    if not _check_esm_available():
        logger.warning(
            "ESMFold not installed. Skip designability evaluation. "
            "Install with: pip install fair-esm"
        )
        return {}

    logger.info("Folding %d sequences with ESMFold...", len(sequences))
    pred_ca_coords = fold_sequences_esmfold(sequences, device=device)

    sc_tms = compute_sc_tm(gen_ca_coords, pred_ca_coords, sequences)

    designable = sc_tms > 0.5
    high_quality = sc_tms > 0.8

    return {
        "sctm_mean": float(sc_tms.mean()),
        "sctm_median": float(np.median(sc_tms)),
        "sctm_std": float(sc_tms.std()),
        "designable_ratio": float(designable.mean()),
        "high_quality_ratio": float(high_quality.mean()),
        "n_samples": len(sc_tms),
    }
