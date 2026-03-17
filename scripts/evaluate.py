"""Evaluate generated protein structures from sample.py output.

Reads a sample output directory and runs geometry, diversity, and
optionally designability (ESMFold) evaluations.

Usage:
    # Geometry + diversity only (no GPU needed beyond tmtools)
    python scripts/evaluate.py --sample_dir outputs/sample_pretrain_v3_.../

    # Include designability (needs ESMFold, ~16GB GPU)
    python scripts/evaluate.py --sample_dir outputs/sample_pretrain_v3_.../ --designability

    # Limit number of structures (for quick check)
    python scripts/evaluate.py --sample_dir outputs/sample_pretrain_v3_.../ --max_samples 20

Output:
    Prints metrics to console and saves eval_results.json in the sample dir.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from enzflow.data.pdb_parser import parse_structure
from enzflow.evaluation.designability import evaluate_designability
from enzflow.evaluation.diversity import evaluate_diversity
from enzflow.evaluation.geometry import evaluate_geometry

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_CA_INDEX = 1


def load_samples(
    sample_dir: Path, max_samples: int | None = None
) -> tuple[list, list[str]]:
    """Load generated structures from PDB files.

    Args:
        sample_dir: Directory containing sample_XXXX_LYYY.pdb files.
        max_samples: If set, only load this many structures.

    Returns:
        (structures, sequences): parsed ProteinStructure list and
        one-letter sequence strings.
    """
    from enzflow.data.residue_constants import RESTYPE_3TO1, RESTYPES

    pdb_files = sorted(sample_dir.glob("sample_*.pdb"))
    if max_samples:
        pdb_files = pdb_files[:max_samples]

    if not pdb_files:
        raise FileNotFoundError(f"No sample_*.pdb files in {sample_dir}")

    structures = []
    sequences = []
    for pdb_path in pdb_files:
        struct = parse_structure(pdb_path)
        structures.append(struct)
        seq = "".join(
            RESTYPE_3TO1[RESTYPES[aa.item()]] for aa in struct.aatype
        )
        sequences.append(seq)

    logger.info("Loaded %d structures from %s", len(structures), sample_dir)
    return structures, sequences


def run_geometry(structures: list) -> dict[str, float]:
    """Run geometry evaluation on all structures, return aggregated metrics."""
    all_metrics: dict[str, list[float]] = {}

    for struct in structures:
        coords = struct.coords.numpy()
        atom_mask = struct.atom_mask.numpy()
        m = evaluate_geometry(coords, atom_mask)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    # Aggregate: mean across structures
    agg: dict[str, float] = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        if k.endswith("_ideal"):
            agg[k] = float(arr[0])
        else:
            agg[f"{k}_mean"] = float(arr.mean())
            if len(arr) > 1:
                agg[f"{k}_std"] = float(arr.std())

    return agg


def run_diversity(structures: list, sequences: list[str]) -> dict[str, float]:
    """Run diversity evaluation."""
    ca_coords_list = [
        struct.coords[:, _CA_INDEX, :].numpy() for struct in structures
    ]
    return evaluate_diversity(ca_coords_list, sequences)


def run_designability(
    structures: list, sequences: list[str], device: str
) -> dict[str, float]:
    """Run designability evaluation."""
    ca_coords_list = [
        struct.coords[:, _CA_INDEX, :].numpy() for struct in structures
    ]
    return evaluate_designability(ca_coords_list, sequences, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated protein structures"
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to sample.py output directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of structures to evaluate",
    )
    parser.add_argument(
        "--designability",
        action="store_true",
        help="Run ESMFold designability check (needs GPU + fair-esm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for ESMFold",
    )
    parser.add_argument(
        "--no_diversity",
        action="store_true",
        help="Skip diversity (pairwise TM-score) computation",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    structures, sequences = load_samples(sample_dir, args.max_samples)

    results: dict[str, dict] = {}

    # --- Geometry ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Geometry evaluation")
    logger.info("=" * 60)
    geo = run_geometry(structures)
    results["geometry"] = geo

    logger.info("  Bond lengths (MAE):")
    for name in ["N_CA", "CA_C", "C_O", "C_N_pep"]:
        key = f"bond_{name}_mae_mean"
        if key in geo:
            logger.info(
                "    %-8s  %.4f A  (ideal %.3f, measured %.3f)",
                name,
                geo[key],
                geo.get(f"bond_{name}_ideal", 0),
                geo.get(f"bond_{name}_mean_mean", 0),
            )
    if "bond_mae_all_mean" in geo:
        logger.info(
            "    ALL      MAE=%.4f A,  within 0.05A: %.1f%%",
            geo["bond_mae_all_mean"],
            geo.get("bond_within_0.05A_mean", 0) * 100,
        )

    logger.info("  Bond angles (MAE):")
    for name in ["N_CA_C", "CA_C_N", "C_N_CA"]:
        key = f"angle_{name}_mae_deg_mean"
        if key in geo:
            logger.info(
                "    %-8s  MAE %.2f deg  (ideal %.1f, measured %.1f)",
                name,
                geo[key],
                geo.get(f"angle_{name}_ideal", 0),
                geo.get(f"angle_{name}_mean_deg_mean", 0),
            )
    if "angle_within_5deg_mean" in geo:
        logger.info(
            "    ALL      within 5 deg: %.1f%%",
            geo["angle_within_5deg_mean"] * 100,
        )

    logger.info("  Ramachandran:")
    if "rama_allowed_ratio_mean" in geo:
        logger.info(
            "    Allowed ratio: %.1f%%", geo["rama_allowed_ratio_mean"] * 100
        )

    logger.info("  Clashes:")
    if "clash_count_mean" in geo:
        logger.info(
            "    Count: %.1f,  Ratio: %.4f",
            geo["clash_count_mean"],
            geo.get("clash_ratio_mean", 0),
        )

    # --- Diversity ---
    if not args.no_diversity:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Diversity evaluation")
        logger.info("=" * 60)
        div = run_diversity(structures, sequences)
        results["diversity"] = div
        logger.info(
            "  Pairwise TM-score: mean=%.3f, std=%.3f, median=%.3f  (N=%d)",
            div["pairwise_tm_mean"],
            div["pairwise_tm_std"],
            div["pairwise_tm_median"],
            div["n_structures"],
        )
        logger.info("  (Lower mean = more diverse. Random ~ 0.17)")

    # --- Designability ---
    if args.designability:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Designability evaluation (ESMFold)")
        logger.info("=" * 60)
        des = run_designability(structures, sequences, args.device)
        results["designability"] = des
        if des:
            logger.info(
                "  scTM: mean=%.3f, median=%.3f, std=%.3f",
                des["sctm_mean"],
                des["sctm_median"],
                des["sctm_std"],
            )
            logger.info(
                "  Designable (scTM>0.5): %.1f%%  (%d/%d)",
                des["designable_ratio"] * 100,
                int(des["designable_ratio"] * des["n_samples"]),
                des["n_samples"],
            )
            logger.info(
                "  High quality (scTM>0.8): %.1f%%",
                des["high_quality_ratio"] * 100,
            )

    # --- Save results ---
    out_path = sample_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
