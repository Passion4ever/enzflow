"""Unconditional protein sampling from a trained enzflow model.

Generates protein structures + sequences via Euler ODE integration of the
learned velocity field, starting from pure Gaussian noise (t=0) to data (t=1).

Total samples = batch_size x num_batches. Tune batch_size to fit GPU memory.

Usage:
    # 4 samples/batch x 5 batches = 20 samples, length 100
    CUDA_VISIBLE_DEVICES=1 python scripts/sample.py \
        --ckpt checkpoints/pretrain_v2_0306_005732/step_820000.pt \
        --batch_size 4 --num_batches 5 --length 100

    # OOM? Lower batch_size
    python scripts/sample.py --ckpt CKPT --batch_size 1 --num_batches 20 --length 200

    # Multiple lengths: each batch uses the next length in cycle
    python scripts/sample.py --ckpt CKPT --batch_size 4 --num_batches 3 \
        --length 80 150 300

    # More Euler steps for better quality
    python scripts/sample.py --ckpt CKPT --batch_size 4 --num_batches 5 \
        --length 100 --steps 50

Output directory structure (real data on /data03, symlinked from outputs/):
    outputs/
      sample_pretrain_v2_step820000_0308_163000/
        sample_0000_L100.pdb
        sample_0001_L100.pdb
        ...
        sequences.fasta    # for ESMFold
        metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor

from enzflow.data.pdb_parser import ProteinStructure, write_pdb
from enzflow.data.residue_constants import RESTYPE_3TO1, RESTYPES, get_atom14_mask
from enzflow.data.transforms import COORD_SCALE
from enzflow.model import AllAtomFlowModel

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str, device: torch.device) -> AllAtomFlowModel:
    """Load model weights from a training checkpoint.

    Reads model hyperparams from the checkpoint's ``config`` dict
    (same dict written by ``save_checkpoint`` in training).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]

    logger.info(
        "Model config: d_token=%d, n_trunk=%d, loaded from step %d",
        model_cfg["d_token"],
        model_cfg["n_trunk"],
        ckpt["step"],
    )

    model = AllAtomFlowModel(**model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Euler ODE sampler
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_batch(
    model: AllAtomFlowModel,
    batch_size: int,
    seq_len: int,
    num_steps: int = 20,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of proteins via Euler ODE integration.

    Integration goes from t=0 (noise) to t=1 (data) in ``num_steps`` steps.

    atom_mask strategy: the model is sensitive to atom_mask (trained with real
    per-residue masks). At the start we use the average mask across all 20 amino
    acids (backbone=1.0, sidechain~0.3-0.7). As ODE progresses and seq_logits
    become more confident, we linearly blend toward the hard mask derived from
    the predicted aatype. This keeps early steps in-distribution while letting
    later steps use precise masks.

    Args:
        model: Trained AllAtomFlowModel in eval mode.
        batch_size: Number of proteins to generate in parallel.
        seq_len: Number of residues per protein.
        num_steps: Euler integration steps (more = better quality).
        device: CUDA device.

    Returns:
        (coords, aatype, atom_mask):
            coords: ``[B, N, 14, 3]`` -- generated atom14 coordinates.
            aatype: ``[B, N]`` -- predicted amino acid types (0-19).
            atom_mask: ``[B, N, 14]`` -- bool mask for real atoms.
    """
    if device is None:
        device = torch.device("cuda")

    B = batch_size
    N = seq_len

    # Atom14 mask lookup: [20, 14] float, 1.0 where atom exists
    atom14_mask_table = get_atom14_mask().to(device)  # [20, 14]

    # Average mask: mean over all 20 amino acids.
    # backbone (N,CA,C,O) = 1.0, sidechain slots = 0.3~0.7
    avg_mask = atom14_mask_table.mean(dim=0)  # [14]

    # --- Initial state: scaled Gaussian noise ---
    # Noise scale matches expected data scale for target length.
    # Fitted from training data: noise_scale = 0.133 * N^0.40
    noise_scale = 0.133 * (N ** 0.40)
    x = noise_scale * torch.randn(B, N, 14, 3, device=device)

    # --- Unconditional inputs ---
    ec_embed = torch.zeros(B, 1024, device=device)
    aatype_input = torch.full((B, N), 20, dtype=torch.long, device=device)
    motif_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    seq_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    residue_index = (
        torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)
    )

    # Initial atom_mask: random per-position AA type to avoid 8-atom bias.
    # avg_mask rounds to exactly the 8-atom pattern shared by N/D/I/L/M,
    # which locks seq_head into predicting only those 5 AA types.
    # Random sampling gives diverse atom counts (4-14) matching training.
    random_aa = torch.randint(0, 20, (B, N), device=device)
    atom_mask = atom14_mask_table[random_aa].bool()  # [B, N, 14]

    dt = 1.0 / num_steps

    for step_i in range(num_steps):
        t_val = step_i * dt
        t = torch.full((B,), t_val, device=device)

        v_pred, seq_logits = model(
            x_t=x,
            t=t,
            atom_mask=atom_mask,
            aatype=aatype_input,
            residue_index=residue_index,
            ec_embed=ec_embed,
            motif_mask=motif_mask,
            seq_mask=seq_mask,
        )

        # Clamp velocity to prevent divergence (normalized coord space)
        v_pred = v_pred.clamp(-10, 10)

        # Euler step: x_{t+dt} = x_t + dt * v
        x = x + dt * v_pred

        # Update atom_mask: use seq_logits probability-weighted mask.
        # soft_mask[b,n,j] = sum_aa p(aa) * mask_table[aa,j]
        # As predictions sharpen, this converges to the hard mask.
        seq_probs = seq_logits.softmax(dim=-1)  # [B, N, 20]
        soft_mask = torch.einsum("bna,aj->bnj", seq_probs, atom14_mask_table)

        # After first step, trust seq_logits directly (soft_mask).
        # No blending with avg_mask needed since initial mask is already diverse.
        atom_mask = soft_mask >= 0.5  # [B, N, 14]

    # Final sequence: argmax of last seq_logits
    aatype_pred = seq_logits.argmax(dim=-1)  # [B, N]
    atom_mask_final = atom14_mask_table[aatype_pred].bool()  # [B, N, 14]

    return x, aatype_pred, atom_mask_final


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def aatype_to_seq(aatype: Tensor) -> str:
    """Convert aatype tensor (0-19) to one-letter sequence string."""
    return "".join(RESTYPE_3TO1[RESTYPES[aa.item()]] for aa in aatype)


def save_sample(
    coords: Tensor,
    aatype: Tensor,
    atom_mask: Tensor,
    out_path: Path,
) -> str:
    """Save generated structure as PDB. Returns the sequence string."""
    N = coords.shape[0]
    structure = ProteinStructure(
        coords=(coords * COORD_SCALE).cpu().float(),
        atom_mask=atom_mask.cpu(),
        aatype=aatype.cpu(),
        residue_index=torch.arange(1, N + 1),
        chain_id=tuple("A" for _ in range(N)),
    )
    write_pdb(structure, out_path)
    return aatype_to_seq(aatype)


def make_run_name(ckpt_path: str) -> str:
    """Derive a run name from the checkpoint path.

    Example: checkpoints/pretrain_v2_0306_005732/step_820000.pt
          -> sample_pretrain_v2_step820000_0308_163000
    """
    ckpt = Path(ckpt_path)
    # Extract parent dir name (run name) and step
    run_part = ckpt.parent.name  # e.g. pretrain_v2_0306_005732
    step_match = re.search(r"step_(\d+)", ckpt.stem)
    step_part = f"step{step_match.group(1)}" if step_match else ckpt.stem
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    return f"sample_{run_part}_{step_part}_{timestamp}"


def setup_output_dir(run_name: str) -> Path:
    """Create output directory under outputs/{run_name}/."""
    run_dir = _OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sample proteins from a trained enzflow model"
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Samples per batch (tune for GPU memory)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches (total = batch_size x num_batches)",
    )
    parser.add_argument(
        "--length",
        type=int,
        nargs="+",
        default=[100],
        help="Sequence length(s). Each batch uses the next length in cycle.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of Euler integration steps",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # --- Output dir ---
    run_name = make_run_name(args.ckpt)
    outdir = setup_output_dir(run_name)

    # --- Load model ---
    logger.info("Loading checkpoint: %s", args.ckpt)
    model = load_model(args.ckpt, device)

    # --- Sample ---
    # Each length gets batch_size x num_batches samples
    lengths = args.length
    samples_per_length = args.batch_size * args.num_batches
    total_samples = samples_per_length * len(lengths)
    all_sequences = []
    sample_idx = 0
    t0 = time.time()

    logger.info(
        "Generating %d samples (%d per length x %d lengths), steps=%d",
        total_samples,
        samples_per_length,
        len(lengths),
        args.steps,
    )
    logger.info("Lengths: %s (%d batches x %d each)", lengths, args.num_batches, args.batch_size)
    logger.info("Output: %s", outdir)
    if args.seed is not None:
        logger.info("Seed: %d", args.seed)
    logger.info("")

    total_batches = args.num_batches * len(lengths)
    batch_count = 0

    for seq_len in lengths:
        for _bi in range(args.num_batches):
            batch_count += 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                coords, aatype, atom_mask = sample_batch(
                    model,
                    args.batch_size,
                    seq_len,
                    num_steps=args.steps,
                    device=device,
                )

            # Save each sample in the batch
            for j in range(args.batch_size):
                pdb_path = outdir / f"sample_{sample_idx:04d}_L{seq_len}.pdb"
                seq = save_sample(
                    coords[j], aatype[j], atom_mask[j], pdb_path,
                )
                all_sequences.append({
                    "id": sample_idx,
                    "length": seq_len,
                    "sequence": seq,
                })
                sample_idx += 1

            batch_seqs = [e["sequence"] for e in all_sequences[-args.batch_size:]]
            preview = batch_seqs[0][:30] + ("..." if len(batch_seqs[0]) > 30 else "")
            logger.info(
                "Batch %d/%d  L=%d  [%s]  %d samples done",
                batch_count,
                total_batches,
                seq_len,
                preview,
                sample_idx,
            )

    elapsed = time.time() - t0
    logger.info("")
    logger.info(
        "Done. %d samples in %.1fs (%.2fs/sample)",
        total_samples,
        elapsed,
        elapsed / total_samples,
    )

    # --- Save sequences as FASTA (for ESMFold) ---
    fasta_path = outdir / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for entry in all_sequences:
            f.write(f">sample_{entry['id']:04d}_L{entry['length']}\n")
            f.write(f"{entry['sequence']}\n")
    logger.info("FASTA: %s", fasta_path)

    # --- Save metadata ---
    meta_path = outdir / "metadata.json"
    meta = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "total_samples": total_samples,
        "lengths": lengths,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "samples": all_sequences,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata: %s", meta_path)
    logger.info("")
    logger.info("Project path: outputs/%s/", run_name)


if __name__ == "__main__":
    main()
