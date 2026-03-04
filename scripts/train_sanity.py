"""Sanity check: overfit a few proteins to verify the training pipeline works.

Usage:
    python scripts/train_sanity.py

Expected: loss starts ~200, drops steadily. Pipeline is working if loss decreases.
"""

import os
import time

import torch

from enzflow.data.dataset import MASK_TOKEN, collate_fn
from enzflow.model import AllAtomFlowModel

# ---- Config ----
DATA_DIR = "data/processed/enzymes"
SAMPLE_IDS = [
    "A0A067XGX8",  # 512 residues
    "A0A1B4XBH6",  # 512 residues
    "A0A1P8B760",  # 512 residues
    "A0A3G2S5J6",  # 512 residues
    "A0A8R9YVA6",  # 512 residues
    "A0AK92",      # 512 residues
]
N_STEPS = 500
LR = 1e-4
PRINT_EVERY = 25

MODEL_CFG = dict(
    d_token=512, d_pair=128, d_atom=128, d_cond=512,
    n_trunk=12, n_atom_layers=3, n_heads=8, d_ec_input=1024,
)


def load_samples(data_dir: str, sample_ids: list[str]) -> list[dict[str, torch.Tensor]]:
    """Load specific protein samples by ID."""
    samples = []
    for sid in sample_ids:
        f = f"{sid}.pt"
        data = torch.load(os.path.join(data_dir, f), map_location="cpu", weights_only=False)
        coords = data["coords"]
        atom_mask = data["atom_mask"]
        aatype = data["aatype"]
        residue_index = data["residue_index"]
        n_res = len(aatype)

        motif_mask = torch.zeros(n_res, dtype=torch.bool)
        masked_aatype = torch.full_like(aatype, MASK_TOKEN)
        ec_embed = torch.zeros(1024)

        samples.append({
            "coords": coords,
            "atom_mask": atom_mask,
            "aatype": masked_aatype,
            "residue_index": residue_index,
            "motif_mask": motif_mask,
            "ec_embed": ec_embed,
            "seq_len": torch.tensor(n_res, dtype=torch.long),
        })
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, bf16: enabled")
    print(f"Loading {len(SAMPLE_IDS)} samples from {DATA_DIR}...")

    samples = load_samples(DATA_DIR, SAMPLE_IDS)
    lengths = [s["seq_len"].item() for s in samples]
    print(f"Sequence lengths: {lengths}")

    batch = collate_fn(samples)
    x_1 = batch["coords"].to(device)
    atom_mask = batch["atom_mask"].to(device)
    aatype = batch["aatype"].to(device)
    residue_index = batch["residue_index"].to(device)
    motif_mask = batch["motif_mask"].to(device)
    seq_mask = batch["seq_mask"].to(device)
    ec_embed = batch["ec_embed"].to(device)

    B = x_1.shape[0]
    print(f"Batch shape: x_1={list(x_1.shape)}")

    # Build model
    model = AllAtomFlowModel(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    loss_mask = atom_mask.float().unsqueeze(-1)
    n_atoms = loss_mask.sum()
    print(f"Total real atoms in batch: {int(n_atoms.item())}")
    print(f"\n{'Step':>6}  {'Loss':>10}  {'|v_pred|':>10}  {'Time':>8}")
    print("-" * 42)

    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        model.train()

        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        t_expand = t[:, None, None, None]
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        v_target = x_1 - x_0

        # Forward in bf16
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(
                x_t=x_t, t=t, atom_mask=atom_mask, aatype=aatype,
                residue_index=residue_index, ec_embed=ec_embed,
                motif_mask=motif_mask, seq_mask=seq_mask,
            )
            diff = (v_pred - v_target) * loss_mask
            loss = (diff ** 2).sum() / (n_atoms * 3)

        # Backward with grad scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % PRINT_EVERY == 0 or step == 1:
            v_mag = v_pred.detach().float().abs().mean().item()
            elapsed = time.time() - t0
            print(f"{step:>6}  {loss.item():>10.6f}  {v_mag:>10.6f}  {elapsed:>7.1f}s")

    print("-" * 42)
    final_loss = loss.item()
    if final_loss < 0.1:
        print(f"OK: loss={final_loss:.6f} -- pipeline works!")
    else:
        print(f"Loss={final_loss:.6f} -- still decreasing, need more steps or data")


if __name__ == "__main__":
    main()
