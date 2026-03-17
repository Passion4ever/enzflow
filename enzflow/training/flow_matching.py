"""Rectified Flow matching loss for all-atom protein generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from enzflow.data.transforms import COORD_SCALE

# Backbone atom indices in atom14 representation
_N, _CA, _C, _O = 0, 1, 2, 3

# Ideal backbone bond lengths in Angstroms (from Engh & Huber)
_IDEAL_BONDS: dict[str, tuple[tuple[int, int], float, bool]] = {
    # name: ((atom_i, atom_j), ideal_length, is_inter_residue)
    "N_CA": ((_N, _CA), 1.458, False),
    "CA_C": ((_CA, _C), 1.525, False),
    "C_O": ((_C, _O), 1.229, False),
    "C_N_pep": ((_C, _N), 1.329, True),  # C[i] -- N[i+1]
}


def sample_t(batch_size: int, device: torch.device) -> Tensor:
    """Sample timesteps t ~ U(0, 1).

    Can be swapped for Beta distribution later.

    Returns:
        Tensor of shape [B] on *device*.
    """
    return torch.rand(batch_size, device=device)


def _backbone_bond_loss(
    x_1_pred: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Compute MSE loss on backbone bond lengths vs ideal values.

    Args:
        x_1_pred: Predicted clean coordinates [B, N, 14, 3].
        atom_mask: Atom validity mask [B, N, 14].

    Returns:
        Scalar bond length loss (mean squared error in Angstroms^2).
    """
    losses = []

    for _name, ((ai, aj), ideal, is_inter) in _IDEAL_BONDS.items():
        if is_inter:
            # Inter-residue: C[i] -- N[i+1]
            valid = atom_mask[:, :-1, ai] & atom_mask[:, 1:, aj]  # [B, N-1]
            diff = x_1_pred[:, :-1, ai, :] - x_1_pred[:, 1:, aj, :]  # [B, N-1, 3]
        else:
            # Intra-residue
            valid = atom_mask[:, :, ai] & atom_mask[:, :, aj]  # [B, N]
            diff = x_1_pred[:, :, ai, :] - x_1_pred[:, :, aj, :]  # [B, N, 3]

        dist = diff.norm(dim=-1) * COORD_SCALE  # convert back to Angstrom
        error = (dist - ideal) ** 2  # squared error vs ideal (Angstrom)
        error = error * valid.float()

        n_valid = valid.sum().clamp(min=1.0)
        losses.append(error.sum() / n_valid)

    return torch.stack(losses).mean()


def rectified_flow_loss(
    model: nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
    seq_loss_weight: float = 0.5,
    bond_loss_weight: float = 0.0,
) -> tuple[Tensor, float, float, float, float]:
    """Compute rectified flow matching loss + sequence CE loss + bond loss.

    Steps:
        1. x_1 = real coords, x_0 = Gaussian noise
        2. t ~ U(0,1)
        3. x_t = t * x_1 + (1 - t) * x_0
        4. Mask aatype for non-motif positions (prevent shortcut)
        5. v_pred, seq_logits = model(x_t, t, ...)
        6. coord_loss = masked MSE(v_pred, x_1 - x_0)
        7. seq_loss = CE(seq_logits, aatype) on valid positions
        8. bond_loss = MSE(predicted bond lengths vs ideal) on x_1_pred
        9. loss = coord_loss + seq_loss_weight * seq_loss + bond_loss_weight * bond_loss

    The bond loss is computed on x_1_pred = x_t + (1-t)*v_pred, the model's
    prediction of the clean structure. Gradients are naturally scaled by (1-t),
    making the loss most effective at intermediate timesteps.

    Args:
        model: AllAtomFlowModel.
        batch: Collated batch dict from dataloader.
        device: Target device.
        seq_loss_weight: Weight for sequence CE loss.
        bond_loss_weight: Weight for auxiliary bond length loss. 0 = disabled.

    Returns:
        (loss, v_mag, coord_loss_val, seq_loss_val, bond_loss_val).
    """
    x_1 = batch["coords"].to(device)          # [B, N, 14, 3]
    atom_mask = batch["atom_mask"].to(device)  # [B, N, 14]
    aatype = batch["aatype"].to(device)        # [B, N] real (0-19), pad=20
    residue_index = batch["residue_index"].to(device)  # [B, N]
    motif_mask = batch["motif_mask"].to(device)        # [B, N]
    seq_mask = batch["seq_mask"].to(device)             # [B, N]
    ec_embed = batch["ec_embed"].to(device)             # [B, d_embed]

    B = x_1.shape[0]

    # Sample noise and time
    # Adaptive noise: scale noise to match each sample's coordinate std.
    # This ensures the noise-data scale ratio is ~1:1 for all protein sizes.
    ca = x_1[:, :, 1, :]  # [B, N, 3] CA coords (already normalized)
    ca_valid = ca * seq_mask.unsqueeze(-1).float()  # zero out padding
    n_valid = seq_mask.sum(dim=1).float().clamp(min=1)  # [B]
    ca_var = (ca_valid ** 2).sum(dim=(1, 2)) / (n_valid * 3)  # [B]
    noise_scale = ca_var.sqrt().clamp(min=0.5)[:, None, None, None]  # [B,1,1,1]
    x_0 = noise_scale * torch.randn_like(x_1)
    t = sample_t(B, device)
    t_expand = t[:, None, None, None]  # [B, 1, 1, 1]

    # Interpolate
    x_t = t_expand * x_1 + (1 - t_expand) * x_0
    v_target = x_1 - x_0  # [B, N, 14, 3]

    # Mask aatype for encoder: non-motif positions -> MASK token (20).
    # Prevents encoder from leaking real sequence to seq_head.
    aatype_input = aatype.clone()
    aatype_input[~motif_mask] = 20

    # Forward
    v_pred, seq_logits = model(
        x_t=x_t,
        t=t,
        atom_mask=atom_mask,
        aatype=aatype_input,
        residue_index=residue_index,
        ec_embed=ec_embed,
        motif_mask=motif_mask,
        seq_mask=seq_mask,
    )

    # --- Coordinate loss: masked MSE ---
    loss_mask = atom_mask.float().unsqueeze(-1)  # [B, N, 14, 1]
    diff = (v_pred - v_target) * loss_mask
    n_atoms = loss_mask.sum().clamp(min=1.0)
    coord_loss = (diff**2).sum() / (n_atoms * 3)

    # --- Sequence loss: CE on real, non-motif positions ---
    # Target: original aatype (0-19). Padding positions (aatype=20) are
    # excluded by seq_mask. Motif positions are excluded by ~motif_mask.
    seq_loss_mask = seq_mask & ~motif_mask  # [B, N]
    if seq_loss_mask.any():
        seq_loss = F.cross_entropy(
            seq_logits[seq_loss_mask],   # [M, 20]
            aatype[seq_loss_mask],       # [M] real labels 0-19
        )
    else:
        seq_loss = torch.zeros(1, device=device, dtype=coord_loss.dtype)

    loss = coord_loss + seq_loss_weight * seq_loss

    # --- Bond length loss on predicted clean structure ---
    bond_loss_val = 0.0
    if bond_loss_weight > 0:
        # x_1_pred = x_t + (1-t) * v_pred
        # Gradient flows through v_pred, naturally scaled by (1-t)
        x_1_pred = x_t + (1 - t_expand) * v_pred
        bond_loss = _backbone_bond_loss(x_1_pred, atom_mask.bool())
        loss = loss + bond_loss_weight * bond_loss
        bond_loss_val = bond_loss.item()

    # Monitoring
    v_mag = v_pred.detach().float().abs().mean().item()

    return loss, v_mag, coord_loss.item(), seq_loss.item(), bond_loss_val
