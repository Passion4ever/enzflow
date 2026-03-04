"""Rectified Flow matching loss for all-atom protein generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sample_t(batch_size: int, device: torch.device) -> Tensor:
    """Sample timesteps t ~ U(0, 1).

    Can be swapped for Beta distribution later.

    Returns:
        Tensor of shape [B] on *device*.
    """
    return torch.rand(batch_size, device=device)


def rectified_flow_loss(
    model: nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, float]:
    """Compute one-step rectified flow matching loss.

    Steps:
        1. x_1 = real coords, x_0 = Gaussian noise
        2. t ~ U(0,1)
        3. x_t = t * x_1 + (1 - t) * x_0
        4. v_pred = model(x_t, t, ...)
        5. loss = masked MSE(v_pred, x_1 - x_0)

    Args:
        model: AllAtomFlowModel.
        batch: Collated batch dict from dataloader.
        device: Target device.

    Returns:
        (loss, v_pred_magnitude) where loss is scalar and v_pred_magnitude
        is the mean absolute value of the predicted velocity (for monitoring).
    """
    x_1 = batch["coords"].to(device)          # [B, N, 14, 3]
    atom_mask = batch["atom_mask"].to(device)  # [B, N, 14]
    aatype = batch["aatype"].to(device)        # [B, N]
    residue_index = batch["residue_index"].to(device)  # [B, N]
    motif_mask = batch["motif_mask"].to(device)        # [B, N]
    seq_mask = batch["seq_mask"].to(device)             # [B, N]
    ec_embed = batch["ec_embed"].to(device)             # [B, d_embed]

    B = x_1.shape[0]

    # Sample noise and time
    x_0 = torch.randn_like(x_1)
    t = sample_t(B, device)
    t_expand = t[:, None, None, None]  # [B, 1, 1, 1]

    # Interpolate
    x_t = t_expand * x_1 + (1 - t_expand) * x_0
    v_target = x_1 - x_0  # [B, N, 14, 3]

    # Forward
    v_pred = model(
        x_t=x_t,
        t=t,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=residue_index,
        ec_embed=ec_embed,
        motif_mask=motif_mask,
        seq_mask=seq_mask,
    )

    # Masked MSE loss: only count real atoms
    loss_mask = atom_mask.float().unsqueeze(-1)  # [B, N, 14, 1]
    diff = (v_pred - v_target) * loss_mask
    n_atoms = loss_mask.sum().clamp(min=1.0)
    loss = (diff**2).sum() / (n_atoms * 3)

    # Monitoring: mean |v_pred|
    v_mag = v_pred.detach().float().abs().mean().item()

    return loss, v_mag
