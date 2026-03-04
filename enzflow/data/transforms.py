"""SE(3) data augmentation for non-equivariant protein generation.

Since our model is a standard Transformer (no IPA / equivariant layers),
rotational and translational invariance must be learned from data. Every
training sample is randomly rotated and centered so the model never sees
the same orientation twice.

Two public functions:
    - ``random_rotation_matrix``  -- uniform SO(3) via QR decomposition
    - ``random_se3_augmentation`` -- center on CA centroid + random rotate
"""

from __future__ import annotations

import torch
from torch import Tensor

# Backbone atom indices in atom14 representation
_CA_INDEX = 1


def random_rotation_matrix(device: torch.device | None = None) -> Tensor:
    """Sample a rotation matrix uniformly from SO(3).

    Uses the QR-decomposition method (Stewart 1980): decompose a random
    Gaussian matrix, then correct signs to obtain a proper rotation.

    Args:
        device: Torch device for the output tensor.

    Returns:
        ``FloatTensor[3, 3]`` -- a proper rotation matrix (det = +1,
        R^T R = I).
    """
    # 3x3 iid standard normal
    H = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(H)

    # Make QR decomposition unique: Q @ diag(sign(diag(R)))
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))

    # Ensure det(Q) = +1 (proper rotation, not reflection)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def random_se3_augmentation(
    coords: Tensor,
    atom_mask: Tensor | None = None,
) -> Tensor:
    """Apply random SE(3) augmentation: center then rotate.

    1. Compute the centroid of real CA atoms and subtract it so the
       protein is centered at the origin.
    2. Apply a uniformly sampled SO(3) rotation to all atoms.

    Virtual (masked-out) atom positions are also transformed so they
    remain at the CA position of their residue after augmentation.

    Args:
        coords: ``FloatTensor[N_res, 14, 3]`` -- atom14 coordinates.
        atom_mask: ``BoolTensor[N_res, 14]`` (optional) -- True where a
            real atom exists. Used to select CA atoms for centroid. If
            ``None``, all CA atoms are assumed present.

    Returns:
        ``FloatTensor[N_res, 14, 3]`` -- augmented coordinates (new tensor,
        input is not modified).
    """
    device = coords.device

    # -- Step 1: centroid of CA atoms --
    ca_coords = coords[:, _CA_INDEX, :]  # [N, 3]

    if atom_mask is not None:
        ca_mask = atom_mask[:, _CA_INDEX]  # [N]
        n_valid = ca_mask.sum().clamp(min=1)
        centroid = (ca_coords * ca_mask.unsqueeze(-1).float()).sum(dim=0) / n_valid
    else:
        centroid = ca_coords.mean(dim=0)

    coords = coords - centroid  # [N, 14, 3], broadcast

    # -- Step 2: random SO(3) rotation --
    R = random_rotation_matrix(device=device)  # [3, 3]

    # coords @ R^T  (equiv. to R @ coords^T, but batched-friendly)
    coords = torch.einsum("nai, ji -> naj", coords, R)

    return coords
