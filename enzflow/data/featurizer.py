"""Geometric feature computation for pair representation initialization.

Computes structural features from atom14 coordinates that will be used
to initialize the pair representation in the Pairformer. All spatial
features must be computed from **noisy coordinates x_t** during training,
never from clean x_1, to avoid information leakage.

Three feature types:
    1. Relative sequence position -- clamp(j - i, -32, 32) one-hot
    2. CA-CA distance -- RBF (radial basis function) encoding
    3. CA-CA unit direction vectors

Public API:
    - ``compute_pair_features`` -- all pair features in one call
    - ``rbf_encode`` -- standalone RBF encoding utility
"""

from __future__ import annotations

import torch
from torch import Tensor

# Backbone atom index
_CA_INDEX = 1


def rbf_encode(
    distances: Tensor,
    d_min: float = 0.0,
    d_max: float = 2.5,
    num_rbf: int = 16,
) -> Tensor:
    """Encode distances with Gaussian radial basis functions.

    Places ``num_rbf`` Gaussian kernels evenly between ``d_min`` and
    ``d_max``, and returns the (unnormalized) Gaussian response for each
    input distance.

    Default d_max=2.5 covers normalized CA-CA distances (raw 22 A / COORD_SCALE 9).

    Args:
        distances: Tensor of any shape containing distance values.
        d_min: Center of the first Gaussian.
        d_max: Center of the last Gaussian.
        num_rbf: Number of Gaussian kernels.

    Returns:
        ``Tensor[..., num_rbf]`` -- RBF-encoded distances, one extra
        trailing dimension.
    """
    mu = torch.linspace(d_min, d_max, num_rbf, device=distances.device)  # [num_rbf]
    sigma = (d_max - d_min) / num_rbf
    # distances: [...] -> [..., 1] - [num_rbf] -> [..., num_rbf]
    return torch.exp(-0.5 * ((distances.unsqueeze(-1) - mu) / sigma) ** 2)


def compute_pair_features(
    coords: Tensor,
    residue_index: Tensor,
    rel_pos_clamp: int = 32,
    num_rbf: int = 16,
) -> dict[str, Tensor]:
    """Compute geometric pair features from atom14 coordinates.

    **Important**: during training, pass noisy coordinates x_t here,
    not the clean data x_1.

    Args:
        coords: ``FloatTensor[N, 14, 3]`` -- atom14 coordinates (should
            already be SE(3)-augmented).
        residue_index: ``LongTensor[N]`` -- residue sequence numbers,
            used for relative position encoding.
        rel_pos_clamp: Clamp range for relative positions (default 32,
            giving 2*32+1 = 65 bins).
        num_rbf: Number of RBF kernels for distance encoding.

    Returns:
        Dict with:
            - ``rel_pos``:    ``LongTensor[N, N]``  -- clamped relative
              positions in [0, 2*clamp], suitable for embedding lookup.
            - ``ca_dist_rbf``: ``FloatTensor[N, N, num_rbf]`` -- RBF-encoded
              CA-CA distances.
            - ``ca_unit_vec``: ``FloatTensor[N, N, 3]`` -- unit direction
              vectors from CA_i to CA_j (zero for self-pairs).
    """
    # --- 1. Relative sequence position ---
    # residue_index: [N], compute j - i for all pairs
    # Shift to non-negative: clamp to [-clamp, clamp] then add clamp -> [0, 2*clamp]
    rel_pos = residue_index.unsqueeze(0) - residue_index.unsqueeze(1)  # [N, N]
    rel_pos = rel_pos.clamp(-rel_pos_clamp, rel_pos_clamp) + rel_pos_clamp

    # --- 2. CA-CA distance with RBF encoding ---
    ca_coords = coords[:, _CA_INDEX, :]  # [N, 3]
    ca_diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)  # [N, N, 3]  (j - i)
    ca_dist = ca_diff.norm(dim=-1)  # [N, N]
    ca_dist_rbf = rbf_encode(ca_dist, num_rbf=num_rbf)  # [N, N, num_rbf]

    # --- 3. Unit direction vectors ---
    ca_unit_vec = ca_diff / (ca_dist.unsqueeze(-1) + 1e-8)  # [N, N, 3]
    # Zero out self-pairs (i == i) to avoid noise from 0/eps
    ca_unit_vec.diagonal().zero_()

    return {
        "rel_pos": rel_pos,
        "ca_dist_rbf": ca_dist_rbf,
        "ca_unit_vec": ca_unit_vec,
    }
