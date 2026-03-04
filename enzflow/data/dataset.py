"""Dataset and DataLoader for enzflow protein generation training.

Data flow:
    .pt file -> torch.load -> SE(3) augment -> EC weighted sampling
    -> motif mask + aatype masking -> sample dict -> collate_fn pad -> batch

Pair features are NOT computed here -- they must be computed from noisy
coordinates x_t in the training loop to avoid information leakage.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from enzflow.data.residue_constants import NUM_RES_TYPES
from enzflow.data.transforms import random_se3_augmentation

# Mask token index: one past the 20 standard amino acids
MASK_TOKEN: int = NUM_RES_TYPES  # 20


# ---------------------------------------------------------------------------
# 1. EC Embedding Lookup
# ---------------------------------------------------------------------------


class ECEmbeddingLookup:
    """Lookup table for EC number semantic embeddings.

    The .pt file has structure::

        {"vectors": {(ec_str, level): Tensor[d_embed]}, "d_embed": int, ...}

    Levels 1-5 correspond to increasingly specific EC descriptions.
    """

    def __init__(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._vectors: dict[tuple[str, int], Tensor] = data["vectors"]
        self._d_embed: int = data["d_embed"]

    @property
    def d_embed(self) -> int:
        return self._d_embed

    def get_zero_vector(self) -> Tensor:
        """Return a zero embedding vector [d_embed]."""
        return torch.zeros(self._d_embed)

    def lookup(self, ec: str, level: int) -> Tensor:
        """Look up the embedding for (ec, level). Raises KeyError if missing."""
        return self._vectors[(ec, level)]


# ---------------------------------------------------------------------------
# 2. Protein Dataset
# ---------------------------------------------------------------------------


class ProteinDataset(Dataset):
    """Loads preprocessed .pt protein files for flow matching training.

    Each sample is SE(3)-augmented on the fly. EC embeddings are sampled at
    a random granularity level, and motif / aatype masking is applied.

    Args:
        data_dir: Directory containing ``{id}.pt`` files.
        split_ids: If given, only load these IDs. Otherwise load all .pt files.
        ec_lookup: EC embedding lookup table. None for AFDB pretraining.
        motif_prob: Probability of enabling motif conditioning per sample.
        motif_min_res: Minimum number of motif residues.
        motif_max_res: Maximum number of motif residues.
        ec_level_weights: Sampling weights for levels [0,1,2,3,4,5].
            Level 0 produces a zero vector (EC dropout).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_ids: list[str] | None = None,
        ec_lookup: ECEmbeddingLookup | None = None,
        motif_prob: float = 0.5,
        motif_min_res: int = 2,
        motif_max_res: int = 8,
        ec_level_weights: tuple[float, ...] = (0.10, 0.10, 0.15, 0.15, 0.20, 0.30),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ec_lookup = ec_lookup
        self.motif_prob = motif_prob
        self.motif_min_res = motif_min_res
        self.motif_max_res = motif_max_res
        self.ec_level_weights = ec_level_weights

        # Discover files
        if split_ids is not None:
            self.paths = [self.data_dir / f"{sid}.pt" for sid in split_ids]
            self.paths = [p for p in self.paths if p.exists()]
        else:
            self.paths = sorted(self.data_dir.glob("*.pt"))

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

        # Cache lengths for BucketBatchSampler (cheap: load only aatype shape)
        self._lengths: list[int] = []
        for p in self.paths:
            data = torch.load(p, map_location="cpu", weights_only=False)
            self._lengths.append(len(data["aatype"]))

    def __len__(self) -> int:
        return len(self.paths)

    def get_length(self, idx: int) -> int:
        """Return residue count for sample *idx* (for BucketBatchSampler)."""
        return self._lengths[idx]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        data = torch.load(self.paths[idx], map_location="cpu", weights_only=False)

        coords: Tensor = data["coords"]        # [N, 14, 3]
        atom_mask: Tensor = data["atom_mask"]   # [N, 14]
        aatype: Tensor = data["aatype"]         # [N]
        residue_index: Tensor = data["residue_index"]  # [N]
        ec_numbers: list[str] = data.get("ec_numbers", [])

        N = len(aatype)

        # -- 1. SE(3) augmentation --
        coords = random_se3_augmentation(coords, atom_mask)

        # -- 2. EC embedding sampling --
        ec_embed = self._sample_ec_embed(ec_numbers)

        # -- 3. Motif + aatype masking --
        aatype, motif_mask = self._apply_motif_masking(aatype, N)

        return {
            "coords": coords,                               # [N, 14, 3]
            "atom_mask": atom_mask,                          # [N, 14]
            "aatype": aatype,                                # [N]
            "residue_index": residue_index,                  # [N]
            "motif_mask": motif_mask,                         # [N]
            "ec_embed": ec_embed,                             # [d_embed]
            "seq_len": torch.tensor(N, dtype=torch.long),    # scalar
        }

    # -- private helpers --

    def _sample_ec_embed(self, ec_numbers: list[str]) -> Tensor:
        """Sample an EC embedding at a random granularity level."""
        if not ec_numbers or self.ec_lookup is None:
            return (
                self.ec_lookup.get_zero_vector()
                if self.ec_lookup is not None
                else torch.zeros(1024)
            )

        # Sample level 0-5
        level = random.choices(range(6), weights=self.ec_level_weights, k=1)[0]

        if level == 0:
            return self.ec_lookup.get_zero_vector()

        # Pick a random EC from available numbers
        ec = random.choice(ec_numbers)
        try:
            return self.ec_lookup.lookup(ec, level).clone()
        except KeyError:
            return self.ec_lookup.get_zero_vector()

    def _apply_motif_masking(
        self, aatype: Tensor, n_res: int
    ) -> tuple[Tensor, Tensor]:
        """Apply motif selection and aatype masking.

        Returns:
            (masked_aatype, motif_mask) where motif_mask[i]=True means
            residue i is a motif residue (keeps its real aatype).
        """
        aatype = aatype.clone()
        motif_mask = torch.zeros(n_res, dtype=torch.bool)

        if random.random() < self.motif_prob:
            # Enable motif conditioning
            n_motif = random.randint(
                self.motif_min_res, min(self.motif_max_res, n_res)
            )
            motif_indices = random.sample(range(n_res), n_motif)
            motif_mask[motif_indices] = True
            # Mask scaffold positions
            aatype[~motif_mask] = MASK_TOKEN
        else:
            # Unconditional generation: mask all
            aatype[:] = MASK_TOKEN

        return aatype, motif_mask


# ---------------------------------------------------------------------------
# 3. Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length samples to the max length in the batch.

    Adds ``seq_mask`` (BoolTensor[B, max_N]) -- True for real residues.
    """
    B = len(batch)
    max_N = max(s["seq_len"].item() for s in batch)

    # Pre-allocate padded tensors
    coords = torch.zeros(B, max_N, 14, 3)
    atom_mask = torch.zeros(B, max_N, 14, dtype=torch.bool)
    aatype = torch.full((B, max_N), MASK_TOKEN, dtype=torch.long)
    residue_index = torch.zeros(B, max_N, dtype=torch.long)
    motif_mask = torch.zeros(B, max_N, dtype=torch.bool)
    seq_mask = torch.zeros(B, max_N, dtype=torch.bool)

    ec_embeds = []
    seq_lens = []

    for i, s in enumerate(batch):
        n = s["seq_len"].item()
        coords[i, :n] = s["coords"]
        atom_mask[i, :n] = s["atom_mask"]
        aatype[i, :n] = s["aatype"]
        residue_index[i, :n] = s["residue_index"]
        motif_mask[i, :n] = s["motif_mask"]
        seq_mask[i, :n] = True
        ec_embeds.append(s["ec_embed"])
        seq_lens.append(s["seq_len"])

    return {
        "coords": coords,                         # [B, max_N, 14, 3]
        "atom_mask": atom_mask,                    # [B, max_N, 14]
        "aatype": aatype,                          # [B, max_N]
        "residue_index": residue_index,            # [B, max_N]
        "motif_mask": motif_mask,                  # [B, max_N]
        "seq_mask": seq_mask,                      # [B, max_N]
        "ec_embed": torch.stack(ec_embeds),        # [B, d_embed]
        "seq_len": torch.stack(seq_lens),          # [B]
    }


# ---------------------------------------------------------------------------
# 4. Bucket Batch Sampler
# ---------------------------------------------------------------------------


class BucketBatchSampler(Sampler[list[int]]):
    """Length-bucketed batch sampler for efficient padding.

    Sorts indices by sequence length, groups into buckets of size
    ``batch_size * bucket_size``, shuffles within buckets, forms batches,
    then shuffles batches across buckets.

    Args:
        lengths: Sequence length for each dataset index.
        batch_size: Number of samples per batch.
        bucket_size: Number of batches per bucket (pool).
        drop_last: Drop the last incomplete batch.
        shuffle: Whether to shuffle within and across buckets.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size: int = 5,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs."""
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # Sort indices by length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        # Split into pools of size batch_size * bucket_size
        pool_size = self.batch_size * self.bucket_size
        batches = []

        for pool_start in range(0, len(sorted_indices), pool_size):
            pool = sorted_indices[pool_start : pool_start + pool_size]
            if self.shuffle:
                rng.shuffle(pool)

            # Form batches from this pool
            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start : batch_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        # Shuffle batches across pools
        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)


# ---------------------------------------------------------------------------
# 5. Build DataLoader
# ---------------------------------------------------------------------------


def build_dataloader(
    data_dir: str | Path,
    split_ids: list[str] | None = None,
    ec_vectors_path: str | Path | None = None,
    batch_size: int = 8,
    motif_prob: float = 0.5,
    num_workers: int = 4,
    bucket_size: int = 5,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    motif_min_res: int = 2,
    motif_max_res: int = 8,
    ec_level_weights: tuple[float, ...] = (0.10, 0.10, 0.15, 0.15, 0.20, 0.30),
) -> DataLoader:
    """Convenience builder for a training DataLoader with bucket sampling.

    Args:
        data_dir: Path to directory with .pt files.
        split_ids: Subset of IDs to use (None = all).
        ec_vectors_path: Path to ec_vectors .pt file. None for AFDB pretrain.
        batch_size: Samples per batch.
        motif_prob: Probability of motif conditioning.
        num_workers: DataLoader worker processes.
        bucket_size: Batches per bucket for length sorting.
        drop_last: Drop last incomplete batch.
        shuffle: Enable bucket shuffling.
        seed: Random seed.
        motif_min_res: Min motif residues.
        motif_max_res: Max motif residues.
        ec_level_weights: Sampling weights for EC levels 0-5.

    Returns:
        A configured DataLoader.
    """
    ec_lookup = None
    if ec_vectors_path is not None:
        ec_lookup = ECEmbeddingLookup(ec_vectors_path)

    dataset = ProteinDataset(
        data_dir=data_dir,
        split_ids=split_ids,
        ec_lookup=ec_lookup,
        motif_prob=motif_prob,
        motif_min_res=motif_min_res,
        motif_max_res=motif_max_res,
        ec_level_weights=ec_level_weights,
    )

    sampler = BucketBatchSampler(
        lengths=[dataset.get_length(i) for i in range(len(dataset))],
        batch_size=batch_size,
        bucket_size=bucket_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
