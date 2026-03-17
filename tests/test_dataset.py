"""Tests for enzflow.data.dataset module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from enzflow.data.dataset import (
    MASK_TOKEN,
    BucketBatchSampler,
    ECEmbeddingLookup,
    ProteinDataset,
    build_dataloader,
    collate_fn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_EMBED = 64  # small embedding dim for tests


@pytest.fixture
def ec_vectors_path(tmp_path: Path) -> Path:
    """Create a synthetic ec_vectors .pt file."""
    vectors = {
        ("1.1.1.1", 1): torch.randn(D_EMBED),
        ("1.1.1.1", 2): torch.randn(D_EMBED),
        ("1.1.1.1", 3): torch.randn(D_EMBED),
        ("1.1.1.1", 4): torch.randn(D_EMBED),
        ("1.1.1.1", 5): torch.randn(D_EMBED),
        ("2.3.1.5", 1): torch.randn(D_EMBED),
        ("2.3.1.5", 2): torch.randn(D_EMBED),
        ("2.3.1.5", 3): torch.randn(D_EMBED),
        ("2.3.1.5", 4): torch.randn(D_EMBED),
        ("2.3.1.5", 5): torch.randn(D_EMBED),
    }
    path = tmp_path / "ec_vectors.pt"
    torch.save({"vectors": vectors, "d_embed": D_EMBED, "config": {}}, path)
    return path


def _make_sample(n_res: int, ec_numbers: list[str] | None = None, uid: str = "test") -> dict:
    """Create a synthetic protein sample dict matching .pt file format."""
    coords = torch.randn(n_res, 14, 3)
    atom_mask = torch.ones(n_res, 14, dtype=torch.bool)
    # GLY has only 4 atoms -- mask out side chain slots for last residue
    atom_mask[-1, 4:] = False
    aatype = torch.randint(0, 20, (n_res,))
    residue_index = torch.arange(1, n_res + 1, dtype=torch.long)
    sample = {
        "coords": coords,
        "atom_mask": atom_mask,
        "aatype": aatype,
        "residue_index": residue_index,
        "uniprot_id": uid,
    }
    if ec_numbers is not None:
        sample["ec_numbers"] = ec_numbers
    else:
        sample["ec_numbers"] = []
    return sample


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a directory with synthetic .pt files."""
    d = tmp_path / "proteins"
    d.mkdir()
    # 5 samples of varying lengths
    for i, n in enumerate([10, 20, 15, 30, 25]):
        ec = ["1.1.1.1"] if i % 2 == 0 else []
        sample = _make_sample(n, ec_numbers=ec, uid=f"prot_{i}")
        torch.save(sample, d / f"prot_{i}.pt")
    return d


@pytest.fixture
def data_dir_with_multi_ec(tmp_path: Path) -> Path:
    """Create a directory with proteins having multiple EC numbers."""
    d = tmp_path / "multi_ec"
    d.mkdir()
    sample = _make_sample(20, ec_numbers=["1.1.1.1", "2.3.1.5"], uid="multi")
    torch.save(sample, d / "multi.pt")
    return d


# ===========================================================================
# TestECEmbeddingLookup
# ===========================================================================


class TestECEmbeddingLookup:
    def test_load(self, ec_vectors_path):
        lookup = ECEmbeddingLookup(ec_vectors_path)
        assert lookup.d_embed == D_EMBED

    def test_d_embed(self, ec_vectors_path):
        lookup = ECEmbeddingLookup(ec_vectors_path)
        assert isinstance(lookup.d_embed, int)
        assert lookup.d_embed > 0

    def test_lookup(self, ec_vectors_path):
        lookup = ECEmbeddingLookup(ec_vectors_path)
        vec = lookup.lookup("1.1.1.1", 3)
        assert vec.shape == (D_EMBED,)
        assert vec.dtype == torch.float32

    def test_lookup_keyerror(self, ec_vectors_path):
        lookup = ECEmbeddingLookup(ec_vectors_path)
        with pytest.raises(KeyError):
            lookup.lookup("9.9.9.9", 1)

    def test_zero_vector(self, ec_vectors_path):
        lookup = ECEmbeddingLookup(ec_vectors_path)
        z = lookup.get_zero_vector()
        assert z.shape == (D_EMBED,)
        assert (z == 0).all()


# ===========================================================================
# TestProteinDataset
# ===========================================================================


class TestProteinDataset:
    def test_len(self, data_dir):
        ds = ProteinDataset(data_dir)
        assert len(ds) == 5

    def test_split_filter(self, data_dir):
        ds = ProteinDataset(data_dir, split_ids=["prot_0", "prot_2"])
        assert len(ds) == 2

    def test_split_filter_missing(self, data_dir):
        ds = ProteinDataset(data_dir, split_ids=["prot_0", "nonexistent"])
        assert len(ds) == 1

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            ProteinDataset(empty)

    def test_getitem_keys(self, data_dir):
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        expected_keys = {"coords", "atom_mask", "aatype", "residue_index",
                         "motif_mask", "ec_embed", "seq_len"}
        assert set(sample.keys()) == expected_keys

    def test_getitem_shapes(self, data_dir):
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        N = sample["seq_len"].item()
        assert sample["coords"].shape == (N, 14, 3)
        assert sample["atom_mask"].shape == (N, 14)
        assert sample["aatype"].shape == (N,)
        assert sample["residue_index"].shape == (N,)
        assert sample["motif_mask"].shape == (N,)
        assert sample["ec_embed"].ndim == 1

    def test_getitem_dtypes(self, data_dir):
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        assert sample["coords"].dtype == torch.float32
        assert sample["atom_mask"].dtype == torch.bool
        assert sample["aatype"].dtype == torch.long
        assert sample["residue_index"].dtype == torch.long
        assert sample["motif_mask"].dtype == torch.bool
        assert sample["seq_len"].dtype == torch.long

    def test_se3_randomness(self, data_dir):
        """Two calls should give different coords due to random SE(3)."""
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        c1 = ds[0]["coords"]
        c2 = ds[0]["coords"]
        assert not torch.allclose(c1, c2)

    def test_se3_distance_preservation(self, data_dir):
        """SE(3) + normalization: distances scale by 1/COORD_SCALE."""
        from enzflow.data.transforms import COORD_SCALE
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        # Load raw coords
        raw = torch.load(ds.paths[0], map_location="cpu", weights_only=False)
        raw_ca = raw["coords"][:, 1, :]  # [N, 3]
        raw_dist = torch.cdist(raw_ca.unsqueeze(0), raw_ca.unsqueeze(0)).squeeze(0)

        sample = ds[0]
        aug_ca = sample["coords"][:, 1, :]
        aug_dist = torch.cdist(aug_ca.unsqueeze(0), aug_ca.unsqueeze(0)).squeeze(0)

        assert torch.allclose(raw_dist / COORD_SCALE, aug_dist, atol=1e-4)

    def test_motif_prob_zero(self, data_dir):
        """motif_prob=0 => motif_mask all False, aatype unchanged."""
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        assert not sample["motif_mask"].any()
        # aatype should be real values (0-19), not masked
        assert (sample["aatype"] < MASK_TOKEN).all()

    def test_motif_prob_one(self, data_dir):
        """motif_prob=1 => some motif residues selected."""
        ds = ProteinDataset(data_dir, motif_prob=1.0)
        sample = ds[0]
        assert sample["motif_mask"].any()
        # All aatype should be real values (0-19)
        assert (sample["aatype"] < MASK_TOKEN).all()

    def test_motif_count_range(self, data_dir):
        """Motif residue count should be in [min, max]."""
        ds = ProteinDataset(data_dir, motif_prob=1.0, motif_min_res=2, motif_max_res=5)
        for _ in range(20):
            sample = ds[0]
            n_motif = sample["motif_mask"].sum().item()
            assert 2 <= n_motif <= 5

    def test_ec_shape_no_lookup(self, data_dir):
        """Without ec_lookup, ec_embed should be [1024] zeros."""
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        assert sample["ec_embed"].shape == (1024,)
        assert (sample["ec_embed"] == 0).all()

    def test_get_length(self, data_dir):
        ds = ProteinDataset(data_dir)
        for i in range(len(ds)):
            assert ds.get_length(i) == ds[i]["seq_len"].item()


# ===========================================================================
# TestECDropout
# ===========================================================================


class TestECDropout:
    def test_no_ec_zero_vector(self, data_dir, ec_vectors_path):
        """Proteins without EC numbers always get zero vectors."""
        lookup = ECEmbeddingLookup(ec_vectors_path)
        ds = ProteinDataset(data_dir, ec_lookup=lookup, motif_prob=0.0)
        # prot_1 and prot_3 have empty ec_numbers
        sample = ds[1]  # prot_1 -- no EC
        assert (sample["ec_embed"] == 0).all()

    def test_no_lookup_zero_vector(self, data_dir):
        """Without ec_lookup, always zero."""
        ds = ProteinDataset(data_dir, motif_prob=0.0)
        sample = ds[0]
        assert (sample["ec_embed"] == 0).all()

    def test_ec_sampling_distribution(self, data_dir, ec_vectors_path):
        """With level 0 weight high, most samples should get zero vector."""
        lookup = ECEmbeddingLookup(ec_vectors_path)
        ds = ProteinDataset(
            data_dir, ec_lookup=lookup, motif_prob=0.0,
            ec_level_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        # prot_0 has EC "1.1.1.1"
        n_zero = sum(1 for _ in range(50) if (ds[0]["ec_embed"] == 0).all())
        assert n_zero == 50  # all level 0 => all zero

    def test_multi_ec_protein(self, data_dir_with_multi_ec, ec_vectors_path):
        """Protein with multiple ECs should sample from them."""
        lookup = ECEmbeddingLookup(ec_vectors_path)
        ds = ProteinDataset(
            data_dir_with_multi_ec, ec_lookup=lookup, motif_prob=0.0,
            ec_level_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # always level 5
        )
        seen_vecs = set()
        for _ in range(50):
            vec = ds[0]["ec_embed"]
            seen_vecs.add(tuple(vec[:4].tolist()))  # use first 4 values as fingerprint
        # Should see at least 2 different vectors (from 2 different ECs)
        assert len(seen_vecs) >= 2


# ===========================================================================
# TestCollateFn
# ===========================================================================


class TestCollateFn:
    def _make_batch(self, lengths: list[int]) -> list[dict[str, torch.Tensor]]:
        """Create a batch of synthetic samples with given lengths."""
        batch = []
        for n in lengths:
            batch.append({
                "coords": torch.randn(n, 14, 3),
                "atom_mask": torch.ones(n, 14, dtype=torch.bool),
                "aatype": torch.randint(0, 20, (n,)),
                "residue_index": torch.arange(n, dtype=torch.long),
                "motif_mask": torch.zeros(n, dtype=torch.bool),
                "ec_embed": torch.randn(D_EMBED),
                "seq_len": torch.tensor(n, dtype=torch.long),
            })
        return batch

    def test_keys(self):
        batch = self._make_batch([5, 10])
        out = collate_fn(batch)
        expected = {"coords", "atom_mask", "aatype", "residue_index",
                    "motif_mask", "seq_mask", "ec_embed", "seq_len"}
        assert set(out.keys()) == expected

    def test_padding_shapes(self):
        batch = self._make_batch([5, 10, 7])
        out = collate_fn(batch)
        B, max_N = 3, 10
        assert out["coords"].shape == (B, max_N, 14, 3)
        assert out["atom_mask"].shape == (B, max_N, 14)
        assert out["aatype"].shape == (B, max_N)
        assert out["residue_index"].shape == (B, max_N)
        assert out["motif_mask"].shape == (B, max_N)
        assert out["seq_mask"].shape == (B, max_N)
        assert out["ec_embed"].shape == (B, D_EMBED)
        assert out["seq_len"].shape == (B,)

    def test_seq_mask(self):
        batch = self._make_batch([5, 10])
        out = collate_fn(batch)
        # First sample: 5 real, 5 padded
        assert out["seq_mask"][0, :5].all()
        assert not out["seq_mask"][0, 5:].any()
        # Second sample: all 10 real
        assert out["seq_mask"][1, :10].all()

    def test_padding_values(self):
        batch = self._make_batch([3, 6])
        out = collate_fn(batch)
        # Padded coords should be zero
        assert (out["coords"][0, 3:] == 0).all()
        # Padded atom_mask should be False
        assert not out["atom_mask"][0, 3:].any()
        # Padded aatype should be MASK_TOKEN
        assert (out["aatype"][0, 3:] == MASK_TOKEN).all()

    def test_ec_stack(self):
        batch = self._make_batch([5, 10])
        out = collate_fn(batch)
        # EC embeds should match originals
        assert torch.allclose(out["ec_embed"][0], batch[0]["ec_embed"])
        assert torch.allclose(out["ec_embed"][1], batch[1]["ec_embed"])

    def test_single_sample(self):
        batch = self._make_batch([8])
        out = collate_fn(batch)
        assert out["coords"].shape == (1, 8, 14, 3)
        assert out["seq_mask"][0].all()

    def test_motif_mask_preserved(self):
        batch = self._make_batch([5, 10])
        # Set some motif positions in first sample
        batch[0]["motif_mask"][1] = True
        batch[0]["motif_mask"][3] = True
        out = collate_fn(batch)
        assert out["motif_mask"][0, 1] and out["motif_mask"][0, 3]
        assert not out["motif_mask"][0, 0]
        # Padded area should be False
        assert not out["motif_mask"][0, 5:].any()


# ===========================================================================
# TestBucketBatchSampler
# ===========================================================================


class TestBucketBatchSampler:
    def test_full_coverage(self):
        """All indices should appear exactly once."""
        lengths = list(range(10, 110))  # 100 samples
        sampler = BucketBatchSampler(lengths, batch_size=8, shuffle=False)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(100))

    def test_batch_size(self):
        lengths = [50] * 32
        sampler = BucketBatchSampler(lengths, batch_size=8, drop_last=True)
        for batch in sampler:
            assert len(batch) == 8

    def test_bucket_similar_lengths(self):
        """Within a batch, lengths should be relatively similar."""
        lengths = list(range(50, 550))  # 500 samples, 50-549
        sampler = BucketBatchSampler(lengths, batch_size=8, bucket_size=5, shuffle=False)
        for batch in sampler:
            batch_lengths = [lengths[i] for i in batch]
            length_range = max(batch_lengths) - min(batch_lengths)
            # Within a pool of 40 (8*5), max range is 39
            assert length_range <= 40

    def test_epoch_different(self):
        """Different epochs should produce different orderings."""
        lengths = list(range(100))
        sampler = BucketBatchSampler(lengths, batch_size=4, shuffle=True)

        sampler.set_epoch(0)
        batches_0 = list(sampler)

        sampler.set_epoch(1)
        batches_1 = list(sampler)

        # At least some batches should differ
        assert batches_0 != batches_1

    def test_drop_last(self):
        lengths = [50] * 10
        sampler_keep = BucketBatchSampler(lengths, batch_size=3, drop_last=False)
        sampler_drop = BucketBatchSampler(lengths, batch_size=3, drop_last=True)
        assert len(list(sampler_keep)) == 4  # 3+3+3+1
        assert len(list(sampler_drop)) == 3  # 3+3+3

    def test_len(self):
        lengths = [50] * 20
        sampler = BucketBatchSampler(lengths, batch_size=8, drop_last=False)
        assert len(sampler) == 3  # ceil(20/8)
        sampler_drop = BucketBatchSampler(lengths, batch_size=8, drop_last=True)
        assert len(sampler_drop) == 2  # 20//8


# ===========================================================================
# TestDataLoaderIntegration
# ===========================================================================


class TestDataLoaderIntegration:
    def test_workers_zero(self, data_dir):
        dl = build_dataloader(data_dir, batch_size=2, num_workers=0,
                              motif_prob=0.0, shuffle=False)
        batch = next(iter(dl))
        assert "coords" in batch
        assert batch["coords"].ndim == 4
        assert batch["seq_mask"].any()

    def test_workers_two(self, data_dir):
        dl = build_dataloader(data_dir, batch_size=2, num_workers=2,
                              motif_prob=0.0, shuffle=False)
        batch = next(iter(dl))
        assert batch["coords"].shape[0] == 2

    def test_full_epoch(self, data_dir):
        """Iterate entire epoch, check all samples consumed."""
        dl = build_dataloader(data_dir, batch_size=2, num_workers=0,
                              motif_prob=0.0, drop_last=False, shuffle=False)
        total = 0
        for batch in dl:
            total += batch["seq_len"].shape[0]
        assert total == 5

    def test_with_ec_lookup(self, data_dir, ec_vectors_path):
        dl = build_dataloader(
            data_dir, batch_size=2, num_workers=0,
            ec_vectors_path=ec_vectors_path, motif_prob=0.0, shuffle=False,
        )
        batch = next(iter(dl))
        assert batch["ec_embed"].shape == (2, D_EMBED)
