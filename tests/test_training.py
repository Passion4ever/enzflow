"""Tests for enzflow.training modules."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

from enzflow.training.checkpoint import (
    cleanup_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from enzflow.training.flow_matching import rectified_flow_loss, sample_t
from enzflow.training.scheduler import get_cosine_schedule_with_warmup

# ---- sample_t ----


class TestSampleT:
    def test_shape(self):
        t = sample_t(8, torch.device("cpu"))
        assert t.shape == (8,)

    def test_range(self):
        t = sample_t(10000, torch.device("cpu"))
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_device(self):
        t = sample_t(4, torch.device("cpu"))
        assert t.device.type == "cpu"


# ---- rectified_flow_loss ----


class DummyModel(nn.Module):
    """Minimal model that returns zeros for testing."""

    def forward(self, x_t, t, atom_mask, aatype, residue_index,
                ec_embed, motif_mask, seq_mask):
        return torch.zeros_like(x_t)


class TestRectifiedFlowLoss:
    @pytest.fixture()
    def batch(self):
        B, N = 2, 10
        return {
            "coords": torch.randn(B, N, 14, 3),
            "atom_mask": torch.ones(B, N, 14, dtype=torch.bool),
            "aatype": torch.zeros(B, N, dtype=torch.long),
            "residue_index": torch.arange(N).unsqueeze(0).expand(B, -1),
            "motif_mask": torch.zeros(B, N, dtype=torch.bool),
            "seq_mask": torch.ones(B, N, dtype=torch.bool),
            "ec_embed": torch.zeros(B, 1024),
        }

    def test_returns_loss_and_vmag(self, batch):
        model = DummyModel()
        loss, v_mag = rectified_flow_loss(model, batch, torch.device("cpu"))
        assert loss.shape == ()
        assert loss.item() > 0  # nonzero since model returns 0 but target != 0
        assert isinstance(v_mag, float)

    def test_loss_is_finite(self, batch):
        model = DummyModel()
        loss, _ = rectified_flow_loss(model, batch, torch.device("cpu"))
        assert torch.isfinite(loss)

    def test_partial_mask(self, batch):
        # Mask out half the atoms
        batch["atom_mask"][:, :, 7:] = False
        model = DummyModel()
        loss, _ = rectified_flow_loss(model, batch, torch.device("cpu"))
        assert torch.isfinite(loss)


# ---- scheduler ----


class TestCosineSchedule:
    def test_warmup_phase(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps=100, max_steps=1000)

        # Step 0: lr = 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-7)

        # Step 50: lr = 0.05 (half of warmup)
        for _ in range(50):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05, abs=1e-4)

    def test_peak_lr(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps=100, max_steps=1000)

        for _ in range(100):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-5)

    def test_cosine_decay(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps=100, max_steps=1000)

        # Go to end
        for _ in range(1000):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-5)

    def test_midpoint(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps=0, max_steps=1000)

        for _ in range(500):
            sched.step()
        expected = 0.05 * (1 + math.cos(math.pi * 0.5))
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-4)


# ---- checkpoint ----


class TestCheckpoint:
    @pytest.fixture()
    def setup(self, tmp_path):
        model = nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        sched = get_cosine_schedule_with_warmup(opt, 10, 100)
        config = {"d_token": 256}
        return model, opt, scaler, sched, config, tmp_path

    def test_save_and_load(self, setup):
        model, opt, scaler, sched, config, tmp_path = setup
        ckpt_path = tmp_path / "test.pt"

        save_checkpoint(ckpt_path, 42, model, opt, scaler, sched, config, val_loss=1.23)
        assert ckpt_path.exists()

        # config.json should be created
        assert (tmp_path / "config.json").exists()

        # Load into fresh model
        model2 = nn.Linear(4, 4)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        scaler2 = torch.amp.GradScaler("cpu", enabled=False)
        sched2 = get_cosine_schedule_with_warmup(opt2, 10, 100)

        step, val_loss = load_checkpoint(
            ckpt_path, model2, opt2, scaler2, sched2, torch.device("cpu")
        )
        assert step == 42
        assert val_loss == pytest.approx(1.23)

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_save_and_load_no_scaler(self, setup):
        model, opt, _scaler, sched, config, tmp_path = setup
        ckpt_path = tmp_path / "no_scaler.pt"

        save_checkpoint(ckpt_path, 10, model, opt, None, sched, config)
        assert ckpt_path.exists()

        model2 = nn.Linear(4, 4)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        sched2 = get_cosine_schedule_with_warmup(opt2, 10, 100)

        step, _ = load_checkpoint(ckpt_path, model2, opt2, None, sched2, "cpu")
        assert step == 10
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_config_json_content(self, setup):
        model, opt, scaler, sched, config, tmp_path = setup
        save_checkpoint(tmp_path / "a.pt", 1, model, opt, scaler, sched, config)

        with open(tmp_path / "config.json") as f:
            loaded = json.load(f)
        assert loaded == config


class TestCleanupCheckpoints:
    def test_keeps_recent(self, tmp_path):
        # Create fake checkpoints
        for step in [1000, 2000, 3000, 4000, 5000]:
            (tmp_path / f"step_{step:06d}.pt").touch()
        (tmp_path / "best.pt").touch()

        cleanup_checkpoints(tmp_path, max_ckpts=3)

        remaining = sorted(f.name for f in tmp_path.iterdir())
        assert "best.pt" in remaining
        assert "step_005000.pt" in remaining
        assert "step_004000.pt" in remaining
        assert "step_003000.pt" in remaining
        assert "step_002000.pt" not in remaining
        assert "step_001000.pt" not in remaining

    def test_preserves_best(self, tmp_path):
        for step in [1000, 2000]:
            (tmp_path / f"step_{step:06d}.pt").touch()
        (tmp_path / "best.pt").touch()

        cleanup_checkpoints(tmp_path, max_ckpts=1)

        remaining = [f.name for f in tmp_path.iterdir()]
        assert "best.pt" in remaining
        assert "step_002000.pt" in remaining
        assert "step_001000.pt" not in remaining

    def test_ignores_non_step_files(self, tmp_path):
        (tmp_path / "step_001000.pt").touch()
        (tmp_path / "config.json").touch()
        (tmp_path / "notes.txt").touch()

        cleanup_checkpoints(tmp_path, max_ckpts=1)

        remaining = [f.name for f in tmp_path.iterdir()]
        assert "config.json" in remaining
        assert "notes.txt" in remaining
