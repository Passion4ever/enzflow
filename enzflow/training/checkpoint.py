"""Checkpoint save / load / cleanup utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str | Path,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: LRScheduler,
    config: dict,
    val_loss: float | None = None,
) -> None:
    """Save training checkpoint.

    Also writes ``config.json`` next to the checkpoint for easy inference
    loading (only once, on first save).

    *scaler* may be ``None`` when AMP is managed externally (e.g. Accelerate).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "val_loss": val_loss,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)

    # Write config.json once for convenience
    config_path = path.parent / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: LRScheduler,
    device: torch.device | str = "cpu",
) -> tuple[int, float]:
    """Load checkpoint and restore all state.

    *scaler* may be ``None`` when AMP is managed externally (e.g. Accelerate).

    Returns:
        (step, best_val_loss) from the checkpoint.
    """
    state = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    scheduler.load_state_dict(state["scheduler"])

    step = state["step"]
    val_loss = state.get("val_loss") or float("inf")

    return step, val_loss


def cleanup_checkpoints(
    ckpt_dir: str | Path,
    max_ckpts: int = 3,
    best_path: str = "best.pt",
) -> None:
    """Keep only the most recent *max_ckpts* step checkpoints + best.

    Step checkpoints are identified by the pattern ``step_XXXXX.pt``.
    """
    ckpt_dir = Path(ckpt_dir)
    pattern = re.compile(r"^step_(\d+)\.pt$")

    step_files: list[tuple[int, Path]] = []
    for f in ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            step_files.append((int(m.group(1)), f))

    # Sort by step number descending, keep most recent max_ckpts
    step_files.sort(key=lambda x: x[0], reverse=True)
    to_remove = step_files[max_ckpts:]

    best = ckpt_dir / best_path
    for _, f in to_remove:
        if f != best:
            f.unlink(missing_ok=True)
