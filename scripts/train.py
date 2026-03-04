"""Full training script for enzflow all-atom protein generation.

GPU count is controlled entirely by CUDA_VISIBLE_DEVICES:

    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py

    # 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py

    # All GPUs
    python scripts/train.py

    # With wandb logging
    python scripts/train.py --wandb

    # Quick test
    python scripts/train.py --max_steps 200 --log_every 10

    # Resume
    python scripts/train.py --resume checkpoints/step_005000.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from enzflow.data.dataset import build_dataloader
from enzflow.model import AllAtomFlowModel
from enzflow.training import (
    cleanup_checkpoints,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    rectified_flow_loss,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train enzflow model")

    # Model
    p.add_argument("--d_token", type=int, default=512)
    p.add_argument("--d_pair", type=int, default=128)
    p.add_argument("--d_atom", type=int, default=128)
    p.add_argument("--d_cond", type=int, default=512)
    p.add_argument("--n_trunk", type=int, default=12)
    p.add_argument("--n_atom_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=8)

    # Optimization
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--data_dir", type=str, default="data/processed/enzymes")
    p.add_argument("--ec_vectors_path", type=str, default="weights/ec_vectors_train.pt")
    p.add_argument("--num_workers", type=int, default=4)

    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="enzflow")
    p.add_argument("--wandb_run", type=str, default=None, help="wandb run name")

    # Saving
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--max_ckpts", type=int, default=3)

    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")

    return p.parse_args()


def infinite_loader(loader, batch_sampler):
    """Yield batches forever, cycling through epochs.

    *batch_sampler* is the **original** (unwrapped) BucketBatchSampler so
    that ``set_epoch`` still works after Accelerate wraps the DataLoader.
    """
    epoch = 0
    while True:
        batch_sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def main():
    args = parse_args()
    mixed_precision = "bf16" if (args.bf16 and not args.no_bf16) else "no"

    # --- Accelerator (handles DDP + AMP + optional wandb) ---
    log_with = "wandb" if args.wandb else None
    accelerator = Accelerator(mixed_precision=mixed_precision, log_with=log_with)
    set_seed(args.seed)

    accelerator.print(
        f"Processes: {accelerator.num_processes}, "
        f"device: {accelerator.device}, mixed_precision: {mixed_precision}"
    )

    # --- Model config ---
    model_cfg = dict(
        d_token=args.d_token,
        d_pair=args.d_pair,
        d_atom=args.d_atom,
        d_cond=args.d_cond,
        n_trunk=args.n_trunk,
        n_atom_layers=args.n_atom_layers,
        n_heads=args.n_heads,
        d_ec_input=1024,
    )
    # Full config for logging
    train_cfg = {
        **model_cfg,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "mixed_precision": mixed_precision,
        "seed": args.seed,
        "num_processes": accelerator.num_processes,
    }

    # --- Data (all .pt files in data_dir) ---
    accelerator.print(f"Building dataloader from {args.data_dir}...")
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split_ids=None,  # use all files
        ec_vectors_path=args.ec_vectors_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        seed=args.seed,
    )
    # Keep reference before Accelerate wraps the dataloader
    orig_batch_sampler = train_loader.batch_sampler
    n_samples = len(train_loader.dataset)
    accelerator.print(f"Training samples: {n_samples}")

    # --- Model ---
    accelerator.print("Building model...")
    model = AllAtomFlowModel(**model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.max_steps
    )

    # --- Resume (before prepare, so state maps to raw model) ---
    start_step = 0
    if args.resume:
        accelerator.print(f"Resuming from {args.resume}...")
        start_step, _ = load_checkpoint(
            args.resume, model, optimizer, None, scheduler, "cpu"
        )
        accelerator.print(f"Resumed at step {start_step}")

    # --- Accelerate prepare (handles DDP + AMP) ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- Init wandb tracker (after prepare, only on main process) ---
    if args.wandb:
        accelerator.init_trackers(
            args.wandb_project,
            config=train_cfg,
            init_kwargs={"wandb": {"name": args.wandb_run}},
        )

    # --- Training loop ---
    ckpt_dir = Path(args.ckpt_dir)
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    train_iter = infinite_loader(train_loader, orig_batch_sampler)
    t0 = time.time()

    accelerator.print(f"\nStarting training from step {start_step} to {args.max_steps}")
    accelerator.print(
        f"{'Step':>8} | {'loss':>10} | {'grad':>8} | "
        f"{'lr':>10} | {'|v|':>8} | {'time':>8}"
    )
    accelerator.print("-" * 70)

    for step in range(start_step, args.max_steps):
        batch = next(train_iter)

        # Forward + loss (autocast managed by Accelerate)
        with accelerator.autocast():
            loss, v_mag = rectified_flow_loss(model, batch, accelerator.device)

        # Backward (Accelerate handles loss scaling + gradient sync)
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        optimizer.step()
        scheduler.step()

        # --- Logging ---
        if step % args.log_every == 0 or step == start_step:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            accelerator.print(
                f"Step {step:>6} | loss {loss.item():>9.2f} | "
                f"grad {grad_norm:>7.2f} | lr {lr:.1e} | "
                f"|v| {v_mag:>7.4f} | {elapsed:>7.1f}s"
            )
            if args.wandb:
                accelerator.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/lr": lr,
                        "train/v_mag": v_mag,
                    },
                    step=step,
                )

        # --- Checkpoint (main process saves, others wait) ---
        if step > 0 and step % args.ckpt_every == 0:
            if accelerator.is_main_process:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    ckpt_path, step, unwrapped, optimizer, None, scheduler,
                    model_cfg,
                )
                cleanup_checkpoints(ckpt_dir, args.max_ckpts)
            accelerator.wait_for_everyone()
            accelerator.print(f"Step {step:>6} | checkpoint step_{step:06d}.pt")

    # --- Final save ---
    if accelerator.is_main_process:
        final_path = ckpt_dir / f"step_{args.max_steps:06d}.pt"
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(
            final_path, args.max_steps, unwrapped, optimizer, None, scheduler,
            model_cfg,
        )
    accelerator.wait_for_everyone()

    elapsed = time.time() - t0
    accelerator.print(f"\nTraining complete. {args.max_steps} steps in {elapsed / 3600:.1f}h")
    accelerator.print(f"Final checkpoint: {ckpt_dir / f'step_{args.max_steps:06d}.pt'}")

    if args.wandb:
        accelerator.end_training()


def _auto_launch():
    """Re-launch via accelerate if multiple GPUs are visible."""
    if "LOCAL_RANK" in os.environ:
        return  # already inside a distributed launch
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return  # single GPU or CPU, run directly
    print(f"Detected {n_gpus} GPUs -- launching distributed training...")
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(n_gpus),
        *sys.argv,
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    _auto_launch()
    main()
