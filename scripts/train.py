"""Full training script for enzflow all-atom protein generation.

Usage:
    # YAML config (recommended)
    python scripts/train.py --config configs/pretrain_afdb.yaml

    # Override specific params via CLI
    python scripts/train.py --config configs/pretrain_afdb.yaml --lr 3e-4 --max_steps 100

    # Quick test
    python scripts/train.py --config configs/pretrain_afdb.yaml \
        --max_steps 200 --log_every 10 --ckpt_every 100

    # Resume training
    python scripts/train.py --config configs/pretrain_afdb.yaml \
        --resume checkpoints/pretrain_afdb_v1/step_005000.pt

    # Multi-GPU (auto-detected via CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config configs/pretrain_afdb.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
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
from enzflow.training.notify import notify


def load_config(config_path: str | None, cli_overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides.

    CLI overrides take priority over YAML values.
    """
    # Defaults
    cfg = {
        "run_name": None,
        "model": {
            "d_token": 512,
            "d_pair": 128,
            "d_atom": 128,
            "d_cond": 512,
            "n_trunk": 12,
            "n_atom_layers": 3,
            "n_heads": 8,
        },
        "training": {
            "batch_size": 6,
            "max_steps": 200_000,
            "lr": 1e-4,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "bf16": True,
            "seed": 42,
        },
        "data": {
            "data_dir": "data/processed/enzymes",
            "ec_vectors_path": "weights/ec_vectors_train.pt",
            "num_workers": 4,
            "max_seq_len": 512,
        },
        "logging": {
            "log_every": 50,
            "ckpt_every": 5000,
            "max_ckpts": 3,
            "wandb": False,
            "wandb_project": "enzflow",
        },
        "resume": None,
    }

    # Load YAML
    if config_path:
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)
        # Merge: YAML overwrites defaults
        for key, val in yaml_cfg.items():
            if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(val)
            else:
                cfg[key] = val

    # Apply CLI overrides (flat keys mapped to nested config)
    _cli_map = {
        # model
        "d_token": ("model", "d_token"),
        "d_pair": ("model", "d_pair"),
        "d_atom": ("model", "d_atom"),
        "d_cond": ("model", "d_cond"),
        "n_trunk": ("model", "n_trunk"),
        "n_atom_layers": ("model", "n_atom_layers"),
        "n_heads": ("model", "n_heads"),
        # training
        "batch_size": ("training", "batch_size"),
        "max_steps": ("training", "max_steps"),
        "lr": ("training", "lr"),
        "warmup_steps": ("training", "warmup_steps"),
        "weight_decay": ("training", "weight_decay"),
        "grad_clip": ("training", "grad_clip"),
        "bf16": ("training", "bf16"),
        "seed": ("training", "seed"),
        # data
        "data_dir": ("data", "data_dir"),
        "ec_vectors_path": ("data", "ec_vectors_path"),
        "num_workers": ("data", "num_workers"),
        # logging
        "log_every": ("logging", "log_every"),
        "ckpt_every": ("logging", "ckpt_every"),
        "max_ckpts": ("logging", "max_ckpts"),
        "wandb": ("logging", "wandb"),
        "wandb_project": ("logging", "wandb_project"),
        # top-level
        "run_name": None,
        "resume": None,
    }

    for cli_key, val in cli_overrides.items():
        if val is None:
            continue
        mapping = _cli_map.get(cli_key)
        if mapping is None:
            cfg[cli_key] = val
        else:
            section, param = mapping
            cfg[section][param] = val

    # Generate run directory name: {run_name}_{timestamp}
    # Skip if already set by _auto_launch (via --run_name CLI override)
    if "LOCAL_RANK" not in os.environ:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        if cfg["run_name"]:
            cfg["run_name"] = f"{cfg['run_name']}_{timestamp}"
        else:
            cfg["run_name"] = timestamp

    return cfg


def parse_args() -> tuple[str | None, dict]:
    """Parse CLI args. Returns (config_path, cli_overrides)."""
    p = argparse.ArgumentParser(description="Train enzflow model")

    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")

    # Model overrides
    p.add_argument("--d_token", type=int, default=None)
    p.add_argument("--d_pair", type=int, default=None)
    p.add_argument("--d_atom", type=int, default=None)
    p.add_argument("--d_cond", type=int, default=None)
    p.add_argument("--n_trunk", type=int, default=None)
    p.add_argument("--n_atom_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)

    # Training overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--bf16", action="store_true", default=None)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    # Data overrides
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ec_vectors_path", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)

    # Logging overrides
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=None)
    p.add_argument("--max_ckpts", type=int, default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--wandb_project", type=str, default=None)

    args = p.parse_args()

    # Handle --no_bf16
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "no_bf16")}
    if args.no_bf16:
        overrides["bf16"] = False

    return args.config, overrides


def setup_logging(run_dir: Path, accelerator: Accelerator) -> logging.Logger:
    """Set up file + console logging."""
    logger = logging.getLogger("enzflow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler (main process only)
    if accelerator.is_main_process:
        fh = logging.FileHandler(run_dir / "train.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console handler (main process only)
    if accelerator.is_main_process:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def infinite_loader(loader, batch_sampler):
    """Yield batches forever, cycling through epochs."""
    epoch = 0
    while True:
        batch_sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _load_dotenv(path: str = ".env") -> None:
    """Load .env file into os.environ if it exists."""
    p = Path(path)
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def main():
    _load_dotenv()
    config_path, cli_overrides = parse_args()
    cfg = load_config(config_path, cli_overrides)

    tcfg = cfg["training"]
    dcfg = cfg["data"]
    lcfg = cfg["logging"]
    mcfg = cfg["model"]

    mixed_precision = "bf16" if tcfg["bf16"] else "no"

    # --- Accelerator ---
    log_with = "wandb" if lcfg["wandb"] else None
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with=log_with,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(tcfg["seed"])

    # --- Run directory ---
    run_dir = Path("checkpoints") / cfg["run_name"]
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save config to run directory
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        # Copy original YAML config if provided
        if config_path:
            shutil.copy2(config_path, run_dir / "config_original.yaml")
    accelerator.wait_for_everyone()

    # --- Logging ---
    logger = setup_logging(run_dir, accelerator)

    logger.info(
        "Processes: %d, device: %s, mixed_precision: %s",
        accelerator.num_processes, accelerator.device, mixed_precision,
    )
    logger.info("Run: %s", cfg["run_name"])
    logger.info("Run dir: %s", run_dir)

    # --- Model config ---
    model_cfg = dict(**mcfg, d_ec_input=1024)

    # --- Data ---
    logger.info("Building dataloader from %s...", dcfg["data_dir"])
    train_loader = build_dataloader(
        data_dir=dcfg["data_dir"],
        split_ids=None,
        ec_vectors_path=dcfg["ec_vectors_path"],
        batch_size=tcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        drop_last=True,
        shuffle=True,
        seed=tcfg["seed"],
        max_seq_len=dcfg.get("max_seq_len", 512),
    )
    orig_batch_sampler = train_loader.batch_sampler
    n_samples = len(train_loader.dataset)
    logger.info("Training samples: %d", n_samples)

    # --- Model ---
    logger.info("Building model...")
    model = AllAtomFlowModel(**model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, tcfg["warmup_steps"], tcfg["max_steps"]
    )

    # --- Resume ---
    start_step = 0
    if cfg["resume"]:
        logger.info("Resuming from %s...", cfg["resume"])
        start_step, _ = load_checkpoint(
            cfg["resume"], model, optimizer, None, scheduler, "cpu"
        )
        logger.info("Resumed at step %d", start_step)

    # --- Accelerate prepare ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- wandb ---
    if lcfg["wandb"]:
        accelerator.init_trackers(
            lcfg["wandb_project"],
            config=cfg,
            init_kwargs={"wandb": {"name": cfg["run_name"]}},
        )

    # --- Training loop ---
    model.train()
    train_iter = infinite_loader(train_loader, orig_batch_sampler)
    t0 = time.time()
    cur_step = start_step

    logger.info("")
    logger.info("Starting training from step %d to %d", start_step, tcfg["max_steps"])
    logger.info(
        "%8s | %10s | %8s | %10s | %8s | %8s",
        "Step", "loss", "grad", "lr", "|v|", "time",
    )
    logger.info("-" * 70)

    try:
        for step in range(start_step, tcfg["max_steps"]):
            cur_step = step
            batch = next(train_iter)

            with accelerator.autocast():
                loss, v_mag = rectified_flow_loss(model, batch, accelerator.device)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), tcfg["grad_clip"]
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            optimizer.step()
            scheduler.step()

            # --- Logging ---
            if step % lcfg["log_every"] == 0 or step == start_step:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                logger.info(
                    "Step %6d | loss %9.2f | grad %7.2f | "
                    "lr %.1e | |v| %7.4f | %7.1fs",
                    step, loss.item(), grad_norm, lr, v_mag, elapsed,
                )
                if lcfg["wandb"]:
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/grad_norm": grad_norm,
                            "train/lr": lr,
                            "train/v_mag": v_mag,
                        },
                        step=step,
                    )

            # --- Checkpoint ---
            if step > 0 and step % lcfg["ckpt_every"] == 0:
                if accelerator.is_main_process:
                    ckpt_path = run_dir / f"step_{step:06d}.pt"
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint(
                        ckpt_path, step, unwrapped, optimizer, None,
                        scheduler, model_cfg,
                    )
                    cleanup_checkpoints(run_dir, lcfg["max_ckpts"])
                accelerator.wait_for_everyone()
                logger.info("Step %6d | checkpoint step_%06d.pt", step, step)

    except Exception as e:
        # --- Emergency checkpoint ---
        elapsed = time.time() - t0
        err_type = type(e).__name__
        logger.error("CRASHED at step %d: %s: %s", cur_step, err_type, e)
        if accelerator.is_main_process:
            try:
                ckpt_path = run_dir / f"emergency_step_{cur_step:06d}.pt"
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    ckpt_path, cur_step, unwrapped, optimizer, None,
                    scheduler, model_cfg,
                )
                logger.info("Emergency checkpoint saved: %s", ckpt_path)
            except Exception:
                logger.error("Failed to save emergency checkpoint")
        # --- Crash notification ---
        msg = (
            f"*{cfg['run_name']}* CRASHED\n"
            f"Step: {cur_step}, Time: {elapsed / 3600:.1f}h\n"
            f"Error: {err_type}: {e}"
        )
        if accelerator.is_main_process:
            notify(cfg, msg, subject=f"enzflow CRASHED: {err_type}")
        raise

    # --- Final save ---
    if accelerator.is_main_process:
        final_path = run_dir / f"step_{tcfg['max_steps']:06d}.pt"
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(
            final_path, tcfg["max_steps"], unwrapped, optimizer, None, scheduler,
            model_cfg,
        )
    accelerator.wait_for_everyone()

    elapsed = time.time() - t0
    logger.info("")
    logger.info(
        "Training complete. %d steps in %.1fh", tcfg["max_steps"], elapsed / 3600
    )
    logger.info(
        "Final checkpoint: %s", run_dir / f"step_{tcfg['max_steps']:06d}.pt"
    )

    # --- Completion notification ---
    if accelerator.is_main_process:
        msg = (
            f"*{cfg['run_name']}* DONE\n"
            f"Steps: {tcfg['max_steps']}, Time: {elapsed / 3600:.1f}h\n"
            f"Final loss: {loss.item():.2f}"
        )
        notify(cfg, msg, subject="enzflow training complete")

    if lcfg["wandb"]:
        accelerator.end_training()


def _auto_launch():
    """Re-launch via accelerate if CUDA_VISIBLE_DEVICES has multiple GPUs.

    Default behavior: single GPU (first visible card).
    Multi-GPU only when explicitly set, e.g. CUDA_VISIBLE_DEVICES=0,1,2.
    Pre-generates run_name so all processes share the same directory.
    """
    if "LOCAL_RANK" in os.environ:
        return  # already inside a distributed launch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return  # not set -> single GPU default
    n_gpus = len([x for x in cvd.split(",") if x.strip()])
    if n_gpus <= 1:
        return
    print(f"Detected {n_gpus} GPUs (CUDA_VISIBLE_DEVICES={cvd})"
          " -- launching distributed training...")
    # Pre-generate a shared run_name so all processes use the same dir
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    extra_args = []
    if "--run_name" not in sys.argv:
        # Read run_name from YAML config if provided
        base_name = None
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                with open(sys.argv[i + 1]) as f:
                    ycfg = yaml.safe_load(f)
                base_name = ycfg.get("run_name")
                break
        if base_name:
            extra_args = ["--run_name", f"{base_name}_{timestamp}"]
        else:
            extra_args = ["--run_name", timestamp]
    # Read bf16 setting from config
    mixed_precision = "bf16"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            with open(sys.argv[i + 1]) as f:
                ycfg = yaml.safe_load(f)
            if not ycfg.get("training", {}).get("bf16", True):
                mixed_precision = "no"
            break
    if "--no_bf16" in sys.argv:
        mixed_precision = "no"

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(n_gpus),
        "--num_machines", "1",
        "--mixed_precision", mixed_precision,
        "--dynamo_backend", "no",
        *sys.argv, *extra_args,
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    _auto_launch()
    main()
