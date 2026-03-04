"""Learning rate scheduler: linear warmup + cosine decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    """Linear warmup from 0 to base lr, then cosine decay to 0.

    Args:
        optimizer: Optimizer whose lr to schedule.
        warmup_steps: Steps for linear warmup phase.
        max_steps: Total training steps.

    Returns:
        LambdaLR scheduler (call .step() every training step).
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
