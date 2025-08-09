from __future__ import annotations

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def build_adamw(
    params,
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool | None = None,
) -> Optimizer:
    adamw_kwargs = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if fused is None:
        fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(params, fused=fused, **adamw_kwargs)
    except TypeError:
        return torch.optim.AdamW(params, **adamw_kwargs)


def build_warmup_then_scheduler(
    optimizer: Optimizer,
    total_update_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float | None = None,
    after: str = "linear",  # or "cosine"
):
    if warmup_ratio is not None and warmup_steps == 0:
        warmup_steps = int(warmup_ratio * total_update_steps)

    remain_steps = max(0, total_update_steps - warmup_steps)
    schedulers = []
    milestones = []
    if warmup_steps > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=warmup_steps))
        milestones.append(warmup_steps)
    if remain_steps > 0:
        if after == "cosine":
            tail = CosineAnnealingLR(optimizer, T_max=remain_steps)
        else:
            tail = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=remain_steps)
        schedulers.append(tail)
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


