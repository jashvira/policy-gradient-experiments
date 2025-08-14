from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import sys
# Ensure repository root is on sys.path for absolute imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.config_base import BaseTrainConfig


@dataclass
class EITrainConfig(BaseTrainConfig):
    # Override model for Expert Iteration (Base model, not Instruct)
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    
    # Expert Iteration specific settings
    n_ei_steps: int = 5
    G: int = 8  # number of rollouts per question
    batch_size_per_ei_step: int = 1024  # questions sampled per EI step
    
    # Generation settings for rollouts
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_min_tokens: int = 4  # prevent empty generations
    sampling_max_tokens: int = 1024
    
    # SFT settings per EI step (inherit most from base, override where needed)
    sft_epochs_per_step: int = 2  # epochs to run SFT in each EI step
    
    # Override logging settings
    project: str = "ei-math"
    run_name: str = "ei-run"
    
    # EI-specific settings
    save_checkpoints: bool = True
    checkpoint_dir: str = "./ei_checkpoints"
    
    # Validation settings (inherit from base but can override)
    val_samples: Union[int, str] = 500  # smaller for faster validation during EI