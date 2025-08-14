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
class SFTTrainConfig(BaseTrainConfig):
    # SFT-specific Data and IO (preserving your magic numbers!)
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # Override base
    sft_data_path: Optional[str] = None  # defaults to <repo_root>/MATH/sft.jsonl if None
    save_dir: str = "./sft_output"
    save_at_end: bool = False
    val_log_dir: str = "logs/val_generations"
    
    # Override base config with your proven values
    batch_size: int = 64  # Your magic number (base has 16)
    val_samples: Union[int, str] = "all"  # Your setting (base has "all")
    unique_train_examples: Union[int, str] = "all"  # SFT-specific
    num_epochs: int = 20  # SFT-specific
    
    # Keep your proven optimization settings
    gradient_accumulation_steps: int = 8  # Your value (base has 8) 
    lr: float = 5e-5  # Your value (base has 5e-5)
    warmup_ratio: Optional[float] = 0.0  # Your value (base has 0.0)
    
    # Keep your evaluation settings
    val_every_steps: int = 5  # Your value (base has 5)
    eval_before_training: bool = False  # SFT-specific
    
    # Keep your logging settings
    project: str = "sft-math"  # Your value (override base)
    run_name: str = "sft-run"  # Your value (override base)
    
    # SFT-specific vLLM
    model_name_for_vllm: Optional[str] = None  # defaults to model_name if None

    def resolve(self, repo_root: Optional[Path] = None) -> "SFTTrainConfig":
        """Fill SFT-specific defaults after construction."""
        # Call parent resolve first
        super().resolve(repo_root)
        
        # SFT-specific resolution
        if self.model_name_for_vllm is None:
            self.model_name_for_vllm = self.model_name
        if self.sft_data_path is None and repo_root is not None:
            self.sft_data_path = str((repo_root / "MATH" / "sft.jsonl").resolve())
        return self


