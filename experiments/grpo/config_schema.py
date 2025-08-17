from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Literal
import sys

# Ensure repository root is on sys.path for absolute imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.config_base import BaseTrainConfig


@dataclass
class GRPOTrainConfig(BaseTrainConfig):
    """Configuration for Group Relative Policy Optimisation (GRPO) training."""
    
    # GRPO-specific training parameters
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256  # On-policy: equals rollout_batch_size
    epochs_per_rollout_batch: int = 1  # On-policy: single epoch per rollout
    
    # Generation/sampling parameters
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # Disallow empty string responses
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0
    
    # Loss function configuration
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    advantage_eps: float = 1e-6
    cliprange: float = 0.2  # For grpo_clip loss type
    
    # Override base config with GRPO-appropriate values
    learning_rate: float = 1e-5  # Lower learning rate for RL
    lr: float = 1e-5  # Alias for learning_rate
    adam_beta2: float = 0.95  # GRPO-specific beta2 value
    gradient_accumulation_steps: int = 128  # Microbatch size = 2 for H100
    gpu_memory_utilization: float = 0.85
    vllm_gpu_mem_utilization: float = 0.85  # Override base
    
    # Data and validation
    data_path: Optional[str] = None  # Defaults to MATH dataset if None
    val_every_grpo_steps: int = 10  # Evaluate every N GRPO steps
    val_rollout_batch_size: int = 64  # Smaller batch for validation
    val_samples: int = 1024  # Number of validation samples for robust evaluation
    save_every_grpo_steps: int = 50  # Save checkpoint every N steps
    
    # Model and checkpoint settings
    save_dir: str = "./grpo_output"
    save_at_end: bool = True
    val_log_dir: str = "logs/grpo_val_generations"
    
    # Override project name for GRPO
    project: str = "grpo-math"
    run_name: str = "grpo-run"
    
    def resolve(self, repo_root: Optional[Path] = None) -> "GRPOTrainConfig":
        """Fill GRPO-specific defaults after construction."""
        # Set repo_root for internal use
        if repo_root is None:
            repo_root = _REPO_ROOT
        self.repo_root = repo_root
        
        # Call parent resolve first
        super().resolve(repo_root)
        
        # Set lr alias if not explicitly provided
        if hasattr(self, 'lr') and self.lr != self.learning_rate:
            self.learning_rate = self.lr
        
        # Default data path to MATH dataset
        if self.data_path is None:
            self.data_path = str(self.repo_root / "MATH" / "train.jsonl")
        
        # Validate hyperparameter constraints
        self._validate_hyperparameters()
        
        return self
    
    def _validate_hyperparameters(self):
        """Validate GRPO hyperparameter constraints."""
        # Sanity check asserts from the specification
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        
        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        
        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        
        # Compute derived values for reference
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size
        
        print(f"GRPO Config derived values:")
        print(f"  micro_train_batch_size: {self.micro_train_batch_size}")
        print(f"  n_prompts_per_rollout_batch: {self.n_prompts_per_rollout_batch}")
        print(f"  n_microbatches_per_rollout_batch: {self.n_microbatches_per_rollout_batch}")