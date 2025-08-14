from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


@dataclass
class BaseTrainConfig:
    """Base configuration class shared between SFT and EI training."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"
    
    # Batching and optimization
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_fused: Optional[bool] = None
    grad_clip: float = 1.0
    
    # Scheduler
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = 0.0
    lr_scheduler: str = "cosine"
    
    # Validation/Evaluation
    val_every_steps: int = 5
    val_samples: Union[int, str] = "all"
    val_temperature: float = 0.0
    val_top_p: float = 1.0
    max_new_tokens: int = 1024
    
    # Logging
    project: str = "math-experiments"
    run_name: str = "experiment-run"
    wandb_entity: Optional[str] = None
    seed: int = 42
    
    # vLLM
    vllm_gpu_mem_utilization: float = 0.85
    
    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "BaseTrainConfig":
        """Load config from YAML file."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Only YAML config files are supported. Got: {path}")
        
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required to load YAML configs.") from e
        
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def resolve(self, repo_root: Optional[Path] = None) -> "BaseTrainConfig":
        """Fill dependent defaults after construction."""
        if repo_root is not None and not Path(self.model_name).exists():
            # Check if model is in .models directory
            local_model_path = repo_root / ".models" / Path(self.model_name).name
            if local_model_path.exists():
                self.model_name = str(local_model_path)
        return self