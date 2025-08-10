from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


@dataclass
class SFTTrainConfig:
    # Data and IO
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    sft_data_path: Optional[str] = None  # defaults to <repo_root>/MATH/sft.jsonl if None
    save_dir: str = "./sft_output"
    save_at_end: bool = False
    val_log_dir: str = "logs/val_generations"
    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"

    # Batching and sampling
    batch_size: int = 64
    val_samples: Union[int, str] = "all"
    unique_train_examples: Union[int, str] = "all"  # int or "all"
    num_epochs: int = 10

    # Optimization
    gradient_accumulation_steps: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_fused: Optional[bool] = None  # auto if None
    grad_clip: float = 1.0

    # Scheduler
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = 0.03
    lr_scheduler: str = "cosine"  # or "linear"

    # Evaluation/generation
    val_every_steps: int = 100
    max_new_tokens: int = 1024
    val_temperature: float = 0.0
    val_top_p: float = 0.95
    eval_before_training: bool = True

    # Logging
    project: str = "sft-math"
    run_name: str = "sft-run"
    wandb_entity: Optional[str] = None
    seed: int = 42

    # vLLM
    vllm_gpu_mem_utilization: float = 0.85
    model_name_for_vllm: Optional[str] = None  # defaults to model_name if None

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "SFTTrainConfig":
        """Load config strictly from a YAML file into the dataclass.

        - Only `.yaml`/`.yml` files are supported
        - Unknown keys raise an error (to avoid silent typos)
        - Missing keys fall back to dataclass defaults
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Only YAML config files are supported. Got: {path}")

        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required to load YAML configs. Please add pyyaml to dependencies.") from e

        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")

        # Construct with kwargs to ensure unknown keys raise immediately
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve(self, repo_root: Optional[Path] = None) -> "SFTTrainConfig":
        """Fill dependent defaults after construction."""
        if self.model_name_for_vllm is None:
            self.model_name_for_vllm = self.model_name
        if self.sft_data_path is None and repo_root is not None:
            self.sft_data_path = str((repo_root / "MATH" / "sft.jsonl").resolve())
        return self


