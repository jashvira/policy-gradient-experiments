from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union


@dataclass
class SFTTrainConfig:
    # Data and IO
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    sft_data_path: Optional[str] = None  # defaults to <repo_root>/MATH/sft.jsonl if None
    save_dir: str = "./sft_output"
    val_log_dir: str = "logs/val_generations"
    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"

    # Batching and sampling
    batch_size: int = 4
    val_samples: int = 32
    unique_train_examples: Union[int, str] = "all"  # int or "all"

    # Optimization
    gradient_accumulation_steps: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_fused: Optional[bool] = None  # auto if None
    grad_clip: float = 1.0

    # Scheduler
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = None
    lr_scheduler: str = "linear"  # or "cosine"

    # Evaluation/generation
    val_every_steps: int = 100
    max_new_tokens: int = 128
    val_temperature: float = 0.7
    val_top_p: float = 0.95

    # Logging
    project: str = "sft-math"
    run_name: str = "sft-run"
    wandb_entity: Optional[str] = None
    seed: int = 42

    # vLLM
    vllm_gpu_mem_utilization: float = 0.85
    model_name_for_vllm: Optional[str] = None  # defaults to model_name if None

    @classmethod
    def _load_mapping(path: Path) -> dict:
        """Load a dict from JSON or YAML based on file suffix."""
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML is required to load YAML configs. Please add pyyaml to dependencies.") from e
            return yaml.safe_load(path.read_text()) or {}
        else:
            return json.loads(path.read_text())

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "SFTTrainConfig":
        """Load config (YAML or JSON) with simple inheritance via an 'extends' field.

        If the file contains {extends: base.yaml}, we load the base first (relative
        to the current file), then overlay the child's keys.
        """
        path = Path(path)
        data = cls._load_mapping(path)

        parent_data: dict = {}
        extends = data.get("extends") or data.get("$extends")
        if extends:
            parent_path = (path.parent / extends).resolve()
            parent_data = cls._load_mapping(parent_path)

        merged = {**parent_data, **data}
        merged.pop("extends", None)
        merged.pop("$extends", None)

        cfg = cls()
        for key, value in merged.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve(self, repo_root: Optional[Path] = None) -> "SFTTrainConfig":
        """Fill dependent defaults after construction."""
        if self.model_name_for_vllm is None:
            self.model_name_for_vllm = self.model_name
        if self.sft_data_path is None and repo_root is not None:
            self.sft_data_path = str((repo_root / "MATH" / "sft.jsonl").resolve())
        return self


