from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model_and_tokenizer(
    model_name: str,
    train_device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """Load a HF causal LM and tokenizer, move to the requested device.

    Returns (model, tokenizer, device)
    """
    device = torch.device(train_device)

    # Load model directly to the target device for Flash Attention compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map={"": device}  # Force all layers to the specified device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


