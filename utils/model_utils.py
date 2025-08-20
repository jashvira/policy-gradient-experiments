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

    # For decoder-only architectures, ensure left padding to avoid HF warnings
    # and to keep prompt tokens aligned at the right side during generation.
    try:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    except Exception:
        # Some tokenizers may not expose these attributes; ignore safely
        pass

    # Ensure a valid pad token id is set; fall back to eos if needed
    if getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    # Align model config with tokenizer pad token id when possible
    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    return model, tokenizer, device


