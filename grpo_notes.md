# GRPO Generation & Forward Pass Mechanics

## The Sequence

1. **Take 1 prompt**: "Janet has 16 eggs..."
2. **Generate 8 different answers** to that same question
3. **Reward functions score all 8 answers**
4. **GRPO compares** which answers got higher rewards
5. **Repeat with gradient accumulation** over 8 such prompt-sets

## Parameters
- `per_device_train_batch_size=1` → 1 prompt at a time
- `num_generations=8` → 8 completions per prompt
- `gradient_accumulation_steps=8` → accumulate over 8 prompts before weight update

## Memory Logic
- "Batch size of 1" = 1 prompt processed per forward pass
- That 1 prompt spawns 8 generations for reward comparison
- Different from regular training where 1 sample = 1 input/output pair