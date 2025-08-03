#!/usr/bin/env python3
"""
GSM8K Evaluation Script
Fast, single-GPU evaluation of GRPO checkpoints on GSM8K test set.
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from tqdm import tqdm
from datetime import datetime
from vllm import SamplingParams

# Import shared utility functions
from utils import SYSTEM_PROMPT, extract_xml_answer, extract_hash_answer
from eval_utils import prepare_model_for_eval, load_model_smart, collate_fn




def evaluate_gsm8k(model, tokenizer, batch_size=32, wandb_project="gsm8k-eval", model_name="model"):
    """Standalone evaluation function that takes pre-loaded model and tokenizer."""
    print(f"Running GSM8K evaluation with batch_size={batch_size}")
    print(f"Model device: {next(model.parameters()).device}")

    # Create organized folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_folder = Path("evaluations") / f"{timestamp}_{model_name}_b{batch_size}"
    eval_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {eval_folder}")

    # Prepare model for eval if not already done
    model = prepare_model_for_eval(model)

    # Initialize WandB with better organization
    wandb.init(
        project=wandb_project,
        name=f"{model_name.replace('/', '-')}-{timestamp}",
        tags=[
            "evaluation",
            f"batch_size_{batch_size}",
            "gsm8k_test",
            model_name.split('/')[0] if '/' in model_name else "local"  # Model family
        ],
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "timestamp": timestamp,
            "eval_folder": str(eval_folder),
            "dataset": "gsm8k_test",
            "total_problems": 1319
        },
        settings=wandb.Settings(console="off")
    )

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Run evaluation
    print("=" * 50)
    results = evaluate_model(model, tokenizer, dataset, batch_size)
    print("=" * 50)

    # Log key metrics to WandB
    wandb.log({
        "accuracy": results["accuracy"],
        "throughput": results["total"] / results["eval_time"]
    })

    # Summary for run overview
    wandb.summary.update({
        "accuracy": results["accuracy"],
        "eval_time": results["eval_time"]
    })

    # Save results in organized folder
    results_df = pd.DataFrame({
        "question": results["questions"],
        "target": results["targets"],
        "prediction": results["predictions"],
        "is_correct": [p == t and t is not None for p, t in zip(results["predictions"], results["targets"])]
    })

    csv_path = eval_folder / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

    # Save summary
    summary_path = eval_folder / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"GSM8K Test Set Evaluation\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Correct: {results['correct']}/{results['total']}\n")
        f.write(f"Eval Time: {results['eval_time']:.2f}s\n")
        f.write(f"Throughput: {results['total'] / results['eval_time']:.2f} problems/sec\n")
    print(f"Summary saved to: {summary_path}")

    # Save evaluation metadata
    metadata_path = eval_folder / "metadata.json"
    import json
    metadata = {
        "model_name": model_name,
        "batch_size": batch_size,
        "timestamp": timestamp,
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "eval_time": results["eval_time"],
        "throughput": results["total"] / results["eval_time"]
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Upload files as WandB artifact
    artifact = wandb.Artifact("gsm8k_eval_results", type="evaluation")
    artifact.add_file(str(csv_path))
    artifact.add_file(str(summary_path))
    artifact.add_file(str(metadata_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"All files saved in: {eval_folder}")
    return results["accuracy"]


def prepare_dataset(tokenizer):
    """Load and preprocess GSM8K test set."""
    print("Loading GSM8K test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    def format_example(example):
        # Use same chat template as training
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ]
        prompt_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True
        )
        return {
            "prompt_ids": prompt_ids,
            "question": example["question"],
            "target": extract_hash_answer(example["answer"])
        }

    print("Preprocessing dataset...")
    dataset = dataset.map(format_example, num_proc=4)
    return dataset




def evaluate_model(model, tokenizer, dataset, batch_size=8, use_vllm=True):
    """Run batched inference and compute accuracy."""
    predictions = []
    targets = []
    questions = []

    print(f"Running inference on {len(dataset)} examples...")
    start_time = time.time()

    if use_vllm and hasattr(model, 'generate'):
        # Use vLLM for batch generation
        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.0,  # Greedy decoding
            stop=[tokenizer.eos_token]
        )

        # Prepare prompts for vLLM
        prompts = []
        for example in dataset:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ]
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            targets.append(example["target"])
            questions.append(example["question"])

        print(f"Generating {len(prompts)} responses with vLLM...")
        outputs = model.generate(prompts, sampling_params)

        for output in outputs:
            generated = output.outputs[0].text

            # Extract answer using XML parser
            try:
                pred_answer = extract_xml_answer(generated).replace(",", "").replace("$", "").strip()
            except:
                pred_answer = "PARSE_ERROR"

            predictions.append(pred_answer)

    else:
        # Fallback to standard transformers generation
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in tqdm(dataloader, desc="Evaluating batches"):

                # Generate responses
                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )

                # Extract generated text (remove prompt)
                prompt_lengths = batch["attention_mask"].sum(dim=1)
                for j, output in enumerate(outputs):
                    prompt_len = prompt_lengths[j].item()
                    generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)

                    # Extract answer using XML parser
                    try:
                        pred_answer = extract_xml_answer(generated).replace(",", "").replace("$", "").strip()
                    except:
                        pred_answer = "PARSE_ERROR"

                    predictions.append(pred_answer)
                    targets.append(batch["targets"][j])
                    questions.append(batch["questions"][j])

    eval_time = time.time() - start_time

    # Compute accuracy
    correct = sum(1 for p, t in zip(predictions, targets) if p == t and t is not None)
    total = sum(1 for t in targets if t is not None)
    accuracy = correct / total if total > 0 else 0.0

    print(f"Evaluation completed in {eval_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "eval_time": eval_time,
        "predictions": predictions,
        "targets": targets,
        "questions": questions
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K test accuracy")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., Qwen/Qwen2.5-1.5B-Instruct) or checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--wandb_project", type=str, default="gsm8k-eval",
                       help="WandB project name")
    parser.add_argument("--no-vllm", action="store_true", 
                       help="Disable vLLM and use standard transformers inference")

    args = parser.parse_args()

    use_vllm = not args.no_vllm  # Default True, False if --no-vllm
    
    print(f"Evaluating model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using vLLM: {use_vllm}")

    # Load model and determine tokenizer
    model, tokenizer_name = load_model_smart(args.model, use_vllm=use_vllm)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize WandB
    model_name = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "-")

    wandb.init(
        project=args.wandb_project,
        name=f"eval-{model_name}",
        config={
            "model": args.model,
            "batch_size": args.batch_size,
            "model_name": model_name
        }
    )

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Run evaluation
    print("=" * 50)
    results = evaluate_model(model, tokenizer, dataset, args.batch_size, use_vllm=use_vllm)
    print("=" * 50)

    # Log to WandB
    wandb.log({
        "gsm8k_accuracy": results["accuracy"],
        "gsm8k_correct": results["correct"],
        "gsm8k_total": results["total"],
        "eval_time_seconds": results["eval_time"],
        "throughput_problems_per_sec": results["total"] / results["eval_time"]
    })

    # Create organized folder structure like evaluate_gsm8k() 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_folder = Path("evaluations") / f"{timestamp}_{model_name}_b{args.batch_size}"
    eval_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {eval_folder}")

    # Save detailed results to CSV
    results_df = pd.DataFrame({
        "question": results["questions"],
        "target": results["targets"],
        "prediction": results["predictions"],
        "is_correct": [p == t and t is not None for p, t in zip(results["predictions"], results["targets"])]
    })

    csv_path = eval_folder / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

    # Save summary
    summary_path = eval_folder / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"GSM8K Test Set Evaluation\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Correct: {results['correct']}/{results['total']}\n")
        f.write(f"Eval Time: {results['eval_time']:.2f}s\n")
        f.write(f"Throughput: {results['total'] / results['eval_time']:.2f} problems/sec\n")
    print(f"Summary saved to: {summary_path}")

    # Save evaluation metadata
    metadata_path = eval_folder / "metadata.json"
    import json
    metadata = {
        "model_name": args.model,
        "batch_size": args.batch_size,
        "timestamp": timestamp,
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "eval_time": results["eval_time"],
        "throughput": results["total"] / results["eval_time"]
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Upload files as WandB artifact
    artifact = wandb.Artifact("gsm8k_eval_results", type="evaluation")
    artifact.add_file(str(csv_path))
    artifact.add_file(str(summary_path))
    artifact.add_file(str(metadata_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"All files saved in: {eval_folder}")
    return results["accuracy"]


if __name__ == "__main__":
    main()