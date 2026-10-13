import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "src"))

from eraserproof.models.load import load_base_model


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def build_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_eval_model(model_name: str, adapter_path: str | None = None):
    model, tokenizer, device = load_base_model(model_name=model_name, training=False)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    model.to(device)

    return model, tokenizer, device


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 32) -> str:
    prompt_text = build_prompt(tokenizer, prompt)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def target_sequence_nll(model, tokenizer, prompt: str, target: str, device: str):
    prompt_text = build_prompt(tokenizer, prompt)
    answer_text = target.strip()
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    prompt_length = prompt_tokens["input_ids"].shape[1]
    labels = full_tokens["input_ids"].clone()
    labels[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(**full_tokens, labels=labels)

    return float(outputs.loss.item())


def evaluate_examples(model, tokenizer, examples, device: str, max_new_tokens: int = 32):
    nll_values = []
    exact_matches = []

    for example in examples:
        prompt = example["prompt"]
        answer = example["answer"]

        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        exact = normalize_text(generated) == normalize_text(answer)

        nll = target_sequence_nll(
            model,
            tokenizer,
            prompt,
            answer,
            device,
        )

        nll_values.append(nll)
        exact_matches.append(exact)

    return {
        "target_sequence_nll": sum(nll_values) / max(len(nll_values), 1),
        "exact_extraction_accuracy": sum(int(v) for v in exact_matches) / max(len(exact_matches), 1),
        "exact_matches": exact_matches,
    }


def load_examples(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_metrics(model_name: str, metrics: dict):
    return {
        "model": model_name,
        "target_sequence_nll": round(metrics["target_sequence_nll"], 4),
        "exact_extraction_accuracy": round(metrics["exact_extraction_accuracy"], 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate base and adapter models on the synthetic QA benchmark.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base Hugging Face model name.")
    parser.add_argument("--retain", type=Path, default=ROOT / "data" / "processed" / "retain.jsonl", help="Retain data JSONL.")
    parser.add_argument("--forget", type=Path, default=ROOT / "data" / "processed" / "forget.jsonl", help="Forget data JSONL.")
    parser.add_argument("--full-model", type=Path, default=ROOT / "checkpoints" / "full", help="Adapter checkpoint for M_full.")
    parser.add_argument("--retrain-model", type=Path, default=ROOT / "checkpoints" / "retrain", help="Adapter checkpoint for M_retrain.")
    args = parser.parse_args()

    base_model, base_tokenizer, base_device = load_eval_model(args.base_model)
    full_model, full_tokenizer, full_device = load_eval_model(args.base_model, str(args.full_model))
    retrain_model, retrain_tokenizer, retrain_device = load_eval_model(args.base_model, str(args.retrain_model))

    retain_examples = load_examples(args.retain)
    forget_examples = load_examples(args.forget)

    models = [
        ("Base Qwen", base_model, base_tokenizer, base_device),
        ("M_full", full_model, full_tokenizer, full_device),
        ("M_retrain", retrain_model, retrain_tokenizer, retrain_device),
    ]

    print(f"{'model':<12} {'retain_nll':>12} {'retain_exact':>13} {'forget_nll':>12} {'forget_exact':>13}")
    print("-" * 80)

    for model_name, model, tokenizer, device in models:
        retain_metrics = evaluate_examples(model, tokenizer, retain_examples, device)
        forget_metrics = evaluate_examples(model, tokenizer, forget_examples, device)

        print(
            f"{model_name:<12} "
            f"{retain_metrics['target_sequence_nll']:>12.4f} "
            f"{retain_metrics['exact_extraction_accuracy']:>13.4f} "
            f"{forget_metrics['target_sequence_nll']:>12.4f} "
            f"{forget_metrics['exact_extraction_accuracy']:>13.4f}"
        )


if __name__ == "__main__":
    main()
