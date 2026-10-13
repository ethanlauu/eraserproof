import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from eraserproof.unlearning.gradient_difference import gradient_difference_unlearn, load_full_model


def main():
    parser = argparse.ArgumentParser(description="Run Gradient Difference unlearning from the M_full checkpoint.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--full-model", type=Path, default=ROOT / "checkpoints" / "full")
    parser.add_argument("--retain", type=Path, default=ROOT / "data" / "processed" / "retain.jsonl")
    parser.add_argument("--forget", type=Path, default=ROOT / "data" / "processed" / "forget.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "unlearned-gradient-difference")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, tokenizer, device = load_full_model(
        model_name=args.base_model,
        checkpoint_path=str(args.full_model),
        training=True,
    )

    model = gradient_difference_unlearn(
        model,
        tokenizer,
        str(args.retain),
        str(args.forget),
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_weight=args.lambda_weight,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"Saved unlearned model to {args.output}")


if __name__ == "__main__":
    main()
