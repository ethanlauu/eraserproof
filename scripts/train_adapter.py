import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from eraserproof.models.load import load_base_model
from eraserproof.training.collator import CausalLMCollator
from eraserproof.training.dataset import QADataset
from eraserproof.training.lora import attach_lora


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on QA examples.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the JSONL training data.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save the trained model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "baseline.json",
        help="Path to the training JSON config file.",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(config["seed"])

    model, tokenizer, device = load_base_model(
        model_name=config["model_name"],
        training=True,
    )

    model = attach_lora(
        model,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"],
    )
    model.to(device)

    dataset = QADataset(
        args.data,
        tokenizer,
        max_length=config["max_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss encountered during epoch {epoch}.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        average_loss = total_loss / max(num_batches, 1)
        print(f"epoch={epoch} average_loss={average_loss:.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
