import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch

from eraserproof.models.load import load_base_model


def main():
    model, tokenizer, device = load_base_model()

    print(f"Running on: {device}")

    messages = [
        {
            "role": "user",
            "content": "Explain machine unlearning in one sentence.",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    print()
    print(response)


if __name__ == "__main__":
    main()