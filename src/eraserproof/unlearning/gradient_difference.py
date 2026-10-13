import random
from itertools import cycle

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import PeftModel

from eraserproof.models.load import load_base_model
from eraserproof.training.collator import CausalLMCollator
from eraserproof.training.dataset import QADataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_full_model(model_name: str, checkpoint_path: str, training: bool = True):
    model, tokenizer, device = load_base_model(model_name=model_name, training=training)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device)
    return model, tokenizer, device


def gradient_difference_unlearn(
    model,
    tokenizer,
    retain_path,
    forget_path,
    device,
    *,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    lambda_weight: float = 0.5,
    batch_size: int = 1,
    max_length: int = 128,
    seed: int = 42,
):
    set_seed(seed)

    retain_dataset = QADataset(retain_path, tokenizer, max_length=max_length)
    forget_dataset = QADataset(forget_path, tokenizer, max_length=max_length)

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer),
    )
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer),
    )

    forget_iter = cycle(forget_loader)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for param in model.parameters():
        param.requires_grad = True

    model.train()

    for epoch in range(1, epochs + 1):
        total_retain_loss = 0.0
        total_forget_loss = 0.0
        count = 0

        for batch in retain_loader:
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)

            retain_batch = {key: value.to(device) for key, value in batch.items()}
            forget_batch = {key: value.to(device) for key, value in forget_batch.items()}

            optimizer.zero_grad()

            retain_outputs = model(**retain_batch)
            forget_outputs = model(**forget_batch)

            retain_loss = retain_outputs.loss
            forget_loss = forget_outputs.loss

            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            count += 1

            loss = retain_loss - lambda_weight * forget_loss
            loss.backward()
            optimizer.step()

        avg_retain = total_retain_loss / max(count, 1)
        avg_forget = total_forget_loss / max(count, 1)
        print(f"epoch={epoch} retain_loss={avg_retain:.6f} forget_loss={avg_forget:.6f}")

    return model
