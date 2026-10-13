import json

import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, path, tokenizer, max_length: int = 128):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        prompt = example["prompt"]
        answer = example["answer"]

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )["input_ids"]

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        labels = input_ids.clone()
        labels[: len(prompt_tokens)] = -100

        if len(input_ids) > len(prompt_tokens):
            labels[len(prompt_tokens) :] = input_ids[len(prompt_tokens) :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
