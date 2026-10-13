import torch


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        max_length = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full(
                            (pad_length,),
                            self.tokenizer.pad_token_id,
                            dtype=item["input_ids"].dtype,
                        ),
                    ]
                )
            )
            attention_masks.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(
                            (pad_length,),
                            dtype=item["attention_mask"].dtype,
                        ),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full(
                            (pad_length,),
                            -100,
                            dtype=item["labels"].dtype,
                        ),
                    ]
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
