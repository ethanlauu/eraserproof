from peft import LoraConfig, get_peft_model


def attach_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    return get_peft_model(model, config)
