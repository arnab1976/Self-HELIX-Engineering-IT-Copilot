"""Shared QLoRA helpers for L5/L6/L7 (spec 4.4).

QLoRA = base weights in 4-bit NF4 with double quantisation, BF16 compute, LoRA adapters on all linear
layers, paged AdamW. Trainable params per matrix = r * (d_in + d_out).
Written against requirements-train.txt (transformers 4.46, peft 0.13, trl 0.12, bitsandbytes 0.44).
"""
from __future__ import annotations

import json
from pathlib import Path


def lora_param_count(dims: dict[str, tuple[int, int]], r: int, layers: int) -> int:
    """Trainable LoRA parameters = layers * sum_modules r * (d_in + d_out)."""
    return layers * sum(r * (a + b) for a, b in dims.values())


def load_4bit(model_id: str):  # pragma: no cover - GPU path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="sdpa")
    return model, tok


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05, target="all-linear"):  # pragma: no cover
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, bias="none", task_type="CAUSAL_LM")


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def sft(model_id: str, train_path: Path, out_dir: Path, r: int, alpha: int, lr: float, epochs: int, max_len: int = 4096):  # pragma: no cover
    """Supervised fine-tuning on chat-format rows {"messages": [...]} with QLoRA."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = load_4bit(model_id)
    ds = load_dataset("json", data_files=str(train_path), split="train")
    args = SFTConfig(output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=2,
                     gradient_accumulation_steps=8, gradient_checkpointing=True, bf16=True, optim="paged_adamw_8bit",
                     lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
                     max_seq_length=max_len, packing=True, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora_config(r, alpha))
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    return out_dir / "adapter"


def dpo(model_id: str, adapter: Path | None, pairs_path: Path, out_dir: Path, beta: float, lr: float, epochs: int):  # pragma: no cover
    """Direct Preference Optimisation on {"prompt", "chosen", "rejected"} rows. ref model = frozen base."""
    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    model, tok = load_4bit(model_id)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    ds = load_dataset("json", data_files=str(pairs_path), split="train")
    args = DPOConfig(output_dir=str(out_dir), beta=beta, learning_rate=lr, num_train_epochs=epochs, per_device_train_batch_size=1,
                     gradient_accumulation_steps=16, gradient_checkpointing=True, bf16=True, optim="paged_adamw_8bit",
                     logging_steps=10, report_to="none", max_length=4096, max_prompt_length=3072)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=None if adapter else lora_config())
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    return out_dir / "adapter"
