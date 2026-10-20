"""L4 FIM completion adapter: continued pretraining with fill-in-the-middle + LoRA on internal repos.

Data comes from the prep pipeline (var/datasets/<version>/fim_train.jsonl): secrets redacted, PII masked,
copyleft excluded, near-duplicates removed, 50% PSM/SPM-transformed. Loss is computed on the whole
sequence here for simplicity; production masks loss to the middle span (see `middle_only_labels`).

    python -m lexora.train.l4_fim --dry-run
    python -m lexora.train.l4_fim --base Qwen/Qwen2.5-Coder-1.5B          # GPU (24 GB is plenty)
Alternative: Axolotl with configs/train/l4_fim_axolotl.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common.config import load_yaml, settings


def load_samples() -> list[dict]:
    root = Path(settings().state_dir) / "datasets"
    version = (root / "LATEST").read_text().strip()
    return [json.loads(ln) for ln in (root / version / "fim_train.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def middle_only_labels(input_ids: list[int], middle_token_id: int) -> list[int]:
    """Mask everything up to and including <fim_middle> with -100 so the loss covers the middle only."""
    if middle_token_id not in input_ids:
        return list(input_ids)
    k = input_ids.index(middle_token_id)
    return [-100] * (k + 1) + list(input_ids[k + 1 :])


def main(argv=None):
    rec = load_yaml("configs/train/recipes.yaml")["l4_fim"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=rec["base_model"])
    ap.add_argument("--epochs", type=int, default=rec["epochs"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = load_samples()
    print(f"L4 FIM samples: {len(rows)}  modes={dict(Counter(r['mode'] for r in rows))}")
    if a.dry_run:
        return rows
    import torch  # pragma: no cover - GPU path
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="auto")
    lc = rec["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    ds = Dataset.from_list(rows).map(lambda b: tok(b["text"], truncation=True, max_length=rec["max_length"]), batched=True,
                                     remove_columns=["text", "mode"])
    out = Path(settings().state_dir) / "artifacts" / "l4_fim"
    args = TrainingArguments(output_dir=str(out), num_train_epochs=a.epochs, learning_rate=rec["learning_rate"],
                             per_device_train_batch_size=4, gradient_accumulation_steps=8, bf16=True, logging_steps=10,
                             lr_scheduler_type="cosine", warmup_ratio=0.03, gradient_checkpointing=True, report_to="none")
    Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(str(out / "adapter"))
    print(f"saved adapter -> {out / 'adapter'}; serve with vLLM --enable-lora --lora-modules fim={out / 'adapter'}")
    return rows


if __name__ == "__main__":
    main()
