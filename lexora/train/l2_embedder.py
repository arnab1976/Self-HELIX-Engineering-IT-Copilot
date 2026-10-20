"""L2 code/document embedder: contrastive fine-tuning with in-batch + hard negatives (InfoNCE / MNRL).

L = -log( e^{s+/tau} / (e^{s+/tau} + sum_j e^{s-_j/tau}) )

Training pairs (spec 4.2): PR description <-> diff, ticket <-> resolving KB article, question <-> accepted
answer. The MVP mines (question, chunk) pairs from data/generated/gold and hard negatives from chunks in
the same repository.

    python -m lexora.train.l2_embedder --dry-run          # build var/datasets/l2_pairs.jsonl only
    python -m lexora.train.l2_embedder --base BAAI/bge-m3  # GPU
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.config import settings
from lexora.prep.pipeline import latest_chunks
from lexora.train.qlora import write_jsonl


def build_pairs(hard_negatives: int = 3, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    chunks = latest_chunks()
    by_symbol = {c.symbol: c for c in chunks}
    gold = [json.loads(ln) for ln in (Path(settings().data_dir) / "generated" / "gold" / "retrieval.jsonl").read_text().splitlines() if ln]
    rows = []
    for g in gold:
        for sym in g["relevant"]:
            pos = by_symbol.get(sym)
            if not pos:
                continue
            pool = [c for c in chunks if c.repo == pos.repo and c.symbol not in g["relevant"]]
            negs = rng.sample(pool, min(hard_negatives, len(pool)))
            rows.append({"anchor": g["q"], "positive": pos.text, "negatives": [n.text for n in negs]})
    # PR/commit-style self-supervision: symbol docstring/name -> body
    for c in chunks:
        if c.kind in ("function", "method"):
            rows.append({"anchor": c.symbol.replace(".", " ").replace("_", " "), "positive": c.text, "negatives": []})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = build_pairs()
    path = write_jsonl(Path(settings().state_dir) / "datasets" / "l2_pairs.jsonl", rows)
    print(f"L2 pairs: {len(rows)} -> {path}")
    if a.dry_run:
        return rows
    from datasets import Dataset  # pragma: no cover - GPU path
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    flat = [{"anchor": r["anchor"], "positive": r["positive"], "negative": (r["negatives"] or [r["positive"]])[0]} for r in rows]
    model = SentenceTransformer(a.base)
    out = Path(settings().state_dir) / "artifacts" / "l2_embedder"
    args = SentenceTransformerTrainingArguments(output_dir=str(out), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
                                                learning_rate=2e-5, warmup_ratio=0.1, bf16=True, report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(flat),
                                         loss=MultipleNegativesRankingLoss(model, scale=20.0))  # scale = 1/tau
    trainer.train()
    model.save(str(out / "final"))
    print(f"saved {out / 'final'}  ->  set HELIX_EMBEDDER=st:{out / 'final'}")
    return rows


if __name__ == "__main__":
    main()
