"""L3 cross-encoder reranker: (query, passage) -> relevance, trained with BCE on positives + hard negatives.

    python -m lexora.train.l3_reranker --dry-run
    python -m lexora.train.l3_reranker --base BAAI/bge-reranker-base      # GPU
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.config import settings
from lexora.train.l2_embedder import build_pairs
from lexora.train.qlora import write_jsonl


def build_triples(neg_per_pos: int = 4) -> list[dict]:
    rows = []
    for p in build_pairs(hard_negatives=neg_per_pos):
        rows.append({"query": p["anchor"], "passage": p["positive"], "label": 1.0})
        rows += [{"query": p["anchor"], "passage": n, "label": 0.0} for n in p["negatives"]]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = build_triples()
    path = write_jsonl(Path(settings().state_dir) / "datasets" / "l3_triples.jsonl", rows)
    print(f"L3 triples: {len(rows)} ({sum(r['label'] == 1 for r in rows)} positive) -> {path}")
    if a.dry_run:
        return rows
    from sentence_transformers import CrossEncoder, InputExample  # pragma: no cover - GPU path
    from torch.utils.data import DataLoader

    model = CrossEncoder(a.base, num_labels=1, max_length=512)
    loader = DataLoader([InputExample(texts=[r["query"], r["passage"]], label=r["label"]) for r in rows], shuffle=True, batch_size=16)
    out = Path(settings().state_dir) / "artifacts" / "l3_reranker"
    model.fit(train_dataloader=loader, epochs=a.epochs, warmup_steps=50)
    model.save(str(out))
    print(f"saved {out}  ->  set HELIX_RERANKER=ce:{out}")
    return rows


if __name__ == "__main__":
    main()
