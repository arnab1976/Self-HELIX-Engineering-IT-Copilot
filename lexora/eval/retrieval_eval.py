"""Retrieval evaluation on data/generated/gold/retrieval.jsonl: Recall@10, MRR, nDCG@10 -> lexora/retriever card.

    python -m lexora.eval.retrieval_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.config import settings
from lexora.eval.gates import check
from lexora.eval.metrics import mrr, ndcg_at_k, recall_at_k
from lexora.registry import ModelCard, get_card, register


def main() -> dict:
    from nexus.gateway.auth import Principal
    from nexus.rag.retriever import retrieve

    gold = [json.loads(ln) for ln in (Path(settings().data_dir) / "generated" / "gold" / "retrieval.jsonl").read_text().splitlines() if ln]
    user = Principal("eval", "eval", ["engineer"])
    rows = []
    for g in gold:
        res = retrieve(g["q"], user, top_n=10)
        ranked = [h.symbol for h in res.hits]
        rel = set(g["relevant"])
        rows.append({"q": g["q"], "recall@10": recall_at_k(ranked, rel, 10), "mrr": mrr(ranked, rel),
                     "ndcg@10": ndcg_at_k(ranked, {r: 1.0 for r in rel}, 10), "top3": ranked[:3]})
    sc = {"recall_at_10": float(np.mean([r["recall@10"] for r in rows])), "mrr": float(np.mean([r["mrr"] for r in rows])),
          "ndcg_at_10": float(np.mean([r["ndcg@10"] for r in rows])), "queries": len(rows)}
    card = get_card("lexora/retriever")
    gate = check("lexora/retriever", sc)
    if card:
        card.scorecard, card.gate, card.version = sc, gate, "0.1.1-mvp"
        register(card)
    else:
        register(ModelCard("lexora/retriever", "0.1.1-mvp", "L2+L3", ["code-retrieval"], "hashing-768", "hybrid", "-", sc, gate))
    print(json.dumps({"retrieval": sc, "passed": gate["passed"]}, indent=1))
    for r in rows:
        if r["recall@10"] < 1:
            print("  miss:", r["q"], "->", r["top3"])
    return sc


if __name__ == "__main__":
    main()
