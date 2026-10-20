"""L8 duplicate detection at inference: embed new ticket, nearest open tickets, logistic probability."""
from __future__ import annotations

import time
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np

from common.embeddings import get_embedder
from common.store import store
from lexora.registry import get_card


@lru_cache
def _model():
    card = get_card("lexora/duplicates")
    if card is None:
        raise RuntimeError("L8 duplicates model not registered: run `make bootstrap`")
    b = joblib.load(card.artifact_path)
    b["ref"] = card.ref
    return b


def find_duplicates(title: str, description: str, ci: str = "", requester: str = "", created_at: float | None = None,
                    k: int = 5) -> list[dict]:
    b = _model()
    open_t = store().query("SELECT * FROM tickets WHERE state IN ('New','In Progress') ORDER BY created_at DESC LIMIT 500")
    if not open_t:
        return []
    emb = get_embedder()
    q = emb.embed_one(f"{title} {description}")
    vecs = emb.embed([f"{t['title']} {t['description']}" for t in open_t])
    sims = vecs @ q
    top = np.argsort(-sims)[: k * 3]
    now = created_at or time.time()
    out = []
    for i in top:
        t = open_t[i]
        gap_days = min(30.0, abs(now - (t["created_at"] or now)) / 86400)
        x = np.array([[float(sims[i]), float(bool(ci) and t["ci"] == ci), float(bool(requester) and t["requester"] == requester), gap_days]])
        p = float(b["model"].predict_proba(x)[0, 1])
        z = float(b["model"].intercept_[0] + x[0] @ b["model"].coef_[0])
        out.append({"ticket": t["id"], "title": t["title"], "cosine": round(float(sims[i]), 3), "same_ci": bool(x[0, 1]),
                    "gap_days": round(gap_days, 2), "z": round(z, 3), "p": round(p, 3), "duplicate": p >= b["threshold"]})
    return sorted(out, key=lambda r: -r["p"])[:k]


def threshold() -> float:
    return float(_model()["threshold"])


def iso_to_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()
