"""Rerankers. RRF fuses ranked lists; the L3 cross-encoder rescores the top-N.

RRF(d) = sum_r 1 / (k + rank_r(d)), k ~ 60 - uses ranks only, so no score calibration across retrievers.
Cross-encoder: HELIX_RERANKER=ce:<model-or-path> (bge-reranker or the fine-tuned L3). Fallback: a
lexical-overlap + dense blend so the pipeline shape stays identical.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from common.textutils import tokenize


def rrf(lists: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    parts: dict[str, dict] = {}
    for name, ranked in lists.items():
        for r, d in enumerate(ranked, 1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
            parts.setdefault(d, {})[name] = r
    return sorted(((d, s, parts[d]) for d, s in scores.items()), key=lambda x: -x[1])


@lru_cache
def _cross_encoder():
    spec = os.environ.get("HELIX_RERANKER", "")
    if not spec.startswith("ce:"):
        return None
    from sentence_transformers import CrossEncoder  # type: ignore

    return CrossEncoder(spec[3:])


def rerank(query: str, items: list[tuple[str, str]], dense: dict[str, float], graph_ids: set[str]) -> list[tuple[str, float]]:
    """items = [(chunk_id, text)] -> [(chunk_id, score)] sorted desc."""
    ce = _cross_encoder()
    if ce is not None:  # pragma: no cover - optional model
        scores = ce.predict([(query, t) for _, t in items])
        return sorted(zip([i for i, _ in items], map(float, scores)), key=lambda x: -x[1])
    q = set(tokenize(query))
    out = []
    for cid, text in items:
        dt = set(tokenize(text))
        overlap = len(q & dt) / max(1, len(q))
        s = 0.6 * overlap + 0.4 * max(0.0, dense.get(cid, 0.0)) + (0.1 if cid in graph_ids else 0.0)
        out.append((cid, float(np.round(s, 4))))
    return sorted(out, key=lambda x: -x[1])
