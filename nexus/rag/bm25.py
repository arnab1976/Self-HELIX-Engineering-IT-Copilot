"""BM25 (Okapi). score(q, d) = sum_t IDF(t) * f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b|d|/avgdl)).
Production uses OpenSearch/Elasticsearch; this in-process version has the same scoring and an `explain`."""
from __future__ import annotations

import math
from collections import Counter

from common.textutils import tokenize


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = {i: Counter(tokenize(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avgdl = sum(self.len.values()) / max(1, len(self.len))
        df: Counter = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _term(self, t: str, i: str) -> float:
        f = self.tf[i].get(t, 0)
        if not f:
            return 0.0
        return self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))

    def search(self, query: str, k: int = 20, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = set(tokenize(query))
        scored = []
        for i in self.ids:
            if allowed is not None and i not in allowed:
                continue
            s = sum(self._term(t, i) for t in q)
            if s > 0:
                scored.append((i, s))
        return sorted(scored, key=lambda x: -x[1])[:k]

    def explain(self, query: str, doc_id: str) -> list[dict]:
        out = []
        for t in sorted(set(tokenize(query))):
            out.append({"term": t, "tf": self.tf[doc_id].get(t, 0), "idf": round(self.idf.get(t, 0.0), 3),
                        "score": round(self._term(t, doc_id), 3)})
        return out
