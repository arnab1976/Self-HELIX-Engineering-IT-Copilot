"""Semantic cache (spec 8.2): nearest cached query within the same ACL group and corpus version, returned
when cosine >= 0.95. Never used for restricted repositories. Production: Redis with vector search."""
from __future__ import annotations

import time

import numpy as np

from common.config import cfg
from common.embeddings import get_embedder


class SemanticCache:
    def __init__(self, max_items: int = 5000, ttl_s: int = 3600):
        self.items: list[dict] = []
        self.max_items, self.ttl_s = max_items, ttl_s

    @staticmethod
    def acl_key(repo_classes: set[str]) -> str:
        return ",".join(sorted(repo_classes))

    def get(self, query: str, acl: str, version: str) -> tuple[dict | None, float]:
        if "restricted" in acl:
            return None, 0.0
        q = get_embedder().embed_one(query)
        now = time.time()
        best, best_s = None, 0.0
        for it in self.items:
            if it["acl"] != acl or it["version"] != version or now - it["ts"] > self.ttl_s:
                continue
            s = float(np.dot(q, it["vec"]))
            if s > best_s:
                best, best_s = it, s
        if best and best_s >= cfg()["rag"]["semantic_cache_threshold"]:
            return best["value"], best_s
        return None, best_s

    def put(self, query: str, acl: str, version: str, value: dict) -> None:
        if "restricted" in acl:
            return
        self.items.append({"vec": get_embedder().embed_one(query), "acl": acl, "version": version, "value": value, "ts": time.time()})
        self.items = self.items[-self.max_items :]


cache = SemanticCache()
