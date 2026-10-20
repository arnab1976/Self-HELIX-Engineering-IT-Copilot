"""Vector stores with ACL payload filtering applied BEFORE ranking.

InMemoryVectorStore: exact cosine over a NumPy matrix (fine for the MVP corpus).
QdrantVectorStore:   HNSW ANN with payload filter on repo_class (HELIX_QDRANT_URL).
"""
from __future__ import annotations

import numpy as np


class InMemoryVectorStore:
    def __init__(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]):
        self.ids, self.vectors, self.payloads = ids, vectors.astype(np.float32), payloads
        self.pos = {i: k for k, i in enumerate(ids)}

    def search(self, qvec: np.ndarray, k: int = 20, allowed_classes: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        mask = np.ones(len(self.ids), bool)
        if allowed_classes is not None:
            mask = np.array([p.get("repo_class") in allowed_classes for p in self.payloads])
        sims = self.vectors @ qvec
        sims[~mask] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]

    def vector(self, chunk_id: str) -> np.ndarray:
        return self.vectors[self.pos[chunk_id]]


class QdrantVectorStore:  # pragma: no cover - requires a running Qdrant
    def __init__(self, url: str, collection: str = "helix_chunks", dim: int = 768):
        from qdrant_client import QdrantClient, models

        self.m = models
        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        import uuid

        pts = [self.m.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)), vector=v.tolist(), payload={**p, "chunk_id": i})
               for i, v, p in zip(ids, vectors, payloads)]
        self.client.upsert(self.collection, pts)

    def search(self, qvec: np.ndarray, k: int = 20, allowed_classes: set[str] | None = None) -> list[tuple[str, float]]:
        flt = None
        if allowed_classes is not None:
            flt = self.m.Filter(must=[self.m.FieldCondition(key="repo_class", match=self.m.MatchAny(any=sorted(allowed_classes)))])
        res = self.client.query_points(self.collection, query=qvec.tolist(), limit=k, query_filter=flt).points
        return [(p.payload["chunk_id"], float(p.score)) for p in res]
