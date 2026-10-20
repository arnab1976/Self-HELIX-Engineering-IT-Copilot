"""Embedding backends for L2 (and L8, semantic cache).

  HashingEmbedder            default; no model download. Feature-hashes identifier tokens + char trigrams
                             into a 768-d L2-normalised vector. Good enough to demo hybrid retrieval.
  SentenceTransformerEmbedder  real model (bge-small/bge-m3 or the fine-tuned L2) when installed.
  TEIEmbedder                Hugging Face Text Embeddings Inference server (production serving).

Choose with HELIX_EMBEDDINGS_URL (TEI) or HELIX_EMBEDDER=st:<model-id>; otherwise hashing.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import httpx
import numpy as np

from common.textutils import char_ngrams, tokenize


class HashingEmbedder:
    name = "hashing-768"

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _idx(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            i, s = self._idx("w:" + t)
            v[i] += 2.0 * s
        for g, c in char_ngrams(text).items():
            i, s = self._idx("c:" + g)
            v[i] += 0.5 * s * min(c, 3)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_id)
        self.name = model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TEIEmbedder:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.name = f"tei:{self.url}"

    def embed(self, texts: list[str]) -> np.ndarray:
        r = httpx.post(f"{self.url}/embed", json={"inputs": texts, "normalize": True}, timeout=30)
        r.raise_for_status()
        return np.asarray(r.json(), dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache
def get_embedder():
    url = os.environ.get("HELIX_EMBEDDINGS_URL", "")
    spec = os.environ.get("HELIX_EMBEDDER", "")
    if url:
        return TEIEmbedder(url)
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    return HashingEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0
