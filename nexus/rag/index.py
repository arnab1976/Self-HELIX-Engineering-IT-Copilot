"""HelixIndex: BM25 + vectors + code graph over the latest LEXORA dataset. Built by scripts/bootstrap.py,
persisted to var/index/index.pkl and loaded once per process."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from common.config import cfg, settings
from common.embeddings import get_embedder
from lexora.prep.chunking import Chunk
from nexus.rag.bm25 import BM25
from nexus.rag.graph import CodeGraph
from nexus.rag.vectorstore import InMemoryVectorStore


def index_text(c: Chunk) -> str:
    return f"{c.symbol} {c.path} {c.text}"


@dataclass
class HelixIndex:
    version: str
    chunks: dict[str, Chunk]
    bm25: BM25
    vectors: InMemoryVectorStore
    graph: CodeGraph
    embedder_name: str

    @property
    def symbols(self) -> set[str]:
        out = set()
        for c in self.chunks.values():
            out.add(c.symbol)
            out.add(c.symbol.split(".")[-1])
        return out


def build(chunks: list[Chunk], version: str) -> HelixIndex:
    emb = get_embedder()
    ids = [c.id for c in chunks]
    vecs = emb.embed([index_text(c) for c in chunks])
    rc = cfg()["rag"]["bm25"]
    return HelixIndex(version, {c.id: c for c in chunks}, BM25({c.id: index_text(c) for c in chunks}, rc["k1"], rc["b"]),
                      InMemoryVectorStore(ids, vecs, [{"repo_class": c.repo_class, "repo": c.repo} for c in chunks]),
                      CodeGraph(chunks), getattr(emb, "name", "embedder"))


def index_path() -> Path:
    return settings().path("index", "index.pkl")


def save(idx: HelixIndex) -> Path:
    p = index_path()
    with open(p, "wb") as f:
        pickle.dump(idx, f)
    return p


_IDX: HelixIndex | None = None


def get_index(rebuild: bool = False) -> HelixIndex:
    global _IDX
    if _IDX is None or rebuild:
        p = index_path()
        if p.exists() and not rebuild:
            with open(p, "rb") as f:
                _IDX = pickle.load(f)
        else:
            from lexora.prep import pipeline

            root = Path(settings().state_dir) / "datasets"
            if not (root / "LATEST").exists():
                pipeline.run()
            version = (root / "LATEST").read_text().strip()
            _IDX = build(pipeline.latest_chunks(), version)
            save(_IDX)
    return _IDX
