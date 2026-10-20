"""Retrieval pipeline for code and knowledge questions (spec 6.3).

1 understand  extract repo/service mentions          4 fuse     reciprocal rank fusion
2 scope       ACL: repos the user can read            5 rerank   L3 cross-encoder to the top N
3 retrieve    BM25 + dense (L2) + graph expansion     6 assemble citations path:lines@sha; injection-check chunks
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.config import cfg
from common.embeddings import get_embedder
from common.tracing import span
from nexus.gateway.auth import Principal
from nexus.guardrails import injection
from nexus.rag.index import HelixIndex, get_index, index_text
from nexus.rag.rerank import rerank, rrf


@dataclass
class Hit:
    chunk_id: str
    score: float
    citation: str
    symbol: str
    repo: str
    kind: str
    text: str
    injection_score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    stages: dict = field(default_factory=dict)
    acl_filtered: int = 0
    index_version: str = ""


def retrieve(query: str, user: Principal, repo: str | None = None, top_n: int | None = None,
             idx: HelixIndex | None = None) -> RetrievalResult:
    idx = idx or get_index()
    rc = cfg()["rag"]
    top_n = top_n or rc["context_chunks"]
    allowed_classes = user.repo_classes
    allowed = {cid for cid, c in idx.chunks.items() if c.repo_class in allowed_classes and (repo is None or c.repo == repo
                                                                                         or c.repo == "docs/engineering-handbook")}
    with span("rag.retrieve", query=query[:80]):
        bm = idx.bm25.search(query, rc["bm25_top_k"], allowed)
        qv = get_embedder().embed_one(query)
        dense = [(i, s) for i, s in idx.vectors.search(qv, rc["dense_top_k"], allowed_classes) if i in allowed]
        seeds = list(dict.fromkeys([i for i, _ in bm[:2]] + [i for i, _ in dense[:2]]))
        graph = idx.graph.expand(seeds, rc["graph_hops"], allowed)
        fused = rrf({"bm25": [i for i, _ in bm], "dense": [i for i, _ in dense], "graph": [g[0] for g in graph]}, rc["rrf_k"])
        cand = fused[: rc["rerank_top_n"]]
        dense_map = dict(dense)
        rr = rerank(query, [(cid, index_text(idx.chunks[cid])) for cid, _, _ in cand], dense_map, {g[0] for g in graph})
    hits = []
    for cid, s in rr[:top_n]:
        c = idx.chunks[cid]
        inj, _ = injection.score(c.text)
        hits.append(Hit(cid, s, c.citation(), c.symbol, c.repo, c.kind, c.text, round(inj, 3)))
    return RetrievalResult(query, hits, {
        "bm25": [{"id": i, "symbol": idx.chunks[i].symbol, "score": round(s, 3)} for i, s in bm[:8]],
        "dense": [{"id": i, "symbol": idx.chunks[i].symbol, "score": round(s, 3)} for i, s in dense[:8]],
        "graph": [{"id": g[0], "symbol": idx.chunks[g[0]].symbol, "via": idx.chunks[g[1]].symbol, "edge": g[2]} for g in graph],
        "rrf": [{"id": d, "symbol": idx.chunks[d].symbol, "score": round(s, 4), "ranks": p} for d, s, p in fused[:10]],
        "reranked": [{"id": i, "symbol": idx.chunks[i].symbol, "score": s} for i, s in rr[:top_n]],
    }, len(idx.chunks) - len(allowed), idx.version)
