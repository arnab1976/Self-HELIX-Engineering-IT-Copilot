"""Dev-only API for LEXORA/NEXUS Agent 04 · Grounded Retrieval (Code RAG).

Retrieves over Agent 3b handoff chunks (the RAG store), then composes an answer
strictly from surviving evidence. No open-world LLM knowledge path.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings

rag_lab_api = APIRouter()
log = logging.getLogger("nexus.api.rag_lab")

_TOKEN = re.compile(r"[a-z0-9_]{2,}", re.I)


class RagRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chunks: list[dict] = Field(default_factory=list, max_length=500)
    index_version: str = Field(default="", max_length=120)
    dataset_version: str = Field(default="", max_length=120)
    primary_source: str = Field(default="", max_length=64)
    categories: list[str] = Field(default_factory=list, max_length=12)
    top_k: int = Field(default=5, ge=1, le=20)
    repo_class: Literal["engineering", "restricted", "public"] = "engineering"
    rrf_k: int = Field(default=60, ge=1, le=200)


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    hits: list[dict] = Field(default_factory=list, max_length=20)
    index_version: str = Field(default="", max_length=120)
    dataset_version: str = Field(default="", max_length=120)
    primary_source: str = Field(default="", max_length=64)
    require_citations: bool = True


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _chunk_id(chunk: dict, i: int) -> str:
    raw = str(chunk.get("symbol") or chunk.get("id") or chunk.get("path") or i)
    return f"c{i}-{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def _chunk_category(chunk: dict, primary_source: str) -> str:
    kind = str(chunk.get("kind") or "").lower()
    path = str(chunk.get("path") or "").lower()
    src = (primary_source or chunk.get("primary_source") or "").lower()
    if src in {"code", "tickets", "knowledge", "logs", "observability"}:
        return src
    if kind in {"function", "method", "class", "module"} and path.endswith((".py", ".ts", ".js", ".go")):
        return "code"
    if "inc" in path or "ticket" in path or kind in {"ticket", "doc"} and "json" in path:
        return "tickets"
    if kind == "log" or path.endswith(".log"):
        return "logs"
    if "kb" in path or "runbook" in path or path.endswith(".md"):
        return "knowledge"
    return src or "mixed"


def _acl_ok(chunk: dict, repo_class: str) -> bool:
    chunk_class = str(chunk.get("repo_class") or chunk.get("acl") or "engineering").lower()
    if repo_class == "restricted":
        return chunk_class in {"restricted", "engineering", "public"}
    if repo_class == "public":
        return chunk_class == "public"
    # engineering caller: exclude restricted
    return chunk_class != "restricted"


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], avgdl: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    k1, b = 1.2, 0.75
    for t in set(query_tokens):
        f = tf.get(t, 0)
        if not f:
            continue
        # Educational IDF: prefer rarer terms within this tiny corpus via frequency penalty
        idf = math.log(1.0 + 1.5 / (1.0 + f * 0.05))
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1.0)))
    return score


def _dense_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qs, ds = set(query_tokens), set(doc_tokens)
    inter = len(qs & ds)
    if not inter:
        return 0.0
    return inter / math.sqrt(len(qs) * len(ds))


def _rrf(lists: dict[str, list[str]], k: int) -> list[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    parts: dict[str, dict] = {}
    for name, ranked in lists.items():
        for r, cid in enumerate(ranked, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
            parts.setdefault(cid, {})[name] = r
    return sorted(((cid, s, parts[cid]) for cid, s in scores.items()), key=lambda x: -x[1])


def _normalise_chunks(chunks: list[dict], primary_source: str) -> list[dict]:
    out = []
    for i, raw in enumerate(chunks):
        text = str(raw.get("text") or "")
        if not text.strip():
            continue
        cid = _chunk_id(raw, i)
        out.append(
            {
                "id": cid,
                "symbol": str(raw.get("symbol") or raw.get("id") or f"chunk-{i + 1}"),
                "kind": str(raw.get("kind") or "module"),
                "kind_label": str(raw.get("kind_label") or raw.get("kind") or "MODULE"),
                "path": str(raw.get("path") or ""),
                "citation": str(
                    raw.get("citation")
                    or f"{raw.get('path') or 'chunk'}:{raw.get('start') or '?'}-{raw.get('end') or '?'}"
                ),
                "text": text,
                "tokens": int(raw.get("tokens") or max(1, len(text) // 4)),
                "category": _chunk_category(raw, primary_source),
                "repo_class": str(raw.get("repo_class") or raw.get("acl") or "engineering"),
                "imports": list(raw.get("imports") or []),
                "calls": list(raw.get("calls") or []),
            }
        )
    return out


@rag_lab_api.post("/nexus/rag/retrieve", tags=["NEXUS RAG lab"])
def rag_retrieve(request: RagRetrieveRequest):
    _require_dev()
    t0 = time.perf_counter()
    if not request.chunks:
        raise HTTPException(
            status_code=400,
            detail="No indexed chunks. Complete Agent 3b Index builder, then import the index in Agent 04.",
        )
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a question before retrieving.")

    docs = _normalise_chunks(request.chunks, request.primary_source)
    allowed_cats = {c.lower() for c in request.categories if c}
    scoped = []
    acl_dropped = 0
    category_dropped = 0
    for d in docs:
        if not _acl_ok(d, request.repo_class):
            acl_dropped += 1
            continue
        if allowed_cats and d["category"] not in allowed_cats:
            category_dropped += 1
            continue
        scoped.append(d)
    by_id = {d["id"]: d for d in scoped}

    q_tokens = _tokens(query)
    avgdl = sum(len(_tokens(d["text"])) for d in scoped) / max(len(scoped), 1)

    bm25_ranked = sorted(
        ((d["id"], _bm25_score(q_tokens, _tokens(d["text"]), avgdl)) for d in scoped),
        key=lambda x: -x[1],
    )
    dense_ranked = sorted(
        ((d["id"], _dense_score(q_tokens, _tokens(d["text"]))) for d in scoped),
        key=lambda x: -x[1],
    )
    # Graph expansion: seed from top lexical/dense, hop via shared imports/calls/symbols
    seed_ids = [i for i, s in bm25_ranked[:2] if s > 0] + [i for i, s in dense_ranked[:2] if s > 0]
    seed_ids = list(dict.fromkeys(seed_ids))
    graph_hits: list[str] = []
    for sid in seed_ids:
        seed = by_id.get(sid)
        if not seed:
            continue
        links = {str(x).lower() for x in (seed.get("imports") or []) + (seed.get("calls") or [])}
        links.add(str(seed.get("symbol") or "").lower())
        for d in scoped:
            if d["id"] == sid:
                continue
            blob = " ".join(
                [str(d.get("symbol") or "").lower()]
                + [str(x).lower() for x in (d.get("imports") or [])]
                + [str(x).lower() for x in (d.get("calls") or [])]
            )
            if any(tok and tok in blob for tok in links):
                graph_hits.append(d["id"])
    graph_hits = list(dict.fromkeys(graph_hits))

    fused = _rrf(
        {
            "bm25": [i for i, s in bm25_ranked if s > 0] or [i for i, _ in bm25_ranked],
            "dense": [i for i, s in dense_ranked if s > 0] or [i for i, _ in dense_ranked],
            "graph": graph_hits or ([seed_ids[0]] if seed_ids else []),
        },
        request.rrf_k,
    )

    hits = []
    for cid, score, ranks in fused[: request.top_k]:
        d = by_id.get(cid)
        if not d:
            continue
        # Drop near-zero evidence
        if score <= 0 and not any(t in d["text"].lower() for t in q_tokens[:6]):
            continue
        hits.append(
            {
                "chunk_id": cid,
                "score": round(score, 4),
                "ranks": ranks,
                "symbol": d["symbol"],
                "kind": d["kind"],
                "kind_label": d["kind_label"],
                "path": d["path"],
                "citation": d["citation"],
                "category": d["category"],
                "repo_class": d["repo_class"],
                "text": d["text"],
                "tokens": d["tokens"],
            }
        )

    seconds = round(time.perf_counter() - t0, 3)
    return {
        "query": query,
        "index_version": request.index_version or "idx-handoff",
        "dataset_version": request.dataset_version,
        "primary_source": request.primary_source,
        "categories_filter": sorted(allowed_cats),
        "corpus_size": len(docs),
        "scoped_size": len(scoped),
        "acl_filtered": acl_dropped,
        "category_filtered": category_dropped,
        "hits": hits,
        "hit_count": len(hits),
        "stages": {
            "bm25": [{"id": i, "score": round(s, 3), "symbol": by_id[i]["symbol"]} for i, s in bm25_ranked[:8] if i in by_id],
            "dense": [{"id": i, "score": round(s, 3), "symbol": by_id[i]["symbol"]} for i, s in dense_ranked[:8] if i in by_id],
            "graph": [{"id": i, "symbol": by_id[i]["symbol"]} for i in graph_hits[:8] if i in by_id],
            "rrf": [{"id": i, "score": round(s, 4), "ranks": p, "symbol": by_id[i]["symbol"]} for i, s, p in fused[:10] if i in by_id],
        },
        "seconds": seconds,
        "next": (
            "Compose a grounded answer from these hits (Agent 04 Step D)."
            if hits
            else "No evidence in the indexed corpus for this query — do not invent an answer."
        ),
    }


@rag_lab_api.post("/nexus/rag/answer", tags=["NEXUS RAG lab"])
def rag_answer(request: RagAnswerRequest):
    _require_dev()
    t0 = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a question before composing an answer.")
    hits = [h for h in request.hits if str(h.get("text") or "").strip()]
    if not hits:
        return {
            "query": query,
            "status": "insufficient_evidence",
            "answer": (
                "Insufficient evidence in the Agent 3b index for this question. "
                "No grounded answer will be generated from model memory."
            ),
            "citations": [],
            "evidence": [],
            "index_version": request.index_version or "idx-handoff",
            "dataset_version": request.dataset_version,
            "primary_source": request.primary_source,
            "require_citations": request.require_citations,
            "seconds": round(time.perf_counter() - t0, 3),
            "next": "Re-run retrieval with a different question, or rebuild the index from Step 02 chunks.",
        }

    # Evidence-based composition (deterministic mock provider style — no open-world knowledge)
    top = hits[:5]
    bullets = []
    citations = []
    for i, h in enumerate(top, 1):
        cite = str(h.get("citation") or h.get("path") or h.get("symbol") or f"evidence-{i}")
        symbol = str(h.get("symbol") or cite)
        snippet = " ".join(str(h.get("text") or "").split())
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
        bullets.append(f"{i}. [{cite}] {symbol}: {snippet}")
        citations.append(
            {
                "n": i,
                "citation": cite,
                "symbol": symbol,
                "chunk_id": h.get("chunk_id") or h.get("id") or "",
                "category": h.get("category") or "",
                "score": h.get("score"),
            }
        )

    lead = (
        f"Based on {len(top)} ACL-passed chunk(s) from index "
        f"{request.index_version or 'idx-handoff'} "
        f"(dataset {request.dataset_version or '—'}):"
    )
    answer = lead + "\n\n" + "\n".join(bullets)
    answer += (
        "\n\nThis answer is composed only from retrieved evidence above. "
        "Claims without a citation were not asserted."
    )

    packet_id = "ans-" + hashlib.sha1(
        (query + "|" + request.index_version + "|" + str(len(top))).encode()
    ).hexdigest()[:10]

    return {
        "query": query,
        "status": "grounded",
        "answer_id": packet_id,
        "answer": answer,
        "citations": citations,
        "evidence": [
            {
                "chunk_id": h.get("chunk_id") or h.get("id"),
                "symbol": h.get("symbol"),
                "citation": h.get("citation"),
                "category": h.get("category"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
            for h in top
        ],
        "index_version": request.index_version or "idx-handoff",
        "dataset_version": request.dataset_version,
        "primary_source": request.primary_source,
        "require_citations": request.require_citations,
        "seconds": round(time.perf_counter() - t0, 3),
        "next": "Answer packet ready for Gateway / workflow agents. Prefer Evaluate & register next for release gates.",
    }
