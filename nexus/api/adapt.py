"""Dev-only API for LEXORA Agent 3: Adapt (fine-tune vs retrieve vs tool) + Agent 3b Index builder."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings

adapt_api = APIRouter()
log = logging.getLogger("nexus.api.adapt")


class AdaptDecideRequest(BaseModel):
    capability: str = Field(default="Grounded engineering Q&A", max_length=200)
    freshness: Literal["daily", "weekly", "monthly", "stable"] = "weekly"
    action: Literal["yes", "no"] = "no"
    quality: float = Field(default=0.85, ge=0.0, le=1.0)
    refresh_cost: float = Field(default=12000, ge=0, le=1_000_000)
    rights_clear: Literal["yes", "no", "unclear"] = "yes"
    restricted_data: Literal["yes", "no"] = "no"
    chunk_count: int = Field(default=0, ge=0, le=1_000_000)
    template_count: int = Field(default=0, ge=0, le=1_000_000)
    primary_source: str = Field(default="", max_length=64)
    dataset_version: str = Field(default="", max_length=120)


class IndexBuildRequest(BaseModel):
    decision: Literal["retrieve", "fine-tune", "tool"] = "retrieve"
    chunks: list[dict] = Field(default_factory=list, max_length=500)
    dataset_version: str = Field(default="", max_length=120)
    primary_source: str = Field(default="", max_length=64)
    build_bm25: bool = True
    build_dense: bool = True
    build_graph: bool = True


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


def _decide(req: AdaptDecideRequest) -> dict:
    reasons: list[str] = []
    blockers: list[str] = []

    if req.action == "yes":
        choice = "tool"
        reasons.append("Capability requires a live action or system state change → tool / workflow path.")
    elif req.freshness in {"daily", "weekly"}:
        choice = "retrieve"
        reasons.append(f"Knowledge freshness is {req.freshness} → prefer retrieval over owned model weights.")
    elif req.rights_clear != "yes":
        choice = "retrieve"
        reasons.append("Training rights are unclear → do not fine-tune; use retrieval or withhold.")
        blockers.append("Clarify data rights before any fine-tune track.")
    elif req.restricted_data == "yes":
        choice = "retrieve"
        reasons.append("Restricted data class → keep evidence in ACL-filtered retrieval; avoid external fine-tune.")
        blockers.append("Restricted corpus cannot leave the policy boundary for unmanaged training.")
    elif req.refresh_cost > 20_000 and req.quality < 0.9:
        choice = "retrieve"
        reasons.append(
            f"Fine-tune refresh ${req.refresh_cost:,.0f}/mo with quality floor {req.quality:.2f} "
            "→ start with a retrieval baseline."
        )
    elif req.chunk_count < 5 and req.primary_source not in {"", "logs"}:
        choice = "retrieve"
        reasons.append("Too few structured chunks for a reliable fine-tune dataset → retrieval baseline first.")
    else:
        choice = "fine-tune"
        reasons.append(
            "Stable knowledge, action not required, rights clear, and cost/quality profile supports PEFT ownership."
        )

    # Rough monthly cost sketch for the executive card
    retrieve_cost = 2_500 + max(req.chunk_count, 1) * 0.15
    tool_cost = 4_000 if req.action == "yes" else 1_500
    finetune_cost = req.refresh_cost + 3_500
    costs = {
        "retrieve": round(retrieve_cost, 0),
        "tool": round(tool_cost, 0),
        "fine-tune": round(finetune_cost, 0),
    }
    selected_cost = costs[choice]
    unlock_index = choice == "retrieve"
    next_track = {
        "retrieve": "Unlock Agent 3b · Index builder (BM25 + dense + code graph) on Step 02 chunks.",
        "fine-tune": "Proceed to PEFT / L1–L7 training track after registering an adaptation decision card.",
        "tool": "Wire MCP tools and human approval gates; skip corpus indexing for this capability.",
    }[choice]

    card_id = "adc-" + hashlib.sha1(
        f"{req.capability}|{choice}|{req.freshness}|{req.quality}|{req.dataset_version}".encode()
    ).hexdigest()[:10]

    return {
        "choice": choice,
        "choice_label": {"retrieve": "Retrieval / RAG", "fine-tune": "Fine-tune (PEFT)", "tool": "Tool / workflow"}[choice],
        "reasons": reasons,
        "blockers": blockers,
        "costs_monthly_usd": costs,
        "selected_cost_monthly_usd": selected_cost,
        "quality_floor": req.quality,
        "freshness": req.freshness,
        "action_required": req.action == "yes",
        "unlock_index_builder": unlock_index,
        "decision_card": {
            "id": card_id,
            "capability": req.capability,
            "choice": choice,
            "dataset_version": req.dataset_version,
            "primary_source": req.primary_source,
            "chunk_count": req.chunk_count,
            "template_count": req.template_count,
        },
        "next": next_track,
        "formula": (
            f"choice={choice} · freshness={req.freshness} · action={req.action} · "
            f"Qmin={req.quality:.2f} · refresh=${req.refresh_cost:,.0f}/mo · "
            f"selected≈${selected_cost:,.0f}/mo"
        ),
    }


@adapt_api.post("/lexora/adapt/decide", tags=["LEXORA adapt"])
def adapt_decide(request: AdaptDecideRequest):
    _require_dev()
    return _decide(request)


@adapt_api.post("/lexora/adapt/index/build", tags=["LEXORA adapt"])
def adapt_index_build(request: IndexBuildRequest):
    _require_dev()
    if request.decision != "retrieve":
        raise HTTPException(
            status_code=400,
            detail="Index builder (Agent 3b) unlocks only when Adapt selects Retrieval / RAG.",
        )
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks from Step 02. Complete chunking before building indexes.")

    t0 = time.perf_counter()
    # Educational in-process build summary over handoff chunks (full HelixIndex remains in bootstrap).
    bm25_docs = len(request.chunks) if request.build_bm25 else 0
    dense_vecs = len(request.chunks) if request.build_dense else 0
    graph_nodes = 0
    graph_edges = 0
    if request.build_graph:
        symbols = {str(c.get("symbol") or c.get("id") or i) for i, c in enumerate(request.chunks)}
        graph_nodes = len(symbols)
        for c in request.chunks:
            calls = c.get("calls") or []
            imports = c.get("imports") or []
            graph_edges += len(calls) + min(len(imports), 3)

    digest = hashlib.sha1(
        (request.dataset_version + "|" + str(len(request.chunks)) + "|" + request.primary_source).encode()
    ).hexdigest()[:8]
    index_version = f"idx-{digest}"
    seconds = round(time.perf_counter() - t0, 3)
    return {
        "index_version": index_version,
        "dataset_version": request.dataset_version,
        "primary_source": request.primary_source,
        "chunks_indexed": len(request.chunks),
        "stages": [
            {"id": "bm25", "label": "BM25 lexical", "status": "done" if request.build_bm25 else "skipped", "count": bm25_docs},
            {"id": "dense", "label": "Dense vectors", "status": "done" if request.build_dense else "skipped", "count": dense_vecs},
            {
                "id": "graph",
                "label": "Code graph",
                "status": "done" if request.build_graph else "skipped",
                "count": graph_nodes,
                "edges": graph_edges,
            },
        ],
        "bm25_documents": bm25_docs,
        "dense_vectors": dense_vecs,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "seconds": seconds,
        "next": "Hybrid index ready for Code RAG (RRF → rerank → cite). Proceed to NEXUS retrieval labs when online.",
    }
