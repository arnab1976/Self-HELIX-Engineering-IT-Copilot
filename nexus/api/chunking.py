"""Dev-only API for the LEXORA parse / chunk / template agent (learning portal step 2)."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings
from lexora.ingest.base import Document
from lexora.prep.chunking import chunk_document
from lexora.prep.logs import mine_templates

chunking_api = APIRouter()
log = logging.getLogger("nexus.api.chunking")


class ChunkRequest(BaseModel):
    code: str = Field(default="", max_length=80_000)
    path: str = Field(default="src/checkout/validators.py", max_length=260)
    method: Literal["tree-sitter", "ast", "token-window", "heading"] = "tree-sitter"
    kind: Literal["code", "doc", "ticket", "log"] = "code"
    overlap: float = Field(default=0.15, ge=0.0, le=0.5)
    max_tokens: int = Field(default=600, ge=100, le=2000)
    repo: str = Field(default="payments/checkout-service", max_length=200)


class TemplateRequest(BaseModel):
    logs: str = Field(default="", max_length=80_000)
    sim_th: float = Field(default=0.5, ge=0.1, le=1.0)


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


def _window_chunks(doc: Document, max_tokens: int, overlap: float) -> list:
    """Fixed token-window fallback used when the learner selects token-window method."""
    from lexora.prep.chunking import Chunk
    from common.textutils import estimate_tokens

    text = doc.text
    if not text.strip():
        return []
    chars = max(max_tokens * 4, 80)
    step = max(int(chars * (1 - overlap)), 40)
    out = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chars]
        if not piece.strip():
            continue
        line_start = text.count("\n", 0, start) + 1
        line_end = line_start + piece.count("\n")
        out.append(
            Chunk(
                doc.repo or "",
                doc.path,
                line_start,
                line_end,
                f"{doc.path}#window-{i + 1}",
                "module",
                piece,
                doc.sha,
                meta={"method": "token-window", "part": i},
            )
        )
    return out


@chunking_api.post("/lexora/chunk/run", tags=["LEXORA chunking"])
def run_chunk(request: ChunkRequest):
    _require_dev()
    code = request.code.strip("\n")
    if not code.strip():
        raise HTTPException(status_code=400, detail="Paste Step 01 sanitised output before chunking.")
    kind = request.kind
    if request.path.endswith(".md"):
        kind = "doc"
    elif request.path.endswith(".py") and kind == "code":
        kind = "code"
    doc = Document(
        source="lab",
        path=request.path,
        text=code,
        kind=kind,
        repo=request.repo,
        meta={"language": "python" if request.path.endswith(".py") else ""},
    )
    if request.method == "token-window":
        chunks = _window_chunks(doc, request.max_tokens, request.overlap)
    elif request.method == "heading" or kind == "doc":
        doc.kind = "doc"
        chunks = chunk_document(doc, max_tokens=request.max_tokens, overlap=request.overlap)
    elif kind in {"ticket", "log"}:
        chunks = chunk_document(doc)
    else:
        chunks = chunk_document(doc)
    payload = []
    for ch in chunks:
        kind_label = {
            "class": "CLASS HEADER",
            "method": "METHOD",
            "function": "FUNCTION",
            "module": "MODULE",
            "doc": "DOCUMENT",
            "ticket": "TICKET",
            "log": "LOG",
        }.get(ch.kind, ch.kind.upper())
        payload.append(
            {
                "id": ch.id,
                "symbol": ch.symbol,
                "kind": ch.kind,
                "kind_label": kind_label,
                "path": ch.path,
                "start": ch.start,
                "end": ch.end,
                "sha": ch.sha,
                "tokens": ch.tokens,
                "imports": ch.imports,
                "citation": ch.citation(),
                "text": ch.text,
            }
        )
    return {
        "method": request.method,
        "path": request.path,
        "overlap": request.overlap,
        "chunks": payload,
        "chunk_count": len(payload),
        "total_tokens": sum(c["tokens"] for c in payload),
        "next": "Chunks are ready for indexing (BM25 + dense + code graph).",
    }


@chunking_api.post("/lexora/templates/mine", tags=["LEXORA chunking"])
def run_templates(request: TemplateRequest):
    _require_dev()
    lines = [ln for ln in request.logs.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Paste raw log lines before mining templates.")
    templates = mine_templates(lines, sim_th=request.sim_th)
    compression = (len(lines) / len(templates)) if templates else 0.0
    return {
        "raw_lines": len(lines),
        "template_count": len(templates),
        "compression": round(compression, 1),
        "sim_th": request.sim_th,
        "templates": [
            {"id": f"T{i}", "cluster_id": t["id"], "template": t["template"], "count": t["count"]}
            for i, t in enumerate(templates, 1)
        ],
        "next": "Templates compress log volume for incident correlation and L7 summarisation.",
    }
