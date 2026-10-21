"""Dev-only API for the LEXORA parse / chunk / template agent (learning portal step 2)."""
from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings
from lexora.ingest.base import Document
from lexora.prep.chunking import chunk_document
from lexora.prep.logs import mine_templates

_HANDOFF_SPLIT = re.compile(r"(?m)^===== (.+?) =====\s*$")

chunking_api = APIRouter()
log = logging.getLogger("nexus.api.chunking")


class ChunkRequest(BaseModel):
    code: str = Field(default="", max_length=150_000)
    path: str = Field(default="src/checkout/validators.py", max_length=260)
    method: Literal["tree-sitter", "ast", "token-window", "heading"] = "tree-sitter"
    kind: Literal["code", "doc", "ticket", "log"] = "code"
    overlap: float = Field(default=0.15, ge=0.0, le=0.5)
    max_tokens: int = Field(default=600, ge=100, le=2000)
    repo: str = Field(default="payments/checkout-service", max_length=200)


class TemplateRequest(BaseModel):
    logs: str = Field(default="", max_length=150_000)
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


def _split_handoff_sections(text: str) -> list[tuple[str, str]]:
    """Split Agent 01 multi-doc handoff (`===== path =====`) into per-file sections."""
    matches = list(_HANDOFF_SPLIT.finditer(text))
    if len(matches) < 2:
        return []
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


def _chunk_one(doc: Document, method: str, max_tokens: int, overlap: float) -> list:
    if method == "token-window":
        return _window_chunks(doc, max_tokens, overlap)
    if method == "heading" or doc.kind == "doc":
        doc.kind = "doc"
        return chunk_document(doc, max_tokens=max_tokens, overlap=overlap)
    if doc.kind in {"ticket", "log"}:
        return chunk_document(doc)
    return chunk_document(doc)


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

    sections = _split_handoff_sections(code)
    chunks = []
    if sections:
        for path, body in sections:
            section_kind = kind
            if path.endswith(".md"):
                section_kind = "doc"
            elif path.endswith(".py"):
                section_kind = "code"
            elif path.endswith(".log"):
                section_kind = "log"
            elif path.endswith(".json") and "ticket" in path.lower():
                section_kind = "ticket"
            doc = Document(
                source="lab",
                path=path,
                text=body,
                kind=section_kind,
                repo=request.repo,
                meta={"language": "python" if path.endswith(".py") else "", "handoff_section": True},
            )
            # Code bundles parse poorly as one AST; prefer structure per file, else windows.
            method = request.method
            if section_kind == "code" and method in {"tree-sitter", "ast"} and "=====" in request.code:
                method = "tree-sitter"
            chunks.extend(_chunk_one(doc, method, request.max_tokens, request.overlap))
    else:
        doc = Document(
            source="lab",
            path=request.path,
            text=code,
            kind=kind,
            repo=request.repo,
            meta={"language": "python" if request.path.endswith(".py") else ""},
        )
        chunks = _chunk_one(doc, request.method, request.max_tokens, request.overlap)
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
    # Agent 01 multi-doc handoff uses ===== path ===== markers — drop those before Drain.
    raw = _HANDOFF_SPLIT.sub("", request.logs)
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("=====")]
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
