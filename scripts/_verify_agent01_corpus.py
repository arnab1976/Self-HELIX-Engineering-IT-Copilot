"""Verify Agent 01 multi-doc handoff yields enough Step 02 chunks per category."""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def main() -> None:
    methods = {
        "code": "tree-sitter",
        "tickets": "token-window",
        "knowledge": "heading",
        "logs": "token-window",
        "observability": "token-window",
    }
    for source, method in methods.items():
        collect = post(
            "/v1/lexora/collect/run",
            {
                "mode": "simulation",
                "sources": [source],
                "sample_text": "",
                "filename": "unused.txt",
                "dedupe_threshold": 0.82,
            },
        )
        text = (collect.get("handoff") or {}).get("text") or collect.get("preview") or ""
        handoff_docs = (collect.get("handoff") or {}).get("handoff_documents")
        kind = (collect.get("handoff") or {}).get("kind") or "doc"
        if kind not in {"code", "doc", "ticket", "log"}:
            kind = "doc"
        chunk = post(
            "/v1/lexora/chunk/run",
            {
                "code": text[:150000],
                "path": (collect.get("handoff") or {}).get("path") or f"{source}.txt",
                "kind": kind,
                "method": method,
                "overlap": 0.15,
                "max_tokens": 400,
            },
        )
        n = chunk.get("chunk_count") or 0
        print(
            f"{source:14} docs_in={collect.get('documents_in')} "
            f"handoff_docs={handoff_docs} chars={len(text)} "
            f"chunks={n} ok={n >= 5}"
        )


if __name__ == "__main__":
    main()
