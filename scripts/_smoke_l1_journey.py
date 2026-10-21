"""Collect tickets via Agent 01, then L1 load must use journey_handoff (not CSV)."""
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
    collect = post(
        "/v1/lexora/collect/run",
        {
            "mode": "simulation",
            "sources": ["tickets"],
            "sample_text": "",
            "filename": "x.txt",
            "dedupe_threshold": 0.82,
        },
    )
    handoff = collect.get("handoff") or {}
    text = handoff.get("text") or ""
    print(
        "collect",
        collect.get("dataset_version"),
        "handoff_docs",
        handoff.get("handoff_documents"),
        "chars",
        len(text),
    )

    loaded = post(
        "/v1/lexora/l1/load",
        {
            "tickets": [],
            "chunks": [],
            "prefer_csv": False,
            "require_journey": True,
            "allow_lab_fallback": False,
            "dataset_version": collect.get("dataset_version") or "",
            "primary_source": "tickets",
            "source_label": "ITSM tickets",
            "adapt_choice": "fine-tune",
            "handoff_text": text[:150000],
        },
    )
    print(
        "l1 source=",
        loaded.get("source"),
        "count=",
        loaded.get("ticket_count"),
        "note=",
        loaded.get("note"),
    )
    assert loaded.get("source") == "journey_handoff", loaded
    assert loaded.get("ticket_count", 0) >= 12
    assert "240" not in str(loaded.get("source"))
    print("OK — L1 uses Agent 01 handoff")


if __name__ == "__main__":
    main()
