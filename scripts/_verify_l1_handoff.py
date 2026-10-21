"""Smoke: Agent 01 tickets handoff → L1 labelled rows (no silent CSV)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexora.ingest.collection_agent import _build_handoff_text, _simulation_documents
from nexus.api.l1_lab import _SECTION, _clean_ticket_row, _parse_kv_block, _tickets_from_handoff_text


def main() -> None:
    docs = _simulation_documents(["tickets"])
    text, n = _build_handoff_text(docs, max_docs=240)
    rows = _tickets_from_handoff_text(text)
    matches = list(_SECTION.finditer(text))
    fail = 0
    dups = 0
    seen: set[str] = set()
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields = _parse_kv_block(text[start:end])
        row = {
            "title": fields.get("title") or "",
            "description": fields.get("description") or "",
            "category": fields.get("category") or "",
            "assignment_group": fields.get("assignment_group") or "",
            "priority": fields.get("priority") or "",
            "split": fields.get("split") or "",
        }
        cleaned = _clean_ticket_row(row)
        if not cleaned:
            fail += 1
            continue
        key = "|".join(
            [
                cleaned["title"],
                cleaned["category"],
                cleaned["assignment_group"],
                cleaned["priority"],
            ]
        )
        if key in seen:
            dups += 1
        seen.add(key)
    print(f"docs={len(docs)} handoff_docs={n} chars={len(text)}")
    print(f"sections={len(matches)} fail={fail} dups={dups} unique={len(seen)} parsed={len(rows)}")
    if rows:
        print("sample:", rows[0])


if __name__ == "__main__":
    main()
