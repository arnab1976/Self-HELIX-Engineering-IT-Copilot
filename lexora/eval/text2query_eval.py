"""Text-to-query execution accuracy: a generated query is correct if its result set equals the gold result set.

    python -m lexora.eval.text2query_eval
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from common.config import settings
from lexora.eval.gates import check
from lexora.registry import get_card, register


def _rows(con: sqlite3.Connection, sql: str):
    return sorted(map(tuple, con.execute(sql).fetchall()))


def main() -> dict:
    from nexus.query.text2query import generate_query

    gold = [json.loads(ln) for ln in (Path(settings().data_dir) / "generated" / "gold" / "text2query.jsonl").read_text().splitlines() if ln]
    con = sqlite3.connect(f"file:{Path(settings().data_dir) / 'generated' / 'warehouse.db'}?mode=ro", uri=True)
    correct, read_only, details = 0, 0, []
    for g in gold:
        r = generate_query(g["q"], g["target"], execute=False)
        ok = False
        if r["validation"]["valid"]:
            read_only += 1
            try:
                ok = _rows(con, r["query"]) == _rows(con, g["gold"])
            except sqlite3.Error:
                ok = False
        correct += ok
        details.append({"q": g["q"], "ok": ok})
    sc = {"execution_accuracy": correct / len(gold), "read_only_rate": read_only / len(gold), "queries": len(gold)}
    gate = check("lexora/text2query", sc)
    card = get_card("lexora/text2query")
    if card:
        card.scorecard, card.gate, card.version = sc, gate, "0.1.1-mvp"
        register(card)
    print(json.dumps({"text2query": sc, "passed": gate["passed"], "misses": [d["q"] for d in details if not d["ok"]]}, indent=1))
    return sc


if __name__ == "__main__":
    main()
