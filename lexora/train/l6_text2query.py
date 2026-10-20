"""L6 text-to-query adapter: QLoRA SFT on (question, schema, query) pairs, kept only if the query executes.

    python -m lexora.train.l6_text2query --dry-run      # builds + execution-filters the pairs
    python -m lexora.train.l6_text2query                # GPU
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from common.config import load_yaml, settings
from lexora.train.qlora import write_jsonl


def schema_text() -> str:
    cat = load_yaml("configs/metric_catalog.yaml")["sql"]["tables"]
    return "\n".join(f"{t}({', '.join(f'{c} {ty}' for c, ty in v['columns'].items())})" for t, v in cat.items())


def execution_filter(rows: list[dict]) -> list[dict]:
    db = Path(settings().data_dir) / "generated" / "warehouse.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    kept = []
    for r in rows:
        try:
            con.execute(r["gold"]).fetchmany(5)
            kept.append(r)
        except sqlite3.Error:
            pass
    con.close()
    return kept


def build_rows() -> list[dict]:
    gold = [json.loads(ln) for ln in (Path(settings().data_dir) / "generated" / "gold" / "text2query.jsonl").read_text().splitlines() if ln]
    kept = execution_filter([g for g in gold if g["target"] == "sql"])
    sch = schema_text()
    return [{"messages": [
        {"role": "system", "content": "Write one read-only SQLite query. Use only these tables:\n" + sch},
        {"role": "user", "content": g["q"]}, {"role": "assistant", "content": g["gold"]}]} for g in kept]


def main(argv=None):
    rec = load_yaml("configs/train/recipes.yaml")["l6_text2query"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=rec["base_model"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = build_rows()
    p = write_jsonl(Path(settings().state_dir) / "datasets" / "l6_sft.jsonl", rows)
    print(f"L6 rows after execution filter: {len(rows)} -> {p}")
    if a.dry_run:
        return rows
    from lexora.train.qlora import sft  # pragma: no cover

    sft(a.base, p, Path(settings().state_dir) / "artifacts" / "l6_text2query", rec["lora"]["r"], rec["lora"]["alpha"],
        rec["learning_rate"], rec["epochs"])
    return rows


if __name__ == "__main__":
    main()
