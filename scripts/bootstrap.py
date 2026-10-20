"""One-shot setup: LEXORA prep -> train CPU baselines (L1, L8, L9) -> dataset builders for GPU artefacts ->
build the NEXUS index -> register placeholder cards for GPU-served tiers -> seed open tickets.

    python scripts/bootstrap.py            (or: make bootstrap)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import settings  # noqa: E402
from common.store import store  # noqa: E402


def step(title: str):
    print(f"\n=== {title} ===", flush=True)


def seed_open_tickets(n: int = 40) -> int:
    rows = list(csv.DictReader(open(Path(settings().data_dir) / "generated" / "tickets.csv", encoding="utf-8")))
    recent = rows[-n:]
    now = time.time()
    s = store()
    s.execute("DELETE FROM tickets WHERE source='seed'")
    for i, r in enumerate(recent):
        s.execute("INSERT OR REPLACE INTO tickets(id,title,description,requester,ci,category,assignment_group,priority,state,created_at,source) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r["id"], r["title"], r["description"], r["requester"], r["ci"], r["category"],
                                                     r["assignment_group"], r["priority"], "In Progress", now - (n - i) * 3600, "seed"))
    # a known open VPN outage ticket so the duplicate flow is easy to demo
    s.execute("INSERT OR REPLACE INTO tickets(id,title,description,requester,ci,category,assignment_group,priority,state,created_at,source) "
              "VALUES ('INC0048211','GlobalProtect VPN drops every few minutes from home','VPN disconnects every 5 minutes after the Monday update, need to reconnect each time',"
              "'u.kiran','VPN-GW-02','Access > VPN','Network-L2','P3','In Progress',?, 'seed')", (now - 5400,))
    return n + 1


def main() -> int:
    t0 = time.time()
    from lexora.prep import pipeline

    step("LEXORA 1/5  data pipeline (secrets, PII, licence, dedupe, chunk, FIM)")
    m = pipeline.run()
    print(json.dumps({k: m[k] for k in ("version", "chunks", "secrets_redacted", "pii_masked", "near_duplicates_dropped", "fim_samples")}))

    step("LEXORA 2/5  train CPU artefacts: L1 triage, L8 duplicates, L9 change risk")
    from lexora.train import l1_triage, l8_duplicates, l9_change_risk

    l1_triage.main([])
    l8_duplicates.main()
    l9_change_risk.main()

    step("LEXORA 3/5  build GPU training datasets (dry runs): L2, L3, L4, L6, L7")
    from lexora.train import l2_embedder, l3_reranker, l4_fim, l6_text2query, l7_incident

    for mod in (l2_embedder, l3_reranker, l4_fim, l6_text2query, l7_incident):
        mod.main(["--dry-run"])

    step("LEXORA 4/5  register model cards for GPU-served tiers (MVP: mock / Ollama endpoints)")
    from lexora.eval.gates import check
    from lexora.registry import ModelCard, register

    for name, art, tags, base, score in [
        ("lexora/fim-coder", "L4", ["code-completion", "fim"], "Qwen2.5-Coder-1.5B + LoRA", {"p95_first_token_ms": 240}),
        ("lexora/code-review", "L5", ["code-qa", "pr-review", "iac-review"], "Qwen2.5-Coder-7B-Instruct + QLoRA", {"hallucinated_api_rate": 0.0}),
        ("lexora/text2query", "L6", ["text-to-sql", "text-to-promql"], "Qwen2.5-Coder-7B-Instruct + QLoRA", {"read_only_rate": 1.0}),
        ("lexora/incident", "L7", ["incident-brief", "postmortem"], "Llama-3.1-8B-Instruct + QLoRA", {}),
        ("lexora/retriever", "L2+L3", ["code-retrieval", "doc-retrieval", "rerank"], "hashing-768 + lexical rerank", {}),
    ]:
        register(ModelCard(name=name, version="0.1.0-mvp", artefact=art, task_tags=tags, base=base, algorithm="see configs/train/recipes.yaml",
                           artifact_path="(train on GPU: python -m lexora.train.*)", scorecard=score, gate=check(name, score)), promote_if_passed=True)

    step("NEXUS 5/5  build hybrid index (BM25 + vectors + code graph) and evaluate retrieval")
    from nexus.rag.index import get_index

    idx = get_index(rebuild=True)
    print(f"index {idx.version}: {len(idx.chunks)} chunks, embedder={idx.embedder_name}")
    from lexora.eval import retrieval_eval, text2query_eval

    r = retrieval_eval.main()
    t = text2query_eval.main()
    print(f"seeded {seed_open_tickets()} open tickets")
    print(f"\nbootstrap complete in {time.time() - t0:.1f}s  ->  make run   (portal: http://localhost:8000/portal)")
    return 0 if r is not None and t is not None else 1


if __name__ == "__main__":
    sys.exit(main())
