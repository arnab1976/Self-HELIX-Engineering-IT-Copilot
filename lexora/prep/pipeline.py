"""LEXORA data pipeline (spec 4.3): collect -> secrets -> PII -> licence -> dedupe -> chunk -> FIM -> register.

    python -m lexora.prep.pipeline            # uses data/sample via lexora.ingest.local
Outputs var/datasets/<version>/{chunks.jsonl, fim_train.jsonl, scan_report.json, manifest.json}
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from common.config import cfg, repo_class, settings
from common.logging import get_logger, log_event
from lexora.ingest.base import Document
from lexora.ingest.local import load_all
from lexora.prep import dedupe, fim, licence, pii, secrets
from lexora.prep.chunking import Chunk, chunk_document

log = get_logger("lexora.prep")


def sanitise(doc: Document) -> tuple[Document | None, dict]:
    report = {"id": doc.id, "secrets": [], "pii": 0, "licence": None, "action": "keep"}
    text, found = secrets.redact(doc.text)
    report["secrets"] = [{"type": f.type, "value": f.masked(), "entropy": round(f.entropy, 2), "confidence": f.confidence,
                          "placeholder": f.placeholder} for f in found]
    if secrets.scan(text):                       # anything left after redaction -> drop the file
        report["action"] = "dropped: unresolved secret"
        return None, report
    text, pfound = pii.anonymise(text)
    report["pii"] = len(pfound)
    lic = licence.detect(doc.text, doc.path)
    report["licence"] = lic.licence
    report["copyleft"] = lic.copyleft
    clean = Document(doc.source, doc.path, text, doc.kind, doc.repo, {**doc.meta, "licence": lic.licence, "copyleft": lic.copyleft})
    return clean, report


def run(docs: list[Document] | None = None, out_root: Path | None = None, dedupe_threshold: float = 0.8,
        fim_family: str = "qwen", seed: int = 7) -> dict:
    t0 = time.time()
    docs = docs if docs is not None else load_all()
    reports, clean_docs = [], []
    for d in docs:
        c, rep = sanitise(d)
        reports.append(rep)
        if c:
            clean_docs.append(c)

    code = [d for d in clean_docs if d.kind == "code"]
    kept, dropped = dedupe.deduplicate([(d.id, d.text) for d in code], threshold=dedupe_threshold)
    for r in reports:
        if r["id"] in dropped:
            r["action"] = f"dropped: near-duplicate of {dropped[r['id']][0]} (J~{dropped[r['id']][1]:.2f})"
    keep_ids = set(kept) | {d.id for d in clean_docs if d.kind != "code"}
    final = [d for d in clean_docs if d.id in keep_ids]

    chunks: list[Chunk] = []
    rc = cfg()["rag"]
    for d in final:
        for ch in chunk_document(d, **({"max_tokens": rc["doc_chunk_tokens"], "overlap": rc["doc_chunk_overlap"]} if d.kind == "doc" else {})):
            ch.licence = d.meta.get("licence")
            ch.repo_class = repo_class(d.repo) or "internal"
            ch.meta["source"] = d.source
            chunks.append(ch)

    rng = random.Random(seed)
    fim_samples = [fim.make_sample(d.text, rng, fim_family) for d in final if d.kind == "code" and not d.meta.get("copyleft")]

    version = "ds-eng-" + hashlib.sha1("".join(sorted(c.id + c.sha for c in chunks)).encode()).hexdigest()[:8]
    out = (out_root or Path(settings().state_dir) / "datasets") / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c.to_dict()) for c in chunks) + "\n", encoding="utf-8")
    (out / "fim_train.jsonl").write_text("\n".join(json.dumps({"text": s.text, "mode": s.mode}) for s in fim_samples) + "\n", encoding="utf-8")
    (out / "scan_report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    manifest = {
        "version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents_in": len(docs), "documents_out": len(final), "chunks": len(chunks), "fim_samples": len(fim_samples),
        "secrets_redacted": sum(len(r["secrets"]) for r in reports), "pii_masked": sum(r["pii"] for r in reports),
        "copyleft_excluded_from_fim": sum(1 for d in final if d.meta.get("copyleft")),
        "near_duplicates_dropped": len(dropped),
        "source_mix": dict(Counter(d.source for d in final)), "licences": dict(Counter(str(d.meta.get("licence")) for d in final)),
        "seconds": round(time.time() - t0, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    latest = (out_root or Path(settings().state_dir) / "datasets") / "LATEST"
    latest.write_text(version, encoding="utf-8")
    log_event(log, "dataset registered", **manifest)
    return manifest


def latest_chunks() -> list[Chunk]:
    root = Path(settings().state_dir) / "datasets"
    version = (root / "LATEST").read_text().strip()
    rows = [json.loads(ln) for ln in (root / version / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in rows:
        r.pop("id", None)
    return [Chunk(**r) for r in rows]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
