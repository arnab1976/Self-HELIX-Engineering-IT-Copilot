"""L8 duplicate-ticket detector: L2 embeddings + nearest neighbours + logistic threshold.

p(duplicate) = sigmoid(w0 + w1*cos + w2*same_ci + w3*same_requester + w4*gap_hours/24)
The threshold is chosen on validation pairs so that precision >= 0.90 (release gate).

    python -m lexora.train.l8_duplicates
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from common.config import cfg, settings
from common.embeddings import get_embedder
from lexora.eval.gates import check
from lexora.eval.metrics import precision_recall_at_threshold, threshold_for_precision
from lexora.ingest.itsm_connector import load_tickets_csv
from lexora.registry import ModelCard, register

FEATURES = ["cosine", "same_ci", "same_requester", "gap_days"]


def pair_features(a: dict, b: dict, va: np.ndarray, vb: np.ndarray) -> list[float]:
    gap = abs((datetime.fromisoformat(a["created_at"]) - datetime.fromisoformat(b["created_at"])).total_seconds()) / 86400
    return [float(va @ vb), float(a["ci"] == b["ci"]), float(a["requester"] == b["requester"]), min(gap, 30.0)]


def main() -> dict:
    tickets = {t["id"]: t for t in load_tickets_csv()}
    with open(Path(settings().data_dir) / "generated" / "duplicate_pairs.csv", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    emb = get_embedder()
    ids = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vecs = dict(zip(ids, emb.embed([f"{tickets[i]['title']} {tickets[i]['description']}" for i in ids])))
    X = np.array([pair_features(tickets[p["a"]], tickets[p["b"]], vecs[p["a"]], vecs[p["b"]]) for p in pairs])
    y = np.array([int(p["label"]) for p in pairs])
    n = len(y)
    tr, va, te = slice(0, int(0.6 * n)), slice(int(0.6 * n), int(0.8 * n)), slice(int(0.8 * n), n)
    clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    target = cfg()["release_gates"]["lexora/duplicates"]["precision"][1]
    th = threshold_for_precision(clf.predict_proba(X[va])[:, 1], y[va], target)
    prec, rec = precision_recall_at_threshold(clf.predict_proba(X[te])[:, 1], y[te], th)
    out = Path(settings().state_dir) / "artifacts" / "l8_duplicates"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "duplicates.joblib"
    joblib.dump({"model": clf, "features": FEATURES, "threshold": th, "embedder": emb.name}, path)
    scorecard = {"precision": prec, "recall": rec, "threshold": th}
    gate = check("lexora/duplicates", scorecard)
    weights = dict(zip(["w0", *FEATURES], [float(clf.intercept_[0]), *map(float, clf.coef_[0])]))
    card = register(ModelCard(
        name="lexora/duplicates", version="1.1.0", artefact="L8", task_tags=["duplicate-detection"], base=emb.name,
        algorithm="embedding cosine + metadata -> logistic regression", artifact_path=str(path), scorecard=scorecard,
        gate=gate, params={"threshold": th, "weights": weights}, latency_profile={"p95_ms": 25}))
    print(json.dumps({"card": card.ref, "scorecard": scorecard, "weights": weights, "passed": gate["passed"]}, indent=1))
    return scorecard


if __name__ == "__main__":
    main()
