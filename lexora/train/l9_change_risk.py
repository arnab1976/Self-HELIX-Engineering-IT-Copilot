"""L9 change-risk model: gradient-boosted trees + TreeSHAP explanations, time-based split.

LightGBM is used when installed (`predict(..., pred_contrib=True)` returns exact TreeSHAP values, so the
explanation needs no extra package). Fallback: scikit-learn HistGradientBoosting with occlusion-style
contributions (marked approximate in the model card).

    python -m lexora.train.l9_change_risk
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np

from common.config import load_yaml, settings
from lexora.eval.gates import check
from lexora.eval.metrics import roc_auc
from lexora.registry import ModelCard, register

FEATURES = ["services_touched", "log10_lines_changed", "test_coverage", "author_failure_rate", "dependency_fanout",
            "peak_window", "has_rollback_plan", "is_standard", "is_emergency"]


def featurise(r: dict) -> list[float]:
    return [float(r["services_touched"]), math.log10(max(1.0, float(r["lines_changed"]))), float(r["test_coverage"]),
            float(r["author_failure_rate"]), float(r["dependency_fanout"]), float(r["peak_window"]),
            float(r["has_rollback_plan"]), float(r["change_type"] == "standard"), float(r["change_type"] == "emergency")]


def main() -> dict:
    with open(Path(settings().data_dir) / "generated" / "changes.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["created_at"])
    X = np.array([featurise(r) for r in rows])
    y = np.array([int(r["failed"]) for r in rows])
    cut = int(len(y) * (1 - load_yaml("configs/train/recipes.yaml")["l9_change_risk"]["holdout_fraction"]))
    backend = "lightgbm"
    try:
        import lightgbm as lgb  # type: ignore

        params = load_yaml("configs/train/recipes.yaml")["l9_change_risk"]["params"]
        model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "objective"}, objective="binary", verbose=-1)
        model.fit(X[:cut], y[:cut])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        backend = "sklearn-hgb"
        model = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.05, max_iter=300).fit(X[:cut], y[:cut])
    p = model.predict_proba(X[cut:])[:, 1]
    auc = roc_auc(p, y[cut:])
    order = np.argsort(-p)
    top_decile = order[: max(1, len(order) // 10)]
    lift = float(y[cut:][top_decile].mean() / max(1e-9, y[cut:].mean()))
    out = Path(settings().state_dir) / "artifacts" / "l9_change_risk"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "change_risk.joblib"
    joblib.dump({"model": model, "features": FEATURES, "backend": backend,
                 "medians": np.median(X[:cut], axis=0).tolist(), "base_rate": float(y[:cut].mean())}, path)
    scorecard = {"auc": auc, "top_decile_lift": lift, "base_rate": float(y.mean())}
    gate = check("lexora/change-risk", scorecard)
    card = register(ModelCard(
        name="lexora/change-risk", version="1.0.2", artefact="L9", task_tags=["change-risk"], base=backend,
        algorithm="gradient-boosted trees + TreeSHAP" if backend == "lightgbm" else "HGB + occlusion contributions (approx.)",
        artifact_path=str(path), scorecard=scorecard, gate=gate, params={"features": FEATURES, "split": "time-based"},
        known_failure_modes=["services with little change history", "drift after process changes"]))
    print(json.dumps({"card": card.ref, "scorecard": scorecard, "passed": gate["passed"], "backend": backend}, indent=1))
    return scorecard


if __name__ == "__main__":
    main()
