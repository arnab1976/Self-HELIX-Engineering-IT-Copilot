"""L9 change-risk at inference: probability + per-feature contributions (TreeSHAP via LightGBM pred_contrib)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from common.config import cfg
from lexora.registry import get_card
from lexora.train.l9_change_risk import FEATURES, featurise

LABELS = {"services_touched": "Services touched", "log10_lines_changed": "Lines changed", "test_coverage": "Test coverage",
          "author_failure_rate": "Author change-failure history", "dependency_fanout": "Dependency fan-out",
          "peak_window": "Peak / Friday window", "has_rollback_plan": "Rollback plan", "is_standard": "Standard change",
          "is_emergency": "Emergency change"}


@lru_cache
def _bundle():
    card = get_card("lexora/change-risk")
    if card is None:
        raise RuntimeError("L9 change-risk model not registered: run `make bootstrap`")
    b = joblib.load(card.artifact_path)
    b["ref"] = card.ref
    return b


def score(change: dict) -> dict:
    b = _bundle()
    x = np.array([featurise(change)])
    p = float(b["model"].predict_proba(x)[0, 1])
    if b["backend"] == "lightgbm":
        contrib = b["model"].predict(x, pred_contrib=True)[0]      # exact TreeSHAP in log-odds; last = base value
        phis, base = contrib[:-1], float(contrib[-1])
        method = "TreeSHAP (exact)"
    else:  # occlusion: replace each feature with its training median
        med = np.array(b["medians"])
        lo = lambda v: float(np.log(max(1e-9, v) / max(1e-9, 1 - v)))  # noqa: E731
        z = lo(p)
        phis = []
        for i in range(len(FEATURES)):
            xx = x.copy()
            xx[0, i] = med[i]
            phis.append(z - lo(float(b["model"].predict_proba(xx)[0, 1])))
        base, method = z - sum(phis), "occlusion (approximate)"
    bands = cfg()["change_risk"]["bands"]
    band = "low" if p < bands["low"] else "medium" if p < bands["medium"] else "high"
    factors = sorted([{"feature": LABELS[f], "value": float(x[0, i]), "contribution": round(float(phis[i]), 3)}
                      for i, f in enumerate(FEATURES)], key=lambda d: -abs(d["contribution"]))
    up = [f for f in factors if f["contribution"] > 0][:3]
    down = [f for f in factors if f["contribution"] < 0][:2]
    rec = {"high": "Full CAB review; attach a tested rollback plan, move out of the peak window and canary at 5%.",
           "medium": "Peer review plus canary deploy; proceed if a rollback plan is attached.",
           "low": "Proceed as a standard change with automated checks."}[band]
    brief = (f"Failure risk {p:.1%} ({band}). Main drivers: " + (", ".join(f"{f['feature'].lower()} (+{f['contribution']})" for f in up) or "none")
             + ". Mitigating: " + (", ".join(f"{f['feature'].lower()} ({f['contribution']})" for f in down) or "none") + f". Recommendation: {rec}")
    return {"probability": round(p, 4), "band": band, "base_log_odds": round(base, 3), "factors": factors, "method": method,
            "cab_brief": brief, "model": b["ref"]}
