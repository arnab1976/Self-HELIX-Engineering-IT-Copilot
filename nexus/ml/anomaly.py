"""Incident signal scoring: rolling z-score anomalies and change-overlap correlation (spec 7.1 step 6).

z_t = (x_t - mean(window)) / std(window); anomaly when |z| > 3.
overlap(change) = exp(-|t_anomaly - t_change| / tau)
score = 0.5*overlap + 0.3*new_template_evidence + 0.2*latency_evidence ; confidence = softmax(4*score)
Production adds STL residuals / Isolation Forest (statsmodels, PyOD).
"""
from __future__ import annotations

import math

import numpy as np


def first_anomaly(series: list[float], window: int = 20, threshold: float = 3.0) -> tuple[int | None, list[dict]]:
    x = np.asarray(series, float)
    detail = []
    for i in range(window, len(x)):
        w = x[i - window : i]
        mu, sd = w.mean(), w.std() or 1e-9
        z = (x[i] - mu) / sd
        detail.append({"i": i, "x": round(float(x[i]), 3), "mu": round(float(mu), 3), "sd": round(float(sd), 3), "z": round(float(z), 2)})
        if abs(z) > threshold:
            return i, detail
    return None, detail


def rank_changes(anomaly_minute: int, changes: list[dict], tau: float = 5.0) -> list[dict]:
    """changes: [{label, minute, template_evidence (0-1), latency_evidence (0-1), evidence: [...]}]"""
    scored = []
    for c in changes:
        ov = math.exp(-abs(anomaly_minute - c["minute"]) / tau) if c["minute"] <= anomaly_minute else 0.0
        s = 0.5 * ov + 0.3 * c.get("template_evidence", 0) + 0.2 * c.get("latency_evidence", 0)
        scored.append({**c, "overlap": round(ov, 3), "score": round(s, 3)})
    e = np.exp([4 * c["score"] for c in scored])
    for c, v in zip(scored, e / e.sum()):
        c["confidence"] = round(float(v), 2)
    return sorted(scored, key=lambda c: -c["confidence"])
