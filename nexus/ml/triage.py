"""L1 triage at inference: category, assignment group, priority with calibrated probabilities (T0 tier)."""
from __future__ import annotations

import re
from functools import lru_cache

import joblib
import numpy as np

from lexora.registry import get_card
from lexora.train.calibration import softmax

HEADS = ["category", "assignment_group", "priority"]


class TriageModel:
    def __init__(self, path: str, version: str):
        self.b = joblib.load(path)
        self.version = version

    def predict(self, title: str, description: str = "", top_k: int = 3) -> dict:
        X = self.b["featuriser"].transform([f"{title} . {description}"])
        out = {}
        for h in HEADS:
            clf, T, classes = self.b["heads"][h], self.b["temperature"][h], self.b["classes"][h]
            raw = clf.predict_proba(X)[0]
            p = softmax(np.log(np.clip(raw, 1e-12, 1))[None, :], T)[0]
            order = np.argsort(-p)[:top_k]
            out[h] = {"label": str(classes[order[0]]), "p": round(float(p[order[0]]), 3),
                      "top": [{"label": str(classes[i]), "p": round(float(p[i]), 3)} for i in order],
                      "p_uncalibrated": round(float(raw[order[0]]), 3), "temperature": T}
        text = f"{title} {description}".lower()
        if re.search(r"outage|all users|production|customers|sev ?1", text) and out["priority"]["label"] not in ("P1", "P2"):
            out["priority"] = {**out["priority"], "label": "P2", "rule_override": "production impact keywords"}
        return out


@lru_cache
def get_triage() -> TriageModel:
    card = get_card("lexora/triage")
    if card is None:
        raise RuntimeError("L1 triage model not registered: run `make bootstrap`")
    return TriageModel(card.artifact_path, card.ref)


INTENTS = {
    "Access > Password/MFA": "password_mfa", "Access > VPN": "vpn", "Access > Request": "access_request",
    "Software > Install": "software_install", "Hardware > Laptop": "hardware", "Email & Collaboration": "email",
    "Application > Incident": "app_incident",
}


def intent_of(text: str) -> tuple[str, float, dict]:
    """Helpdesk intent = L1 category head mapped to a workflow branch (spec 7.2 step 1)."""
    pred = get_triage().predict(text)
    cat = pred["category"]
    return INTENTS.get(cat["label"], "other"), cat["p"], pred
