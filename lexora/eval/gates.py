"""Release gates: an artefact is promoted to NEXUS only if every gated metric passes (configs/helix.yaml)."""
from __future__ import annotations

from common.config import cfg


def check(model_name: str, scorecard: dict[str, float]) -> dict:
    gates = cfg()["release_gates"].get(model_name, {})
    results = {}
    for metric, (op, target) in gates.items():
        if metric not in scorecard:
            results[metric] = {"op": op, "target": target, "value": None, "pass": None, "note": "not measured in MVP"}
            continue
        v = scorecard[metric]
        ok = v >= target if op == ">=" else v <= target
        results[metric] = {"op": op, "target": target, "value": round(float(v), 4), "pass": bool(ok)}
    measured = [r["pass"] for r in results.values() if r["pass"] is not None]
    return {"gates": results, "passed": bool(measured) and all(measured)}
