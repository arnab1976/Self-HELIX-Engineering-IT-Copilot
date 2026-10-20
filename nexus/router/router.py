"""Router (spec 6.1): choose the cheapest tier that meets the quality bar, inside what the policy allows.

    m* = argmax_{m in allowed} Q(m, task) - lc * cost(m)/cost_norm - ll * latency(m)/latency_norm
         subject to Q(m, task) >= Q_min(task) and context <= window(m)
Completion is a fixed route to T1a (no time to score). Restricted repositories never reach external tiers,
because the policy gate removes them before scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.config import cfg
from nexus.router.policy import PolicyInput, policy_gate


@dataclass
class Candidate:
    tier: str
    quality: float | None
    cost: float
    latency_ms: float
    score: float | None
    status: str


@dataclass
class RouteDecision:
    route: str
    tier: str | None
    candidates: list[Candidate] = field(default_factory=list)
    fallback: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {"route": self.route, "tier": self.tier, "fallback": self.fallback, "reason": self.reason,
                "candidates": [c.__dict__ for c in self.candidates]}


class Router:
    def route(self, route: str, roles: list[str], repo_class: str | None = None, complexity: float = 0.3,
              context_tokens: int = 2000, unavailable: set[str] | None = None) -> RouteDecision:
        c = cfg()
        rc = c["router"]
        spec = rc["routes"][route]
        tiers = c["tiers"]
        unavailable = unavailable or set()
        cands: list[Candidate] = []
        for t in spec["candidates"]:
            ti = tiers[t]
            base_q = rc["quality"].get(route, {}).get(t, 0.7)
            q = base_q - (0.03 if ti.get("external") else 0.15) * complexity
            cand = Candidate(t, round(q, 3), ti["cost_per_req"], ti["latency_ms"], None, "eligible")
            pol = policy_gate.evaluate(PolicyInput(roles=roles, repo_class=repo_class, target_tier=t))
            if not pol.allow:
                cand.status = "denied: " + "; ".join(pol.reasons)
            elif t in unavailable:
                cand.status = "unavailable"
            elif context_tokens > ti["context"]:
                cand.status = f"context {context_tokens} > window {ti['context']}"
            elif spec.get("fixed"):
                cand.score, cand.status = 1.0, "fixed route"
            elif q < spec["q_min"]:
                cand.status = f"below Q_min {spec['q_min']}"
            else:
                cand.score = round(q - rc["lambda_cost"] * ti["cost_per_req"] / rc["cost_norm"]
                                   - rc["lambda_latency"] * ti["latency_ms"] / rc["latency_norm"], 4)
            cands.append(cand)
        ok = sorted([x for x in cands if x.score is not None], key=lambda x: -x.score)
        if not ok:
            return RouteDecision(route, None, cands, spec["fallback"], "no tier satisfies policy, quality and context")
        ok[0].status = "selected"
        return RouteDecision(route, ok[0].tier, cands, spec["fallback"], f"highest score {ok[0].score}")


router = Router()
