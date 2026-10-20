"""Policy gate. Queries OPA (configs/policy/helix.rego) when HELIX_OPA_URL is set; otherwise evaluates the
same rules in Python so behaviour is identical in the laptop MVP."""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from common.config import cfg, settings

WRITE_TOOLS = {"runbook.execute", "idp.reset_mfa", "idp.reset_password", "itsm.request_access", "git.open_pr"}


@dataclass
class PolicyInput:
    roles: list[str]
    repo_class: str | None = None
    target_tier: str | None = None
    tool: str | None = None
    has_approval: bool = False
    injection_score: float = 0.0
    secret_findings: int = 0


@dataclass
class PolicyDecision:
    allow: bool
    reasons: list[str] = field(default_factory=list)
    engine: str = "python"


class PolicyGate:
    def evaluate(self, inp: PolicyInput) -> PolicyDecision:
        if settings().opa_url:
            try:
                r = httpx.post(f"{settings().opa_url}/v1/data/helix/gateway/decision", json={"input": inp.__dict__}, timeout=2)
                r.raise_for_status()
                res = r.json().get("result", {})
                return PolicyDecision(bool(res.get("allow")), sorted(res.get("reasons", [])), "opa")
            except httpx.HTTPError:
                pass   # fail over to the local mirror; production would fail closed
        return self._local(inp)

    def _local(self, inp: PolicyInput) -> PolicyDecision:
        c = cfg()
        readable = {cls for r in inp.roles for cls in c["roles"].get(r, {}).get("repo_classes", [])}
        external = bool(inp.target_tier and c["tiers"].get(inp.target_tier, {}).get("external"))
        reasons = []
        if inp.repo_class and inp.repo_class not in readable:
            reasons.append("role cannot read this repository class")
        if external and inp.repo_class and inp.repo_class not in {"public", "internal"}:
            reasons.append("external tier not permitted for this repository class")
        if inp.tool in WRITE_TOOLS and not inp.has_approval:
            reasons.append("write tool requires a bound approval token")
        if inp.injection_score >= c["gateway"]["injection_block_threshold"]:
            reasons.append("prompt-injection score above threshold")
        if inp.secret_findings > 0:
            reasons.append("high-confidence secret present")
        return PolicyDecision(not reasons, reasons)


policy_gate = PolicyGate()
