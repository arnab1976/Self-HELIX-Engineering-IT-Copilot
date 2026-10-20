"""Gateway request scanning: secrets (block or redact), PII (redact), prompt injection (score)."""
from __future__ import annotations

from dataclasses import dataclass, field

from common.config import cfg
from lexora.prep import pii, secrets
from nexus.guardrails import injection


@dataclass
class GuardResult:
    text: str                                  # redacted text that may be forwarded
    secrets: list[dict] = field(default_factory=list)
    pii: list[dict] = field(default_factory=list)
    injection_score: float = 0.0
    injection_hits: list[str] = field(default_factory=list)
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"secrets": self.secrets, "pii_entities": len(self.pii), "injection_score": round(self.injection_score, 3),
                "blocked": self.blocked, "reasons": self.reasons}


def scan_prompt(text: str, redact_secrets: bool | None = None) -> GuardResult:
    g = cfg()["gateway"]
    block_secrets = g["block_on_high_confidence_secret"] if redact_secrets is None else not redact_secrets
    if len(text) > g["max_prompt_chars"]:
        return GuardResult(text[:0], blocked=True, reasons=["prompt too large"])
    found = secrets.scan(text)
    redacted, _ = secrets.redact(text, found)
    redacted, pfound = pii.anonymise(redacted)
    inj, hits = injection.score(text)
    res = GuardResult(
        redacted,
        [{"type": f.type, "value": f.masked(), "confidence": f.confidence, "placeholder": f.placeholder} for f in found],
        [{"type": p.type, "placeholder": p.placeholder} for p in pfound],
        inj, hits,
    )
    if block_secrets and any(f.confidence == "high" for f in found):
        res.blocked = True
        res.reasons.append("high-confidence secret in prompt: " + ", ".join(sorted({f.type for f in found})))
    if inj >= g["injection_block_threshold"]:
        res.reasons.append(f"prompt-injection score {inj:.2f}: instructions quarantined as data")
    return res
