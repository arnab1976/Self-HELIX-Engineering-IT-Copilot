"""Secret detection and redaction (gitleaks / TruffleHog / detect-secrets style rules).

Used in two places:
  * LEXORA prep: every file and commit is scanned; matches become typed placeholders, and files with
    unresolved findings are dropped from training.
  * NEXUS gateway: prompts are scanned before any model call; high-confidence keys block the request.

Rules = provider-specific regexes (high confidence) + keyword assignments scored by Shannon entropy.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from common.textutils import entropy


@dataclass
class SecretFinding:
    type: str
    value: str
    rule: str
    entropy: float
    confidence: str          # high | medium
    placeholder: str
    start: int
    end: int

    def masked(self) -> str:
        v = self.value
        return v[:4] + "…" + v[-4:] if len(v) > 12 else "****"


PROVIDER_RULES: list[tuple[str, re.Pattern, str]] = [
    ("AWS access key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "<AWS_KEY>"),
    ("Stripe live key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{16,}\b"), "<STRIPE_KEY>"),
    ("GitHub token", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{30,}\b"), "<GITHUB_TOKEN>"),
    ("Slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"), "<SLACK_TOKEN>"),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "<GOOGLE_KEY>"),
    ("JSON Web Token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "<JWT>"),
    ("Private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|$)"), "<PRIVATE_KEY>"),
]
KEYWORD_RULE = re.compile(
    r"""(?P<key>[A-Za-z_]*(?:SECRET|PASSWORD|PASSWD|TOKEN|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY)[A-Za-z_]*)\s*[:=]\s*["'](?P<val>[^"'\s]{6,})["']""",
    re.IGNORECASE,
)
ENTROPY_HIGH = 4.0


def scan(text: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    taken: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < te and e > ts for ts, te in taken)

    for name, rx, ph in PROVIDER_RULES:
        for m in rx.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            findings.append(SecretFinding(name, m.group(0), rx.pattern[:40], entropy(m.group(0)), "high", ph, m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in KEYWORD_RULE.finditer(text):
        s, e = m.start("val"), m.end("val")
        if overlaps(s, e):
            continue
        val, key = m.group("val"), m.group("key").upper()
        if val.startswith("<") or val.lower() in {"changeme", "password", "xxxxxx", "example"}:
            continue
        h = entropy(val)
        is_pwd = "PASS" in key
        findings.append(SecretFinding("Password literal" if is_pwd else "Generic secret", val, "keyword assignment + entropy", h,
                                      "high" if (h >= ENTROPY_HIGH or is_pwd) else "medium",
                                      "<PASSWORD>" if is_pwd else "<SECRET>", s, e))
        taken.append((s, e))
    return sorted(findings, key=lambda f: f.start)


def redact(text: str, findings: list[SecretFinding] | None = None) -> tuple[str, list[SecretFinding]]:
    findings = scan(text) if findings is None else findings
    out = text
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        out = out[: f.start] + f.placeholder + out[f.end :]
    return out, findings


def gitleaks_available() -> bool:
    return shutil.which("gitleaks") is not None


def gitleaks_scan_dir(path: str) -> str:
    """Run the real gitleaks binary when installed (full history scan in CI)."""
    res = subprocess.run(["gitleaks", "detect", "--no-banner", "--source", path, "--report-format", "json", "--report-path", "-"],
                         capture_output=True, text=True, timeout=600)
    return res.stdout
