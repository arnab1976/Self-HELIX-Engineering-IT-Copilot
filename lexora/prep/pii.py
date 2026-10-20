"""PII detection and anonymisation. Uses Microsoft Presidio when installed, regex recognisers otherwise.

Emails are replaced with a stable hashed surrogate (<EMAIL:ab12cd>) so a requester stays joinable across
tickets without exposing the address.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class PIIFinding:
    type: str
    value: str
    placeholder: str
    start: int
    end: int


EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\w)\+?\d{1,3}[\s-]?\d{4,5}[\s-]?\d{4,6}(?!\w)")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _surrogate(kind: str, value: str) -> str:
    return f"<{kind}:{hashlib.sha1(value.lower().encode()).hexdigest()[:6]}>"


def scan(text: str, include_ip: bool = False) -> list[PIIFinding]:
    engine = _presidio()
    if engine is not None:  # production path: Presidio NER + pattern recognisers
        res = engine.analyze(text=text, language="en", entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"])
        return [PIIFinding(r.entity_type, text[r.start:r.end],
                           _surrogate("EMAIL", text[r.start:r.end]) if r.entity_type == "EMAIL_ADDRESS" else f"<{r.entity_type}>",
                           r.start, r.end) for r in res if r.score >= 0.6]
    out: list[PIIFinding] = []
    for m in EMAIL.finditer(text):
        out.append(PIIFinding("EMAIL_ADDRESS", m.group(0), _surrogate("EMAIL", m.group(0)), m.start(), m.end()))
    for m in PHONE.finditer(text):
        if not any(f.start <= m.start() < f.end for f in out):
            out.append(PIIFinding("PHONE_NUMBER", m.group(0), "<PHONE>", m.start(), m.end()))
    if include_ip:
        for m in IPV4.finditer(text):
            out.append(PIIFinding("IP_ADDRESS", m.group(0), "<IP>", m.start(), m.end()))
    return sorted(out, key=lambda f: f.start)


_ENGINE: object = False      # False = not tried yet, None = unavailable


def _presidio():
    """Presidio AnalyzerEngine when installed (needs a spaCy model), else None -> regex recognisers."""
    global _ENGINE
    if _ENGINE is False:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore

            _ENGINE = AnalyzerEngine()
        except Exception:  # noqa: BLE001
            _ENGINE = None
    return _ENGINE


def anonymise(text: str) -> tuple[str, list[PIIFinding]]:
    findings = scan(text, include_ip=True)
    out = text
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        out = out[: f.start] + f.placeholder + out[f.end :]
    return out, findings
