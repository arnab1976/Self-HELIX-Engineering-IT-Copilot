"""Prompt-injection scoring for prompts AND retrieved chunks (code comments, issues, logs can carry injections).

Production: Llama Prompt Guard / LLM Guard classifier (set HELIX_PROMPT_GUARD=hf:<model-id>). The pattern
scorer below is the fallback and a useful feature set either way. Retrieved text is always wrapped as
quoted data and never treated as instructions, so a missed injection still cannot call write tools.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"ignore (all |the |any )?(previous|prior|above|earlier) (instructions|rules|messages)", re.I), 3.0),
    (re.compile(r"disregard (all |the |your )?(rules|instructions|guidelines)", re.I), 2.5),
    (re.compile(r"\b(reveal|print|show|dump|list) (me )?(every|all|the) (secret|password|token|key|credential)s?", re.I), 2.5),
    (re.compile(r"\byou are now\b|\bact as (an? )?(unrestricted|jailbroken)", re.I), 1.8),
    (re.compile(r"\bsystem prompt\b", re.I), 1.5),
    (re.compile(r"(send|post|upload|exfiltrate) .{0,40}(https?://|webhook)", re.I), 2.5),
    (re.compile(r"note to (the )?(ai|assistant|llm|model)", re.I), 1.5),
    (re.compile(r"(approve|merge) (this|the) (pr|pull request) (automatically|without review)", re.I), 2.0),
]


def score(text: str) -> tuple[float, list[str]]:
    guard = _guard_model()
    if guard is not None:  # pragma: no cover - optional model
        res = guard(text[:2000])[0]
        p = res["score"] if res["label"].upper() in ("INJECTION", "JAILBREAK", "MALICIOUS") else 1 - res["score"]
        return float(p), [res["label"]]
    z, hits = -3.0, []
    for rx, w in PATTERNS:
        if rx.search(text):
            z += w
            hits.append(rx.pattern[:50])
    return 1 / (1 + math.exp(-z)), hits


@lru_cache
def _guard_model():
    spec = os.environ.get("HELIX_PROMPT_GUARD", "")
    if not spec.startswith("hf:"):
        return None
    from transformers import pipeline  # type: ignore

    return pipeline("text-classification", model=spec[3:])


def quote_untrusted(text: str, source: str) -> str:
    """Wrap retrieved content so the model treats it as data."""
    safe = text.replace("</untrusted>", "")
    return f'<untrusted source="{source}">\n{safe}\n</untrusted>'
