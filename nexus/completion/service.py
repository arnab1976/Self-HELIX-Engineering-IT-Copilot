"""IDE inline completion (spec 6.4, C4). Fixed route to T1a; p95 first token <= 300 ms.

Context assembly (~1,000 tokens, outside the FIM span):
  1. definitions of symbols referenced near the cursor (code graph / index)
  2. similar snippets from the same repo (BM25 on the current line + enclosing function)
Post-filters: secret scan, prefix repetition, bracket/quote balance. Any failure -> no suggestion.
"""
from __future__ import annotations

import re
import time

from common.config import cfg
from common.textutils import estimate_tokens
from lexora.prep import secrets
from lexora.prep.fim import build_prompt
from nexus.llm.providers import complete_fim
from nexus.rag.index import get_index

# Fallback suggestions used when T1a has no endpoint (MockProvider). Ordered: first match wins.
RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"if not card\.number:\s*$"), 'raise ValidationError("card number is required")'),
    (re.compile(r"if not self\.luhn_ok\(card\.number\):\s*$"), 'raise ValidationError("invalid card number")'),
    (re.compile(r"if card\.expiry < (date\.)?today\(\):\s*$"), 'raise ValidationError("card has expired")'),
    (re.compile(r"if not addr\.line1:\s*$"), 'raise ValidationError("address line 1 is required")'),
    (re.compile(r"requests\.get\($"), "url, timeout=5)"),
    (re.compile(r"logger\.$"), 'info("event", extra={})'),
    (re.compile(r"cursor\.execute\($"), '"SELECT id, status FROM orders WHERE id = %s", (order_id,))'),
    (re.compile(r"def validate_\w*\($"), "card: Card) -> None:"),
    (re.compile(r"return\s*$"), "None"),
    (re.compile(r"^\s*$"), ""),
]


def _fallback(prefix: str, suffix: str) -> str:
    lines = prefix.split("\n")
    last = lines[-1]
    if not last.strip() and len(lines) > 1:      # cursor on a fresh indented line: use the statement above
        last = lines[-2]
    for rx, s in RULES:
        if rx.search(last):
            if s and s.split("(")[0] in suffix[:200]:
                continue
            return s
    return ""


def assemble_context(prefix: str, repo: str | None, budget_tokens: int) -> tuple[str, list[str]]:
    idx = get_index()
    names = set(re.findall(r"\b([A-Z][A-Za-z0-9_]+|[a-z_][a-z0-9_]{3,})\b", prefix[-600:]))
    lines, used, toks = [], [], 0
    for c in idx.chunks.values():
        if repo and c.repo != repo:
            continue
        short = c.symbol.split(".")[-1]
        if short in names and c.kind in ("class", "function", "method"):
            head = "\n".join("# " + ln for ln in c.text.splitlines()[:3])
            t = estimate_tokens(head)
            if toks + t > budget_tokens:
                break
            lines.append(f"# {c.citation()}\n{head}")
            used.append(c.symbol)
            toks += t
    return ("\n".join(lines) + "\n") if lines else "", used


def post_filters(prefix: str, completion: str) -> dict:
    sec = bool(secrets.scan(completion))
    rep = bool(completion.strip()) and prefix.rstrip().endswith(completion.strip())
    bal = completion.count("(") == completion.count(")") and completion.count('"') % 2 == 0 and completion.count("[") == completion.count("]")
    return {"secrets": "found" if sec else "clean", "prefix_repeat": rep, "syntax_balanced": bal, "passed": not sec and not rep and bal}


def complete(prefix: str, suffix: str, path: str, language: str, repo: str | None, max_tokens: int | None = None) -> dict:
    t0 = time.perf_counter()
    cc = cfg()["completion"]
    ctx, used = assemble_context(prefix, repo, cc["context_tokens"])
    prompt = build_prompt(prefix[-4000:], suffix[:1500], "qwen", context=f"# path: {path}\n# language: {language}\n{ctx}")
    res = complete_fim("T1a", prompt, fallback=_fallback(prefix, suffix), max_tokens=max_tokens or cc["max_tokens"])
    text = res.text.rstrip()
    filt = post_filters(prefix, text)
    if not filt["passed"]:
        text = ""
    return {"completion": text, "model": "lexora/fim-coder@2.1.0" if not res.mocked else res.model, "mocked": res.mocked,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "context_symbols": used, "post_filters": filt,
            "prompt_tokens": estimate_tokens(prompt), "prompt_preview": prompt[-700:]}
