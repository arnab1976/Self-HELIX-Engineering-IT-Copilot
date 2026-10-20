"""Text helpers shared by prep, retrieval and classifiers."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be by does do done for from how i in is it my of on or the this that to was what when "
    "where which who why with you your me we our can cannot please".split()
)
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English and code; replace with the embedder's tokenizer in production."""
    return max(1, math.ceil(len(text) / 4))


def stem(word: str) -> str:
    w = word.lower()
    for suf in ("ations", "ation", "ings", "ing", "ers", "er", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str, *, keep_stopwords: bool = False, do_stem: bool = True) -> list[str]:
    """Identifier-aware tokenizer: splits camelCase and snake_case so BM25 matches symbol names."""
    text = _CAMEL.sub(r"\1 \2", text)
    toks = [t.lower() for t in _SPLIT.split(text) if t]
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks] if do_stem else toks


def entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def char_ngrams(text: str, n: int = 3) -> Counter:
    t = " ".join(tokenize(text, keep_stopwords=True, do_stem=False))
    return Counter(t[i : i + n] for i in range(max(0, len(t) - n + 1)))
