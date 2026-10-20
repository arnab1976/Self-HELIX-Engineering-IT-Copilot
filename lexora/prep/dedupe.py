"""Near-duplicate removal with MinHash + LSH banding.

Pipeline: shingle -> MinHash signature (k permutations) -> b bands x r rows -> bucket per band ->
candidate pairs -> verify estimated Jaccard >= threshold.
P(candidate | similarity s) = 1 - (1 - s^r)^b ; effective threshold ~ (1/b)^(1/r).

Uses `datasketch.MinHashLSH` when installed; the pure-NumPy implementation below is equivalent and
readable for interviews.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from common.textutils import tokenize

_MERSENNE = (1 << 61) - 1
_MAX = (1 << 32) - 1


def shingles(text: str, n: int = 3) -> set[str]:
    t = tokenize(text, keep_stopwords=True, do_stem=False)
    return {" ".join(t[i : i + n]) for i in range(max(1, len(t) - n + 1))}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.k = num_perm
        self.a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    def signature(self, items: set[str]) -> np.ndarray:
        if not items:
            return np.full(self.k, _MAX, dtype=np.uint64)
        hv = np.array([int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "little") for s in items], dtype=np.uint64)
        # (a*x + b) mod p mod 2^32, vectorised over permutations x items
        ph = (np.outer(self.a, hv) + self.b[:, None]) % np.uint64(_MERSENNE) & np.uint64(_MAX)
        return ph.min(axis=1)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def lsh_params(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose S-curve threshold (1/b)^(1/r) is closest to, and not above, `threshold`."""
    best = (num_perm, 1)
    best_gap = 9.0
    for r in range(1, num_perm + 1):
        if num_perm % r:
            continue
        b = num_perm // r
        t = (1 / b) ** (1 / r)
        gap = threshold - t
        if 0 <= gap < best_gap:
            best, best_gap = (b, r), gap
    return best


class LSHIndex:
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.hasher = MinHasher(num_perm)
        self.threshold = threshold
        self.bands, self.rows = lsh_params(num_perm, threshold)
        self.tables: list[dict[bytes, list[str]]] = [defaultdict(list) for _ in range(self.bands)]
        self.sigs: dict[str, np.ndarray] = {}

    def _band_keys(self, sig: np.ndarray):
        for i in range(self.bands):
            yield i, sig[i * self.rows : (i + 1) * self.rows].tobytes()

    def query(self, sig: np.ndarray) -> list[tuple[str, float]]:
        cands = set()
        for i, key in self._band_keys(sig):
            cands.update(self.tables[i].get(key, []))
        out = [(c, float(np.mean(self.sigs[c] == sig))) for c in cands]
        return sorted([x for x in out if x[1] >= self.threshold], key=lambda x: -x[1])

    def insert(self, key: str, sig: np.ndarray) -> None:
        self.sigs[key] = sig
        for i, k in self._band_keys(sig):
            self.tables[i][k].append(key)


def deduplicate(items: list[tuple[str, str]], threshold: float = 0.8, num_perm: int = 128) -> tuple[list[str], dict[str, tuple[str, float]]]:
    """items = [(key, text)]. Returns (kept keys, {dropped_key: (kept_duplicate_key, est_jaccard)})."""
    idx = LSHIndex(num_perm, threshold)
    kept, dropped = [], {}
    for key, text in items:
        sig = idx.hasher.signature(shingles(text))
        hits = idx.query(sig)
        if hits:
            dropped[key] = hits[0]
        else:
            idx.insert(key, sig)
            kept.append(key)
    return kept, dropped
