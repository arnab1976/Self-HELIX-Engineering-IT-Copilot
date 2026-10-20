"""Evaluation metrics used by the LEXORA release gates. All pure NumPy; each has a docstring formula."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k)  (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    return len(set(ranked[:k]) & relevant) / max(1, len(relevant))


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: dict[str, float], k: int = 10) -> float:
    """nDCG@k = DCG/IDCG, DCG = sum rel_i / log2(i+1)."""
    dcg = sum(gains.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) |acc(b) - conf(b)| over equal-width confidence bins."""
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - probs[m].mean())
    return float(ece)


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC = P(score_pos > score_neg), computed with the rank-sum (Mann-Whitney U) identity."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall_at_threshold(probs: Sequence[float], labels: Sequence[int], threshold: float) -> tuple[float, float]:
    p, y = np.asarray(probs), np.asarray(labels)
    pred = p >= threshold
    tp = float((pred & (y == 1)).sum())
    return tp / max(1.0, pred.sum()), tp / max(1.0, (y == 1).sum())


def threshold_for_precision(probs: Sequence[float], labels: Sequence[int], target: float) -> float:
    """Smallest threshold whose precision >= target (maximises recall at that precision)."""
    for t in np.round(np.arange(0.30, 0.995, 0.01), 2):
        prec, rec = precision_recall_at_threshold(probs, labels, t)
        if prec >= target and rec > 0:
            return float(t)
    return 0.99
