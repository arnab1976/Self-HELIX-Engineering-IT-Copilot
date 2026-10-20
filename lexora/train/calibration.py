"""Temperature scaling (Guo et al. 2017): p_i = softmax(z_i / T). T is fitted on validation data by
minimising negative log-likelihood. It changes confidence, never the argmax, so accuracy is unchanged."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    grid = grid if grid is not None else np.round(np.arange(0.25, 5.0, 0.05), 2)
    losses = [nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])
