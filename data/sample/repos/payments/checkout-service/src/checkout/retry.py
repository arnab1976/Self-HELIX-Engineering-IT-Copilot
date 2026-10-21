"""Retry helpers for checkout outbound calls."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from checkout.errors import TransientError

T = TypeVar("T")


def with_backoff(fn: Callable[[], T], *, attempts: int = 3, base_ms: int = 50) -> T:
    """Run fn with exponential backoff on TransientError."""
    delay = base_ms / 1000.0
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except TransientError as exc:
            last = exc
            if i == attempts - 1:
                break
            time.sleep(delay)
            delay *= 2
    assert last is not None
    raise last


def is_retriable_status(code: int) -> bool:
    return code in {408, 429, 500, 502, 503, 504}
