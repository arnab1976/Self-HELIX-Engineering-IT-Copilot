"""Token-bucket rate limiting per user and route class. Production: Kong/Envoy or Redis-backed buckets."""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, status

from common.config import cfg


class TokenBucket:
    """capacity C, refill r tokens/s; each request costs one token; empty bucket -> HTTP 429."""

    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity, self.rate = float(capacity), float(refill_per_sec)
        self.tokens, self.updated = float(capacity), time.monotonic()

    def take(self, n: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        return False, (n - self.tokens) / self.rate


class RateLimiter:
    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, kind: str = "default") -> None:
        conf = cfg()["gateway"]["completion_rate_limit" if kind == "completion" else "rate_limit"]
        with self._lock:
            b = self._buckets.setdefault(f"{kind}:{key}", TokenBucket(conf["capacity"], conf["refill_per_sec"]))
            ok, retry = b.take()
        if not ok:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                                headers={"Retry-After": str(max(1, round(retry)))})


limiter = RateLimiter()
