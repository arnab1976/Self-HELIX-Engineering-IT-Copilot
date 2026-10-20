"""Log connector: Loki query_range (production) or a local log file (MVP)."""
from __future__ import annotations

from pathlib import Path

import httpx

from common.config import settings


def read_log_file(path: Path | None = None) -> list[str]:
    path = path or Path(settings().data_dir) / "generated" / "logs" / "checkout.log"
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def loki_query_range(base_url: str, logql: str, start_ns: int, end_ns: int, limit: int = 5000) -> list[str]:
    r = httpx.get(f"{base_url}/loki/api/v1/query_range",
                  params={"query": logql, "start": start_ns, "end": end_ns, "limit": limit, "direction": "forward"}, timeout=30)
    r.raise_for_status()
    lines = []
    for stream in r.json()["data"]["result"]:
        lines += [v[1] for v in stream["values"]]
    return lines
