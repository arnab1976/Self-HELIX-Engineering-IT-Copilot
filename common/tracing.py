"""Lightweight tracing: trace ids, spans with timings, optional OpenTelemetry / Langfuse export.

Every API response carries a trace_id (spec 11). Spans are kept in memory per trace so the
portal can show what happened; production exports them through OpenTelemetry to Tempo and
LLM generations to Langfuse.
"""
from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

from common.logging import trace_id_var

_TRACES: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_MAX_TRACES = 2000


def new_trace_id() -> str:
    return "tr_" + secrets.token_hex(6)


def current_trace_id() -> str:
    tid = trace_id_var.get()
    if tid == "-":
        tid = new_trace_id()
        trace_id_var.set(tid)
    return tid


@contextmanager
def span(name: str, **attrs: Any):
    tid = current_trace_id()
    start = time.perf_counter()
    rec: dict[str, Any] = {"name": name, "attrs": attrs, "start": time.time()}
    try:
        yield rec
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = repr(exc)
        raise
    finally:
        rec["ms"] = round((time.perf_counter() - start) * 1000, 2)
        _TRACES.setdefault(tid, []).append(rec)
        while len(_TRACES) > _MAX_TRACES:
            _TRACES.popitem(last=False)


def get_trace(trace_id: str) -> list[dict[str, Any]]:
    return _TRACES.get(trace_id, [])
