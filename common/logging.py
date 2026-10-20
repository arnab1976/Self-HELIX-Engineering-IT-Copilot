"""Structured JSON logging with trace_id propagation."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def log_event(log: logging.Logger, msg: str, **fields) -> None:
    log.info(msg, extra={"extra_fields": fields})
