"""SQLite persistence for the MVP: mock ITSM, feedback, audit, approvals, incidents.

Production swaps this for PostgreSQL (same schema) and the real ServiceNow/Jira APIs behind
nexus/tools. Kept deliberately small and explicit so it is easy to read and change.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from common.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  id TEXT PRIMARY KEY, title TEXT, description TEXT, requester TEXT, ci TEXT,
  category TEXT, assignment_group TEXT, priority TEXT, state TEXT, created_at REAL,
  duplicate_of TEXT, affected_users TEXT DEFAULT '[]', source TEXT DEFAULT 'portal'
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, user TEXT, signal TEXT, target TEXT,
  payload TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, trace_id TEXT, actor TEXT, action TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  id TEXT PRIMARY KEY, workflow TEXT, thread_id TEXT, tool TEXT, params TEXT, status TEXT,
  requested_by TEXT, approver_role TEXT, approver TEXT, token TEXT, created_at REAL, result TEXT
);
CREATE TABLE IF NOT EXISTS incidents (
  id TEXT PRIMARY KEY, service TEXT, status TEXT, brief TEXT, thread_id TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY, result TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS access_grants (
  user TEXT, resource TEXT, role TEXT, granted_at REAL, PRIMARY KEY (user, resource)
);
"""


class Store:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings().path("helix.db"))
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------------------------------------------------------- generic
    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- audit & feedback
    def audit(self, trace_id: str, actor: str, action: str, detail: Any) -> None:
        self.execute(
            "INSERT INTO audit(ts, trace_id, actor, action, detail) VALUES (?,?,?,?,?)",
            (time.time(), trace_id, actor, action, json.dumps(detail, default=str)),
        )

    def add_feedback(self, trace_id: str, user: str, signal: str, target: str, payload: Any) -> int:
        cur = self.execute(
            "INSERT INTO feedback(trace_id, user, signal, target, payload, created_at) VALUES (?,?,?,?,?,?)",
            (trace_id, user, signal, target, json.dumps(payload, default=str), time.time()),
        )
        return int(cur.lastrowid)


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def reset_store(path: Path | None = None) -> Store:
    """Used by tests to get an isolated database."""
    global _store
    _store = Store(path)
    return _store
