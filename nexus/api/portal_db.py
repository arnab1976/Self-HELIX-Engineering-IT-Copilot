"""PostgreSQL persistence for Agent 01 Employee Self-Service portal tickets."""
from __future__ import annotations

import logging
from typing import Any

from common.config import settings

log = logging.getLogger("nexus.api.portal_db")

_DDL = """
CREATE TABLE IF NOT EXISTS portal_tickets (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL DEFAULT '',
    assignment_group TEXT NOT NULL DEFAULT '',
    priority        TEXT NOT NULL DEFAULT 'P3',
    requester       TEXT NOT NULL DEFAULT '',
    ci              TEXT NOT NULL DEFAULT '',
    impact          TEXT NOT NULL DEFAULT '',
    split           TEXT NOT NULL DEFAULT 'train',
    status          TEXT NOT NULL DEFAULT 'New',
    channel         TEXT NOT NULL DEFAULT 'employee_self_service_portal',
    created_at      TIMESTAMPTZ,
    saved_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def postgres_url() -> str:
    s = settings()
    return (getattr(s, "postgres_url", None) or "").strip() or "postgresql://postgres:helix@127.0.0.1:5432/helix"


def _connect():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is not installed. Run: pip install 'psycopg[binary]>=3.1'"
        ) from exc
    return psycopg.connect(postgres_url(), connect_timeout=5)


def ensure_schema() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()


def upsert_tickets(tickets: list[dict[str, Any]]) -> dict[str, Any]:
    """Insert or update portal tickets. Returns counts."""
    if not tickets:
        return {"saved": 0, "ids": []}
    ensure_schema()
    ids: list[str] = []
    with _connect() as conn:
        with conn.cursor() as cur:
            for raw in tickets:
                tid = str(raw.get("id") or "").strip()
                title = str(raw.get("title") or "").strip()
                if not tid or not title:
                    continue
                cur.execute(
                    """
                    INSERT INTO portal_tickets (
                        id, title, description, category, assignment_group, priority,
                        requester, ci, impact, split, status, channel, created_at, saved_at
                    ) VALUES (
                        %(id)s, %(title)s, %(description)s, %(category)s, %(assignment_group)s, %(priority)s,
                        %(requester)s, %(ci)s, %(impact)s, %(split)s, %(status)s, %(channel)s,
                        NULLIF(%(created_at)s, '')::timestamptz, NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        category = EXCLUDED.category,
                        assignment_group = EXCLUDED.assignment_group,
                        priority = EXCLUDED.priority,
                        requester = EXCLUDED.requester,
                        ci = EXCLUDED.ci,
                        impact = EXCLUDED.impact,
                        split = EXCLUDED.split,
                        status = EXCLUDED.status,
                        channel = EXCLUDED.channel,
                        created_at = COALESCE(EXCLUDED.created_at, portal_tickets.created_at),
                        saved_at = NOW()
                    """,
                    {
                        "id": tid,
                        "title": title,
                        "description": str(raw.get("description") or title),
                        "category": str(raw.get("category") or ""),
                        "assignment_group": str(raw.get("assignment_group") or ""),
                        "priority": str(raw.get("priority") or "P3"),
                        "requester": str(raw.get("requester") or ""),
                        "ci": str(raw.get("ci") or ""),
                        "impact": str(raw.get("impact") or ""),
                        "split": str(raw.get("split") or "train"),
                        "status": str(raw.get("status") or "New"),
                        "channel": str(raw.get("channel") or "employee_self_service_portal"),
                        "created_at": str(raw.get("created_at") or ""),
                    },
                )
                ids.append(tid)
        conn.commit()
    return {"saved": len(ids), "ids": ids}


def list_tickets(limit: int = 100) -> list[dict[str, Any]]:
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, category, assignment_group, priority,
                       requester, ci, impact, split, status, channel,
                       created_at::text, saved_at::text
                FROM portal_tickets
                ORDER BY saved_at DESC
                LIMIT %s
                """,
                (max(1, min(limit, 500)),),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def ping() -> dict[str, Any]:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    return {"ok": True, "url_host": postgres_url().split("@")[-1]}
