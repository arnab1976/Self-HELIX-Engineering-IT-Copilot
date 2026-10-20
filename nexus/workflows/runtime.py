"""Run / resume LangGraph workflows with a persistent checkpointer, and record pending human actions."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from common.config import settings
from common.store import store
from common.tracing import current_trace_id


@lru_cache
def checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(settings().path("checkpoints.db"), check_same_thread=False)
    return SqliteSaver(conn)


def _graph(name: str):
    from nexus.workflows import change_risk, helpdesk, incident, pr_review, triage

    return {"helpdesk": helpdesk.graph, "triage": triage.graph, "pr_review": pr_review.graph,
            "incident": incident.graph, "change_risk": change_risk.graph}[name]()


def _result(name: str, thread_id: str, out: dict) -> dict:
    intr = out.pop("__interrupt__", None)
    return {"workflow": name, "thread_id": thread_id, "state": out, "trace_id": current_trace_id(),
            "interrupt": intr[0].value if intr else None, "status": "waiting_for_human" if intr else "completed"}


def run(name: str, inputs: dict, thread_id: str | None = None) -> dict:
    thread_id = thread_id or f"{name}-{secrets.token_hex(4)}"
    cfg = {"configurable": {"thread_id": thread_id}}
    out = _graph(name).invoke(inputs, cfg)
    return _result(name, thread_id, dict(out))


def resume(name: str, thread_id: str, value: Any) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    out = _graph(name).invoke(Command(resume=value), cfg)
    return _result(name, thread_id, dict(out))


def get_state(name: str, thread_id: str) -> dict:
    snap = _graph(name).get_state({"configurable": {"thread_id": thread_id}})
    return {"values": snap.values, "next": list(snap.next)}


# ------------------------------------------------------------------------------------------- pending actions
def create_action(workflow: str, thread_id: str, tool: str, params: dict, requested_by: str, approver_role: str) -> str:
    aid = "act_" + secrets.token_hex(3)
    store().execute("INSERT INTO actions(id, workflow, thread_id, tool, params, status, requested_by, approver_role, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", (aid, workflow, thread_id, tool, json.dumps(params), "pending", requested_by,
                                                  approver_role, time.time()))
    return aid


def get_action(aid: str) -> dict | None:
    a = store().one("SELECT * FROM actions WHERE id=?", (aid,))
    if a:
        a["params"] = json.loads(a["params"])
    return a


def pending_actions() -> list[dict]:
    rows = store().query("SELECT * FROM actions WHERE status='pending' ORDER BY created_at DESC")
    for r in rows:
        r["params"] = json.loads(r["params"])
    return rows


def close_action(aid: str, status: str, approver: str, token: str | None, result: Any) -> None:
    store().execute("UPDATE actions SET status=?, approver=?, token=?, result=? WHERE id=?",
                    (status, approver, token, json.dumps(result, default=str), aid))
