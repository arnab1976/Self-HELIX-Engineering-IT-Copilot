"""Ticket triage agent (C2): runs on every new ticket.

classify (L1) -> duplicates (L8) -> decide -> act
  duplicate found      -> link to the open ticket, add requester as affected user (no new work item)
  p(group) >= 0.85     -> auto-route to the predicted assignment group
  otherwise            -> agent queue with top-3 suggestions
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common.config import cfg
from nexus.ml.duplicates import find_duplicates
from nexus.ml.triage import get_triage
from nexus.tools import registry
from nexus.workflows.runtime import checkpointer


class TriageState(TypedDict, total=False):
    title: str
    description: str
    requester: str
    ci: str
    create: bool               # create the ticket in ITSM (portal/chat) or it already exists (webhook)
    ticket: str
    prediction: dict
    duplicates: list
    decision: str
    result: dict
    kb: list


def classify(s: TriageState) -> TriageState:
    return {"prediction": get_triage().predict(s["title"], s.get("description", ""))}


def duplicates(s: TriageState) -> TriageState:
    return {"duplicates": find_duplicates(s["title"], s.get("description", ""), s.get("ci", ""), s.get("requester", ""))}


def suggest_kb(s: TriageState) -> TriageState:
    try:
        kb = registry.call("kb.search", {"query": f"{s['title']} {s.get('description', '')}", "k": 5})
    except Exception:  # noqa: BLE001 - KB suggestions are best-effort
        kb = []
    cat = s["prediction"]["category"]["label"]
    kb = [k for k in kb if k.get("category") in (None, cat)][:2]     # only suggest articles for the predicted category
    return {"kb": kb}


def decide(s: TriageState) -> TriageState:
    dup = next((d for d in s.get("duplicates", []) if d["duplicate"]), None)
    p = s["prediction"]["assignment_group"]["p"]
    if dup:
        return {"decision": f"duplicate_of:{dup['ticket']}"}
    return {"decision": "auto_route" if p >= cfg()["triage"]["auto_route_threshold"] else "agent_queue"}


def act(s: TriageState) -> TriageState:
    pred, dec = s["prediction"], s["decision"]
    if dec.startswith("duplicate_of:"):
        target = dec.split(":", 1)[1]
        res = registry.call("itsm.update_ticket", {"ticket": target, "add_affected_user": s.get("requester") or "unknown"}, actor="triage-agent")
        return {"result": {"action": "linked_to_existing", "ticket": target, "update": res}}
    group = pred["assignment_group"]["label"] if dec == "auto_route" else "Service-Desk-Queue"
    if s.get("create", True):
        res = registry.call("itsm.create_ticket", {
            "title": s["title"], "description": s.get("description", ""), "requester": s.get("requester", "unknown"),
            "category": pred["category"]["label"], "assignment_group": group, "priority": pred["priority"]["label"],
            "ci": s.get("ci", "")}, actor="triage-agent")
        return {"ticket": res["ticket"], "result": {"action": dec, **res, "suggestions": pred["assignment_group"]["top"]}}
    res = registry.call("itsm.update_ticket", {"ticket": s["ticket"], "assignment_group": group}, actor="triage-agent")
    return {"result": {"action": dec, **res}}


@lru_cache
def graph():
    g = StateGraph(TriageState)
    for n, f in [("classify", classify), ("duplicates", duplicates), ("suggest_kb", suggest_kb), ("decide", decide), ("act", act)]:
        g.add_node(n, f)
    g.add_edge(START, "classify")
    g.add_edge("classify", "duplicates")
    g.add_edge("duplicates", "suggest_kb")
    g.add_edge("suggest_kb", "decide")
    g.add_edge("decide", "act")
    g.add_edge("act", END)
    return g.compile(checkpointer=checkpointer())
