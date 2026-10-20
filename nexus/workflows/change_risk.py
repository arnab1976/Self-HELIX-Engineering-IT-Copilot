"""Change-risk agent (C9): ServiceNow change request -> features -> L9 score + TreeSHAP -> CAB brief -> route."""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from nexus.llm import generate
from nexus.ml.change_risk import score
from nexus.workflows.runtime import checkpointer


class ChangeState(TypedDict, total=False):
    change: dict
    risk: dict
    brief: str
    route: str


def enrich(s: ChangeState) -> ChangeState:
    c = {"services_touched": 1, "lines_changed": 100, "test_coverage": 60, "author_failure_rate": 8, "dependency_fanout": 5,
         "peak_window": 0, "has_rollback_plan": 1, "change_type": "normal", **s["change"]}
    return {"change": c}


def score_node(s: ChangeState) -> ChangeState:
    return {"risk": score(s["change"])}


def brief(s: ChangeState) -> ChangeState:
    r = s["risk"]
    out = generate("T2", [{"role": "system", "content": "Write a 3-sentence CAB brief from these risk factors only."},
                          {"role": "user", "content": str(r["factors"][:5])}], fallback=r["cab_brief"])
    return {"brief": out.text}


def route(s: ChangeState) -> ChangeState:
    return {"route": {"high": "full CAB review", "medium": "peer review + canary", "low": "standard change, automated checks"}[s["risk"]["band"]]}


@lru_cache
def graph():
    g = StateGraph(ChangeState)
    for n, f in [("enrich", enrich), ("score", score_node), ("brief", brief), ("route", route)]:
        g.add_node(n, f)
    g.add_edge(START, "enrich")
    g.add_edge("enrich", "score")
    g.add_edge("score", "brief")
    g.add_edge("brief", "route")
    g.add_edge("route", END)
    return g.compile(checkpointer=checkpointer())
