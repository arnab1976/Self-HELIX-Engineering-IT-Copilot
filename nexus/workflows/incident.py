"""Incident copilot agent (C8; spec 7.1): gather, correlate, draft, approve, act, verify, postmortem.

 1 alert  2 metrics  3 logs (+Drain)  4 traces  5 changes  6 correlate  7 similar incidents  8 draft brief (T1c/L7)
 9 [human gate: SRE approves / modifies / rejects]  10 execute runbook (approval token)  11 verify SLO (loop to 6 once)
12 postmortem draft
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common.config import cfg
from common.store import store
from nexus.gateway.auth import Principal
from nexus.llm import generate
from nexus.ml.anomaly import first_anomaly, rank_changes
from nexus.rag.retriever import retrieve
from nexus.tools import registry
from nexus.workflows.runtime import checkpointer, create_action


class IncidentState(TypedDict, total=False):
    incident_id: str
    thread_id: str
    alert: dict
    service: str
    metrics: dict
    anomaly: dict
    logs: dict
    traces: dict
    changes: list
    hypotheses: list
    similar: list
    brief: dict
    action_id: str
    decision: dict
    execution: dict
    verify: dict
    attempts: int
    postmortem: str
    timeline: list


def _t(s: IncidentState, msg: str) -> list:
    return [*s.get("timeline", []), {"ts": time.strftime("%H:%M:%S"), "event": msg}]


def receive(s: IncidentState) -> IncidentState:
    svc = s.get("alert", {}).get("service", "checkout-service")
    store().execute("INSERT OR REPLACE INTO incidents(id, service, status, brief, thread_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (s["incident_id"], svc, "investigating", "{}", s["thread_id"], time.time(), time.time()))
    return {"service": svc, "attempts": 0, "timeline": [{"ts": time.strftime("%H:%M:%S"), "event": f"alert {s['alert'].get('name')} received"}]}


def gather_metrics(s: IncidentState) -> IncidentState:
    m = registry.call("obs.query_metrics", {"service": s["service"]})
    region = max(m["error_rate_pct"], key=lambda r: m["error_rate_pct"][r][-1])
    c = cfg()["incident"]
    idx, detail = first_anomaly(m["error_rate_pct"][region], c["zscore_window"], c["zscore_threshold"])
    z = next((d for d in detail if d["i"] == idx), None)
    return {"metrics": {"region": region, "error_rate_now": m["error_rate_pct"][region][-1], "p99_now": m["p99_latency_ms"][-1],
                        "p99_before": m["p99_latency_ms"][30], "series": m["error_rate_pct"][region]},
            "anomaly": {"minute": idx, "detail": z, "start": m["start"]}, "timeline": _t(s, f"anomaly at minute {idx} in {region} (z={z['z'] if z else '?'})")}


def gather_logs(s: IncidentState) -> IncidentState:
    a = s["anomaly"]["minute"] or 40
    lg = registry.call("obs.query_logs", {"service": s["service"], "start_minute": max(0, a - 5), "end_minute": 60})
    new = [t for t in lg["templates"] if t["new_since_baseline"]]
    return {"logs": {"templates": lg["templates"][:6], "new_templates": new}, "timeline": _t(s, f"{len(new)} new log templates since baseline")}


def gather_traces(s: IncidentState) -> IncidentState:
    return {"traces": registry.call("obs.query_traces", {"service": s["service"]})}


def gather_changes(s: IncidentState) -> IncidentState:
    ch = registry.call("deploy.list_changes", {"service": None})
    return {"changes": ch, "timeline": _t(s, f"{len(ch)} changes in window")}


def correlate(s: IncidentState) -> IncidentState:
    start = datetime.fromisoformat(s["anomaly"]["start"])
    a = s["anomaly"]["minute"] or 40
    new = s["logs"]["new_templates"]
    err_tpl = next((t for t in new if "ERROR" in t["template"]), None)
    lat_up = s["metrics"]["p99_now"] - s["metrics"]["p99_before"]
    cands = []
    for c in s["changes"]:
        minute = int((datetime.fromisoformat(c["ts"]) - start).total_seconds() // 60)
        same_svc = c["service"] == s["service"]
        cands.append({"cause": f"{'Deploy' if c['kind'] == 'deploy' else 'Feature flag'} {c['service']} {c['detail']} at {c['ts'][11:16]}",
                      "minute": minute, "change": c,
                      "template_evidence": 1.0 if (err_tpl and same_svc and c["kind"] == "deploy") else 0.0,
                      "latency_evidence": 0.9 if same_svc and c["kind"] == "deploy" else 0.1,
                      "evidence": ([f"new log template '{err_tpl['template']}' x{err_tpl['count']}"] if err_tpl and same_svc and c["kind"] == "deploy" else [])
                      + ([f"p99 latency +{lat_up:.0f} ms after change"] if same_svc and c["kind"] == "deploy" else [])})
    cands.append({"cause": "Payment gateway latency", "minute": a - 3, "template_evidence": 0.1, "latency_evidence": 0.4,
                  "evidence": ["upstream span p99 +90 ms"], "change": None})
    ranked = rank_changes(a, cands, cfg()["incident"]["overlap_tau_minutes"])
    hyps = [{"rank": i + 1, "cause": h["cause"], "evidence": h["evidence"], "confidence": h["confidence"], "overlap": h["overlap"],
             "score": h["score"], "change": h["change"]} for i, h in enumerate(ranked)]
    return {"hypotheses": hyps, "timeline": _t(s, f"top hypothesis: {hyps[0]['cause']} ({hyps[0]['confidence']})")}


def similar(s: IncidentState) -> IncidentState:
    q = " ".join(t["template"] for t in s["logs"]["new_templates"][:2]) + " " + s["service"]
    res = retrieve(q, Principal("incident-bot", "incident bot", ["sre"]), top_n=8)
    pms = list(dict.fromkeys(h.symbol.split("-")[0] + "-" + h.symbol.split("-")[1] for h in res.hits if h.symbol.startswith("PM-")))
    return {"similar": pms[:3]}


def draft(s: IncidentState) -> IncidentState:
    top = s["hypotheses"][0]
    ch = top.get("change") or {}
    action = None
    if ch.get("kind") == "deploy" and ch.get("previous"):
        action = {"runbook": "rollback-deployment", "params": {"runbook": "rollback-deployment", "service": ch["service"], "to": ch["previous"]}}
    m = s["metrics"]
    brief = {
        "impact": f"{s['service']} error rate {m['error_rate_now']:.1f}% (SLO {cfg()['incident']['slo_error_rate_pct']}%) in {m['region']} "
                  f"since minute {s['anomaly']['minute']}",
        "timeline": s.get("timeline", []),
        "hypotheses": [{k: v for k, v in h.items() if k != "change"} for h in s["hypotheses"][:3]],
        "similar_incidents": s.get("similar", []),
        "proposed_action": action,
    }
    evidence = json.dumps({k: v for k, v in brief.items() if k != "timeline"}, default=str)
    out = generate("T1c", [{"role": "system", "content": "Write a concise incident brief from the evidence only."},
                           {"role": "user", "content": evidence}],
                   fallback=f"Impact: {brief['impact']}. Most likely cause: {top['cause']} (confidence {top['confidence']}). "
                            f"Evidence: {'; '.join(top['evidence']) or 'timing only'}. Similar: {', '.join(brief['similar_incidents']) or 'none'}. "
                            + (f"Proposed: run {action['runbook']} for {action['params']['service']} to {action['params']['to']} (needs SRE approval)."
                               if action else "No automated action proposed."))
    brief["summary"], brief["model"] = out.text, out.model
    upd = {"brief": brief, "timeline": _t(s, "brief drafted")}
    if action:
        upd["action_id"] = create_action("incident", s["thread_id"], "runbook.execute", action["params"], "incident-bot", "sre")
        brief["proposed_action"]["id"] = upd["action_id"]
    store().execute("UPDATE incidents SET status=?, brief=?, updated_at=? WHERE id=?",
                    ("awaiting_approval" if action else "brief_ready", json.dumps(brief, default=str), time.time(), s["incident_id"]))
    return upd


def human_gate(s: IncidentState) -> IncidentState:
    if not s.get("action_id"):
        return {"decision": {"decision": "none"}}
    d = interrupt({"type": "approval", "action_id": s["action_id"], "approver_role": "sre", "brief": s["brief"]})
    return {"decision": d}


def after_gate(s: IncidentState) -> str:
    return "execute" if s.get("decision", {}).get("decision") == "approve" else "postmortem"


def execute(s: IncidentState) -> IncidentState:
    d = s["decision"]
    dry = registry.call("runbook.execute", {**d["params"], "dry_run": True}, actor=d["approver"],
                        approval_token=d["dry_run_token"], idempotency_key=f"{s['action_id']}-dry")
    res = registry.call("runbook.execute", d["params"], actor=d["approver"], approval_token=d["token"], idempotency_key=s["action_id"])
    return {"execution": {"dry_run": dry, "result": res}, "attempts": s.get("attempts", 0) + 1,
            "timeline": _t(s, f"runbook {d['params']['runbook']} executed by {d['approver']}")}


def verify(s: IncidentState) -> IncidentState:
    v = registry.call("obs.slo_check", {"service": s["service"]})
    return {"verify": v, "timeline": _t(s, "SLO recovered" if v["recovered"] else "SLO still breached")}


def after_verify(s: IncidentState) -> str:
    return "postmortem" if s["verify"]["recovered"] or s.get("attempts", 0) >= 2 else "correlate"


def postmortem(s: IncidentState) -> IncidentState:
    top = s["hypotheses"][0]
    d = s.get("decision", {})
    if d.get("decision") == "approve":
        res = f"{d['params']['runbook']} to {d['params'].get('to')} approved by {d['approver']}"
    elif d.get("decision") == "reject":
        res = f"proposed action rejected by {d.get('approver')}; SRE investigating manually"
    else:
        res = "no automated action"
    text = (f"# Postmortem draft: {s['incident_id']}\n\n## Summary\n{top['cause']} caused {s['brief']['impact']}.\n\n"
            f"## Detection\nAlert {s['alert'].get('name')}; HELIX brief drafted automatically.\n\n## Resolution\n{res}. "
            f"Verified: {s.get('verify', {}).get('recovered', 'n/a')}.\n\n## Timeline\n" +
            "\n".join(f"- {t['ts']} {t['event']}" for t in s.get("timeline", [])) +
            f"\n\n## Similar incidents\n{', '.join(s.get('similar', [])) or 'none'}\n\n## Follow-ups\n- Contract test between checkout and card-vault token format\n"
            "- Alert on new ERROR log templates after deploys\n\n_Draft only: owner review required before publishing._")
    store().execute("UPDATE incidents SET status=?, updated_at=? WHERE id=?",
                    ("resolved" if s.get("verify", {}).get("recovered") else "mitigation_pending", time.time(), s["incident_id"]))
    return {"postmortem": text, "timeline": _t(s, "postmortem drafted")}


@lru_cache
def graph():
    g = StateGraph(IncidentState)
    nodes = [("receive", receive), ("gather_metrics", gather_metrics), ("gather_logs", gather_logs), ("gather_traces", gather_traces),
             ("gather_changes", gather_changes), ("correlate", correlate), ("similar", similar), ("draft", draft),
             ("human_gate", human_gate), ("execute", execute), ("verify", verify), ("postmortem", postmortem)]
    for n, f in nodes:
        g.add_node(n, f)
    g.add_edge(START, "receive")
    for a, b in [("receive", "gather_metrics"), ("gather_metrics", "gather_logs"), ("gather_logs", "gather_traces"),
                 ("gather_traces", "gather_changes"), ("gather_changes", "correlate"), ("correlate", "similar"),
                 ("similar", "draft"), ("draft", "human_gate"), ("execute", "verify")]:
        g.add_edge(a, b)
    g.add_conditional_edges("human_gate", after_gate, ["execute", "postmortem"])
    g.add_conditional_edges("verify", after_verify, ["postmortem", "correlate"])
    g.add_edge("postmortem", END)
    return g.compile(checkpointer=checkpointer())
