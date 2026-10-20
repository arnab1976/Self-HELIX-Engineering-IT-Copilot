"""IT self-service agent (C1, C10; spec 7.2).

classify intent (L1) --+-- password_mfa ---> step_up --> [human: user approves push] --> reset_mfa --> close
                       +-- software_install -> install_app (catalogue only) ----------------------> close
                       +-- access_request --> request_access --> [human: manager approves] --> grant_access --> close
                       +-- other / low confidence -> triage sub-flow (duplicates, route, create ticket) -> close
Resolution outcome feeds L1 retraining and the deflection metric.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common.store import store
from common.tracing import current_trace_id
from nexus.ml.triage import intent_of
from nexus.tools import ToolError, registry
from nexus.tools.catalog import CATALOGUE_APPS
from nexus.workflows import triage as triage_wf
from nexus.workflows.runtime import checkpointer, create_action

LOW_CONFIDENCE = 0.35          # chat messages are shorter than ticket text; below this, escalate


class HelpdeskState(TypedDict, total=False):
    user: str
    manager: str
    device: str
    text: str
    thread_id: str
    intent: str
    confidence: float
    prediction: dict
    challenge: str
    verified: bool
    action_id: str
    approval: dict
    tool_log: list
    reply: str
    resolution: str            # deflected | escalated | pending | denied
    ticket: str


def _log(s: HelpdeskState, entry: dict) -> list:
    return [*s.get("tool_log", []), entry]


def classify(s: HelpdeskState) -> HelpdeskState:
    intent, p, pred = intent_of(s["text"])
    if p < LOW_CONFIDENCE:
        intent = "other"
    return {"intent": intent, "confidence": p, "prediction": pred, "tool_log": [{"step": "intent (L1)", "intent": intent, "p": p}]}


def route(s: HelpdeskState) -> str:
    return {"password_mfa": "step_up", "software_install": "install_app", "access_request": "request_access"}.get(s["intent"], "escalate")


# ---------------------------------------------------------------- password / MFA
def step_up(s: HelpdeskState) -> HelpdeskState:
    ch = registry.call("idp.step_up", {"user": s["user"]}, actor="helpdesk-agent")
    return {"challenge": ch["challenge"], "tool_log": _log(s, {"tool": "idp.step_up", "result": ch})}


def verify_gate(s: HelpdeskState) -> HelpdeskState:
    decision = interrupt({"type": "step_up", "message": "Okta Verify push sent. Approve it on your device to continue.",
                          "challenge": s["challenge"], "user": s["user"]})
    return {"verified": bool(decision.get("approved"))}


def reset(s: HelpdeskState) -> HelpdeskState:
    if not s.get("verified"):
        return {"reply": "Verification was denied, so nothing was changed. If that wasn't you, contact the security desk.",
                "resolution": "denied", "tool_log": _log(s, {"step": "step-up denied"})}
    tool = "idp.reset_password" if re.search(r"password|locked", s["text"], re.I) and "mfa" not in s["text"].lower() else "idp.reset_mfa"
    res = registry.call(tool, {"user": s["user"], "challenge": s["challenge"]}, actor="helpdesk-agent")
    msg = ("Done. Your MFA factors were reset and an enrolment link is on its way. Enrol the new phone within 24 hours."
           if tool == "idp.reset_mfa" else "Done. A password reset link has been sent to your registered email.")
    return {"reply": msg, "resolution": "deflected", "tool_log": _log(s, {"tool": tool, "result": res})}


# ---------------------------------------------------------------- software
def install_app(s: HelpdeskState) -> HelpdeskState:
    app = next((a for a in sorted(CATALOGUE_APPS | {"IntelliJ Ultimate", "Adobe Acrobat"}, key=len, reverse=True)
                if a.lower() in s["text"].lower()), None)
    if not app:
        return {"reply": "Which application do you need? Catalogue apps: " + ", ".join(sorted(CATALOGUE_APPS)) + ".", "resolution": "pending"}
    try:
        res = registry.call("mdm.install_app", {"device": s.get("device") or "LT-0000", "app": app}, actor="helpdesk-agent")
        return {"reply": f"{app} is in the approved catalogue. Installation was pushed through Intune and should finish in about 10 minutes.",
                "resolution": "deflected", "tool_log": _log(s, {"tool": "mdm.install_app", "result": res})}
    except ToolError as e:
        return {"reply": f"{app} isn't a catalogue app, so it needs a request with manager approval. I'll raise a ticket.",
                "resolution": "escalated", "tool_log": _log(s, {"tool": "mdm.install_app", "error": str(e)})}


# ---------------------------------------------------------------- access request (C10)
def request_access(s: HelpdeskState) -> HelpdeskState:
    m = re.search(r"([\w-]+/[\w.-]+)", s["text"])
    resource = m.group(1) if m else "unknown-resource"
    role = "write" if re.search(r"\bwrite|push|admin\b", s["text"], re.I) else "read"
    req = registry.call("itsm.request_access", {"user": s["user"], "resource": resource, "role": role, "manager": s.get("manager") or "manager"},
                        actor="helpdesk-agent")
    params = {"user": s["user"], "resource": resource, "role": role, "request": req["request"]}
    aid = create_action("helpdesk", s["thread_id"], "iam.grant_access", params, s["user"], "manager")
    return {"action_id": aid, "tool_log": _log(s, {"tool": "itsm.request_access", "result": req}),
            "reply": f"I've raised {req['request']} for {role} access to {resource}. It needs approval from your manager ({s.get('manager')})."}


def manager_gate(s: HelpdeskState) -> HelpdeskState:
    decision = interrupt({"type": "approval", "action_id": s["action_id"], "approver_role": "manager", "approver": s.get("manager"),
                          "message": "Manager approval required for access request"})
    return {"approval": decision}


def grant(s: HelpdeskState) -> HelpdeskState:
    a = s.get("approval") or {}
    if a.get("decision") != "approve":
        return {"reply": "Your manager rejected the access request. No access was granted.", "resolution": "denied"}
    res = registry.call("iam.grant_access", a["params"], actor=a.get("approver", "manager"), approval_token=a["token"])
    return {"reply": f"Approved by {a.get('approver')}. You now have {a['params']['role']} access to {a['params']['resource']}.",
            "resolution": "deflected", "tool_log": _log(s, {"tool": "iam.grant_access", "result": res})}


# ---------------------------------------------------------------- everything else
def escalate(s: HelpdeskState) -> HelpdeskState:
    title = s["text"][:90]
    t = triage_wf.graph().invoke({"title": title, "description": s["text"], "requester": s["user"],
                                  "ci": "VPN-GW-02" if "vpn" in s["text"].lower() else "", "create": True},
                                 {"configurable": {"thread_id": s["thread_id"] + "-triage"}})
    res = t["result"]
    kb = t.get("kb") or []
    tip = f" Meanwhile, see knowledge article {kb[0]['kb']}." if kb else ""
    if res["action"] == "linked_to_existing":
        reply = f"This matches an issue already being worked: {res['ticket']}. I've added you as an affected user so you get updates.{tip}"
    else:
        reply = f"I've raised {res['ticket']}, routed to {res['assignment_group']} ({res['priority']}) with a summary of our conversation.{tip}"
    return {"reply": reply, "resolution": "escalated", "ticket": res.get("ticket"),
            "tool_log": _log(s, {"workflow": "triage", "decision": t["decision"], "duplicates": t["duplicates"][:3], "result": res})}


def close(s: HelpdeskState) -> HelpdeskState:
    store().add_feedback(current_trace_id(), s["user"], s.get("resolution", "unknown"), "helpdesk_resolution",
                         {"intent": s.get("intent"), "confidence": s.get("confidence")})
    return {}


@lru_cache
def graph():
    g = StateGraph(HelpdeskState)
    for n, f in [("classify", classify), ("step_up", step_up), ("verify_gate", verify_gate), ("reset", reset),
                 ("install_app", install_app), ("request_access", request_access), ("manager_gate", manager_gate),
                 ("grant", grant), ("escalate", escalate), ("close", close)]:
        g.add_node(n, f)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route, ["step_up", "install_app", "request_access", "escalate"])
    g.add_edge("step_up", "verify_gate")
    g.add_edge("verify_gate", "reset")
    g.add_edge("reset", "close")
    g.add_conditional_edges("install_app", lambda s: "escalate" if s.get("resolution") == "escalated" else "close", ["escalate", "close"])
    g.add_edge("request_access", "manager_gate")
    g.add_edge("manager_gate", "grant")
    g.add_edge("grant", "close")
    g.add_edge("escalate", "close")
    g.add_edge("close", END)
    return g.compile(checkpointer=checkpointer())
