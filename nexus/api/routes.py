"""/v1 endpoints (spec 11). Every endpoint sits behind the gateway: bearer token, rate limit, trace_id."""
from __future__ import annotations

import json
import re
import time
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from common.config import cfg, repo_class, settings
from common.store import store
from common.tracing import current_trace_id, get_trace
from lexora.registry import list_cards
from nexus.api import schemas as S
from nexus.api.adapt import adapt_api
from nexus.api.chunking import chunking_api
from nexus.api.collection import collection_api
from nexus.completion.service import complete as do_complete
from nexus.gateway.auth import Principal, current_user, mint_dev_token
from nexus.gateway.ratelimit import limiter
from nexus.guardrails.dlp import scan_prompt
from nexus.ml.triage import get_triage
from nexus.observability import metrics as M
from nexus.query.text2query import generate_query
from nexus.rag.answer import answer as rag_answer
from nexus.rag.cache import cache
from nexus.rag.retriever import retrieve
from nexus.router import router as model_router
from nexus.router.policy import PolicyInput, policy_gate
from nexus.tools import approvals, registry
from nexus.tools.catalog import reset_sim
from nexus.workflows import runtime

api = APIRouter(prefix="/v1")
STATS: Counter = Counter()

IT_WORDS = re.compile(r"\b(password|mfa|otp|locked out|vpn|globalprotect|install|laptop|outlook|teams|access to|permission|"
                      r"authenticator|software|printer|wifi|battery|ticket)\b", re.I)
CODE_WORDS = re.compile(r"\b(function|class|method|calls?|code|repo|implemented|def|api|endpoint|test|bug|stack ?trace|error \w\d+|E\d{4})\b|`", re.I)


def _rl(user: Principal, kind: str = "default") -> None:
    limiter.check(user.sub, kind)


# ------------------------------------------------------------------------------------------- identity
@api.post("/dev/token", tags=["dev"])
def dev_token(body: dict):
    if not settings().is_dev:
        raise HTTPException(404)
    user = body.get("user", "u.arjun")
    try:
        return {"access_token": mint_dev_token(user), "token_type": "bearer", "user": user, "profile": cfg()["dev_users"][user]}
    except KeyError:
        raise HTTPException(400, f"unknown dev user; choose one of {list(cfg()['dev_users'])}")


@api.get("/me", tags=["identity"])
def me(user: Principal = Depends(current_user)):
    return {"sub": user.sub, "name": user.name, "roles": user.roles, "repo_classes": sorted(user.repo_classes), "manager": user.manager}


# ------------------------------------------------------------------------------------------- chat (code / knowledge / IT)
def _mode(req: S.ChatRequest, text: str) -> str:
    if req.mode != "auto":
        return req.mode
    if req.repo:
        return "code"
    if IT_WORDS.search(text):          # IT intents win: "I need access to <repo>" is an access request, not a code question
        return "it"
    if CODE_WORDS.search(text):
        return "code"
    return "knowledge"


@api.post("/chat", tags=["chat"])
def chat(req: S.ChatRequest, user: Principal = Depends(current_user)):
    _rl(user)
    text = req.messages[-1].content
    guard = scan_prompt(text, redact_secrets=req.redact_secrets)
    if guard.blocked:
        M.GUARDRAIL.labels("secret_blocked").inc()
        STATS["secret_blocked"] += 1
        store().audit(current_trace_id(), user.sub, "guardrail.block", guard.summary())
        raise HTTPException(400, {"blocked": True, "guard": guard.summary(), "trace_id": current_trace_id(),
                                  "message": "Request blocked before any model call. Remove the secret and rotate it."})
    if guard.injection_score >= cfg()["gateway"]["injection_block_threshold"]:
        M.GUARDRAIL.labels("injection").inc()
        STATS["injection_flagged"] += 1
    mode = _mode(req, text)

    if mode == "it":
        tid = req.thread_id or f"helpdesk-{user.sub}-{int(time.time())}"
        res = runtime.run("helpdesk", {"user": user.sub, "manager": user.manager, "device": user.device, "text": guard.text,
                                       "thread_id": tid}, tid)
        if res["state"].get("resolution"):
            M.DEFLECTION.labels(res["state"]["resolution"]).inc()
        reply = res["interrupt"]["message"] if res["interrupt"] and res["interrupt"].get("type") == "step_up" else res["state"].get("reply", "")
        return {"mode": "it", "answer": reply, "thread_id": tid, "status": res["status"], "interrupt": res["interrupt"],
                "intent": res["state"].get("intent"), "confidence": res["state"].get("confidence"),
                "tool_log": res["state"].get("tool_log", []), "guard": guard.summary(), "trace_id": current_trace_id()}

    rc = repo_class(req.repo)
    pol = policy_gate.evaluate(PolicyInput(roles=user.roles, repo_class=rc))
    if not pol.allow:
        M.GUARDRAIL.labels("acl_denied").inc()
        raise HTTPException(403, {"message": "; ".join(pol.reasons), "repo": req.repo, "repo_class": rc})
    route = "code_q" if mode == "code" else "knowledge"
    decision = model_router.route(route, user.roles, rc, req.complexity, context_tokens=4000)
    # no tier met Q_min -> degrade to the best tier the policy allows (never an external tier that was denied)
    tier = decision.tier or next((c.tier for c in decision.candidates if not c.status.startswith("denied")), "T1b")
    M.ROUTED.labels(route, tier).inc()
    acl = cache.acl_key(user.repo_classes)
    from nexus.rag.index import get_index

    version = get_index().version
    cached, sim = cache.get(guard.text + f"|{req.repo}", acl, version)
    if cached:
        return {**cached, "cache": {"hit": True, "similarity": round(sim, 3)}, "trace_id": current_trace_id()}
    res = retrieve(guard.text, user, req.repo)
    out = rag_answer(guard.text, res, tier)
    M.TOKENS_COST.labels(tier).inc(cfg()["tiers"][tier]["cost_per_req"])
    body = {"mode": mode, "answer": out["answer"], "citations": out["citations"], "symbol_checks": out["symbol_checks"],
            "hallucinated": out["hallucinated"], "model": out["model"], "mocked": out["mocked"], "route": decision.to_dict(),
            "retrieval": {"stages": res.stages, "acl_filtered": res.acl_filtered, "index_version": res.index_version,
                          "hits": [{"symbol": h.symbol, "citation": h.citation, "score": h.score, "injection_score": h.injection_score} for h in res.hits]},
            "guard": guard.summary()}
    cache.put(guard.text + f"|{req.repo}", acl, version, body)
    body["cache"] = {"hit": False, "similarity": round(sim, 3)}
    body["trace_id"] = current_trace_id()
    if req.stream:
        def sse():
            yield f"event: meta\ndata: {json.dumps({k: v for k, v in body.items() if k != 'answer'}, default=str)}\n\n"
            for i in range(0, len(body["answer"]), 32):
                yield f"event: token\ndata: {json.dumps(body['answer'][i:i + 32])}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return body


@api.post("/helpdesk/step-up", tags=["chat"])
def helpdesk_step_up(req: S.StepUpRequest, user: Principal = Depends(current_user)):
    """The user approves (or denies) the Okta Verify push for a pending helpdesk thread."""
    st = runtime.get_state("helpdesk", req.thread_id)
    if st["values"].get("user") != user.sub:
        raise HTTPException(403, "this verification belongs to another user")
    res = runtime.resume("helpdesk", req.thread_id, {"approved": req.approved})
    M.DEFLECTION.labels(res["state"].get("resolution", "unknown")).inc()
    return {"answer": res["state"].get("reply"), "status": res["status"], "tool_log": res["state"].get("tool_log", []),
            "trace_id": current_trace_id()}


# ------------------------------------------------------------------------------------------- completion
@api.post("/complete", tags=["code"])
def complete(req: S.CompleteRequest, user: Principal = Depends(current_user)):
    _rl(user, "completion")
    rc = repo_class(req.repo)
    if not user.can_read(rc):
        raise HTTPException(403, f"cannot read {req.repo} ({rc})")
    out = do_complete(req.prefix, req.suffix, req.path, req.language, req.repo, req.max_tokens)
    M.COMPLETIONS.labels("shown" if out["completion"] else ("filtered" if not out["post_filters"]["passed"] else "empty")).inc()
    STATS["completions_shown"] += bool(out["completion"])
    return {**out, "trace_id": current_trace_id()}


# ------------------------------------------------------------------------------------------- tickets
@api.post("/tickets/triage", tags=["itsm"])
def triage_ticket(req: S.TriageRequest, user: Principal = Depends(current_user)):
    _rl(user)
    res = runtime.run("triage", {"title": req.title, "description": req.description, "requester": req.requester or user.sub,
                                 "ci": req.ci, "create": req.create})
    st = res["state"]
    p = st["prediction"]
    return {"category": {"label": p["category"]["label"], "p": p["category"]["p"]},
            "assignment_group": {"label": p["assignment_group"]["label"], "p": p["assignment_group"]["p"]},
            "priority": {"label": p["priority"]["label"], "p": p["priority"]["p"]},
            "auto_route": st["decision"] == "auto_route", "decision": st["decision"],
            "duplicates": [d for d in st["duplicates"] if d["duplicate"]], "duplicate_candidates": st["duplicates"],
            "suggested_kb": [k["kb"] for k in st.get("kb", [])], "result": st["result"], "detail": p,
            "model": get_triage().version, "trace_id": current_trace_id()}


@api.get("/tickets", tags=["itsm"])
def list_tickets(state: str = "open", user: Principal = Depends(current_user)):
    q = "SELECT * FROM tickets WHERE state IN ('New','In Progress') ORDER BY created_at DESC LIMIT 100" if state == "open" else \
        "SELECT * FROM tickets ORDER BY created_at DESC LIMIT 200"
    return store().query(q)


# ------------------------------------------------------------------------------------------- PR review
@api.post("/reviews", tags=["code"])
def review(req: S.ReviewRequest, user: Principal = Depends(current_user)):
    _rl(user)
    if not user.can_read(repo_class(req.repo)):
        raise HTTPException(403, f"cannot read {req.repo}")
    inputs = {"repo": req.repo, "pr": req.pr or 0}
    if req.diff:
        inputs["diff"] = req.diff
    res = runtime.run("pr_review", inputs)
    st = res["state"]
    return {"comments": st["validated"], "posted": st["posted"], "context": st.get("context"), "model": st.get("model"),
            "trace_id": current_trace_id()}


# ------------------------------------------------------------------------------------------- text-to-query
@api.post("/query/generate", tags=["data"])
def query_generate(req: S.QueryRequest, user: Principal = Depends(current_user)):
    _rl(user)
    out = generate_query(req.question, req.target, req.execute, req.query)
    return {**out, "trace_id": current_trace_id()}


# ------------------------------------------------------------------------------------------- incidents & approvals
@api.post("/incidents/{incident_id}/brief", tags=["incident"])
def incident_brief(incident_id: str, req: S.IncidentRequest, user: Principal = Depends(current_user)):
    tid = f"incident-{incident_id}-{int(time.time())}"
    res = runtime.run("incident", {"incident_id": incident_id, "thread_id": tid, "alert": req.alert}, tid)
    st = res["state"]
    return {"incident_id": incident_id, "thread_id": tid, "status": res["status"], "brief": st.get("brief"),
            "metrics": {k: v for k, v in st.get("metrics", {}).items() if k != "series"}, "series": st.get("metrics", {}).get("series"),
            "anomaly": st.get("anomaly"), "logs": st.get("logs"), "interrupt": res["interrupt"], "trace_id": current_trace_id()}


@api.get("/incidents", tags=["incident"])
def incidents(user: Principal = Depends(current_user)):
    rows = store().query("SELECT * FROM incidents ORDER BY created_at DESC LIMIT 50")
    for r in rows:
        r["brief"] = json.loads(r["brief"] or "{}")
    return rows


@api.get("/actions", tags=["incident"])
def actions(user: Principal = Depends(current_user)):
    return runtime.pending_actions()


def _manager_of(sub: str) -> str | None:
    return cfg()["dev_users"].get(sub, {}).get("manager")


@api.post("/actions/{action_id}/approve", tags=["incident"])
def approve(action_id: str, req: S.ApproveRequest, user: Principal = Depends(current_user)):
    a = runtime.get_action(action_id)
    if not a:
        raise HTTPException(404, "unknown action")
    if a["status"] != "pending":
        raise HTTPException(409, f"action already {a['status']}")
    if a["approver_role"] == "sre" and not user.has_role("sre"):
        raise HTTPException(403, "an SRE must approve runbook actions")
    if a["approver_role"] == "manager" and not (user.has_role("manager") and _manager_of(a["requested_by"]) in (user.sub, None)):
        raise HTTPException(403, f"only {a['requested_by']}'s manager can approve this request")
    if req.decision == "reject":
        res = runtime.resume(a["workflow"], a["thread_id"], {"decision": "reject", "approver": user.sub, "comment": req.comment})
        runtime.close_action(action_id, "rejected", user.sub, None, {"comment": req.comment})
        return {"action_id": action_id, "status": "rejected", "workflow_status": res["status"], "state": _slim(res["state"]),
                "trace_id": current_trace_id()}
    params = dict(a["params"])
    if req.decision == "modify" and req.params:
        unknown = set(req.params) - set(params)
        if unknown:
            raise HTTPException(400, f"cannot add parameters {sorted(unknown)}")
        params.update(req.params)
    token = approvals.mint(action_id, a["tool"], params, user.sub)
    value = {"decision": "approve", "approver": user.sub, "token": token, "params": params, "comment": req.comment}
    if a["tool"] == "runbook.execute":
        value["dry_run_token"] = approvals.mint(action_id, a["tool"], {**params, "dry_run": True}, user.sub)
    res = runtime.resume(a["workflow"], a["thread_id"], value)
    runtime.close_action(action_id, "executed", user.sub, token, _slim(res["state"]))
    return {"action_id": action_id, "status": "executed", "approval_token": token, "params": params, "workflow_status": res["status"],
            "state": _slim(res["state"]), "trace_id": current_trace_id()}


def _slim(st: dict) -> dict:
    keep = ("reply", "resolution", "execution", "verify", "postmortem", "timeline", "tool_log")
    return {k: st[k] for k in keep if k in st}


# ------------------------------------------------------------------------------------------- change risk
@api.post("/changes/risk", tags=["change"])
def change_risk(req: S.ChangeRequest, user: Principal = Depends(current_user)):
    res = runtime.run("change_risk", {"change": req.model_dump()})
    st = res["state"]
    return {**st["risk"], "brief": st["brief"], "route": st["route"], "trace_id": current_trace_id()}


# ------------------------------------------------------------------------------------------- feedback, router, catalogue
@api.post("/feedback", tags=["feedback"])
def feedback(req: S.FeedbackRequest, user: Principal = Depends(current_user)):
    fid = store().add_feedback(req.trace_id, user.sub, req.signal, req.target, req.payload)
    M.FEEDBACK.labels(req.target, req.signal).inc()
    return {"feedback_id": fid, "stored": True, "used_for": {"review_comment": "L5 DPO pairs", "completion": "L4 acceptance + FIM data",
                                                             "answer": "L2/L3 hard negatives"}.get(req.target, "analytics")}


@api.post("/router/explain", tags=["platform"])
def router_explain(req: S.RouteRequest, user: Principal = Depends(current_user)):
    return model_router.route(req.route, user.roles, repo_class(req.repo), req.complexity, req.context_tokens).to_dict()


@api.post("/guard/scan", tags=["platform"])
def guard_scan(body: dict, user: Principal = Depends(current_user)):
    g = scan_prompt(body.get("text", ""), redact_secrets=bool(body.get("redact_secrets")))
    return {**g.summary(), "redacted": g.text, "injection_hits": g.injection_hits}


@api.get("/models", tags=["platform"])
def models(user: Principal = Depends(current_user)):
    return list_cards()


@api.get("/tools", tags=["platform"])
def tools(user: Principal = Depends(current_user)):
    return registry.catalogue()


@api.get("/traces/{trace_id}", tags=["platform"])
def trace(trace_id: str, user: Principal = Depends(current_user)):
    return {"trace_id": trace_id, "spans": get_trace(trace_id)}


@api.get("/dashboard", tags=["platform"])
def dashboard(user: Principal = Depends(current_user)):
    s = store()
    fb = s.query("SELECT target, signal, COUNT(*) n FROM feedback GROUP BY target, signal")
    by = {(r["target"], r["signal"]): r["n"] for r in fb}
    res = {k[1]: v for k, v in by.items() if k[0] == "helpdesk_resolution"}
    total_res = sum(res.values()) or 1
    rev_a, rev_d = by.get(("review_comment", "accept"), 0), by.get(("review_comment", "dismiss"), 0)
    comp_a = by.get(("completion", "accept"), 0)
    return {
        "helpdesk": {"resolutions": res, "deflection_rate": round(res.get("deflected", 0) / total_res, 3)},
        "reviews": {"accepted": rev_a, "dismissed": rev_d, "acceptance": round(rev_a / max(1, rev_a + rev_d), 3)},
        "completion": {"shown": STATS["completions_shown"], "accepted": comp_a, "acceptance": round(comp_a / max(1, STATS["completions_shown"]), 3)},
        "tickets": {"open": s.one("SELECT COUNT(*) n FROM tickets WHERE state IN ('New','In Progress')")["n"],
                    "created_by_helix": s.one("SELECT COUNT(*) n FROM tickets WHERE source='helix'")["n"]},
        "incidents": s.query("SELECT status, COUNT(*) n FROM incidents GROUP BY status"),
        "actions": s.query("SELECT status, COUNT(*) n FROM actions GROUP BY status"),
        "safety": {"secrets_blocked": STATS["secret_blocked"], "injection_flagged": STATS["injection_flagged"]},
        "audit_events": s.one("SELECT COUNT(*) n FROM audit")["n"],
    }


@api.post("/admin/reset-demo", tags=["dev"])
def reset_demo(user: Principal = Depends(current_user)):
    if not settings().is_dev:
        raise HTTPException(404)
    reset_sim()
    return {"reset": True, "note": "simulated checkout-service is back on v3.18.0 (unhealthy) for another incident demo"}


api.include_router(collection_api)
api.include_router(chunking_api)
api.include_router(adapt_api)
