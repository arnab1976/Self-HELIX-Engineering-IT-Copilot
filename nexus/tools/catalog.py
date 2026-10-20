"""Tool catalogue (spec 7.4). MVP implementations are backed by SQLite and the generated data; each function
documents the production API it stands in for. Swap the body, keep the signature and ToolSpec."""
from __future__ import annotations

import json
import secrets as pysecrets
import sqlite3
import time
from pathlib import Path

from common.config import cfg, settings
from common.store import store
from nexus.tools.base import ToolError, registry

GEN = lambda *p: Path(settings().data_dir).joinpath("generated", *p)  # noqa: E731
CATALOGUE_APPS = {"Docker Desktop", "IntelliJ IDEA", "Postman", "VS Code", "Zoom"}
RUNBOOK_VERBS = {"rollback-deployment", "restart-pod", "scale-deployment"}


# ------------------------------------------------------------------------------------------- simulated environment
def _sim_path() -> Path:
    return settings().path("sim_state.json")


def sim_state() -> dict:
    p = _sim_path()
    if not p.exists():
        st = {"checkout-service": {"version": "v3.18.0", "healthy": False, "replicas": 6}}
        p.write_text(json.dumps(st))
    return json.loads(p.read_text())


def _save_sim(st: dict) -> None:
    _sim_path().write_text(json.dumps(st))


def reset_sim() -> None:
    _sim_path().unlink(missing_ok=True)


def _id(prefix: str) -> str:
    return f"{prefix}{int(time.time() * 1000) % 10_000_000:07d}"


# ------------------------------------------------------------------------------------------- ITSM (ServiceNow / Jira SM)
@registry.tool("itsm.create_ticket", "ServiceNow", write=True, description="POST /api/now/table/incident")
def create_ticket(title: str, description: str, requester: str, category: str, assignment_group: str, priority: str,
                  ci: str = "", source: str = "helix") -> dict:
    tid = _id("INC")
    store().execute("INSERT INTO tickets(id,title,description,requester,ci,category,assignment_group,priority,state,created_at,source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (tid, title, description, requester, ci, category, assignment_group, priority,
                                                       "New", time.time(), source))
    return {"ticket": tid, "state": "New", "assignment_group": assignment_group, "priority": priority}


@registry.tool("itsm.update_ticket", "ServiceNow", write=True, description="PATCH /api/now/table/incident/{sys_id}")
def update_ticket(ticket: str, add_affected_user: str | None = None, assignment_group: str | None = None,
                  state: str | None = None, duplicate_of: str | None = None) -> dict:
    t = store().one("SELECT * FROM tickets WHERE id=?", (ticket,))
    if not t:
        raise ToolError(f"ticket {ticket} not found")
    if add_affected_user:
        users = sorted(set(json.loads(t["affected_users"] or "[]")) | {add_affected_user})
        store().execute("UPDATE tickets SET affected_users=? WHERE id=?", (json.dumps(users), ticket))
    if assignment_group:
        store().execute("UPDATE tickets SET assignment_group=? WHERE id=?", (assignment_group, ticket))
    if state:
        store().execute("UPDATE tickets SET state=? WHERE id=?", (state, ticket))
    if duplicate_of:
        store().execute("UPDATE tickets SET duplicate_of=?, state='Resolved' WHERE id=?", (duplicate_of, ticket))
    return store().one("SELECT id, state, assignment_group, affected_users, duplicate_of FROM tickets WHERE id=?", (ticket,))


@registry.tool("itsm.get_ticket", "ServiceNow")
def get_ticket(ticket: str) -> dict:
    t = store().one("SELECT * FROM tickets WHERE id=?", (ticket,))
    if not t:
        raise ToolError(f"ticket {ticket} not found")
    return t


@registry.tool("itsm.request_access", "ServiceNow", write=True, description="Creates a RITM that needs manager approval")
def request_access(user: str, resource: str, role: str, manager: str, justification: str = "") -> dict:
    ritm = _id("RITM")
    return {"request": ritm, "user": user, "resource": resource, "role": role, "state": "awaiting_approval", "approver": manager}


@registry.tool("iam.grant_access", "GitHub / IdP groups", write=True, approval="manager",
               description="Grant after manager approval; requires an approval token bound to these exact parameters")
def grant_access(user: str, resource: str, role: str, request: str) -> dict:
    store().execute("INSERT OR REPLACE INTO access_grants VALUES (?,?,?,?)", (user, resource, role, time.time()))
    return {"granted": True, "user": user, "resource": resource, "role": role, "request": request}


# ------------------------------------------------------------------------------------------- IdP (Okta / Entra ID)
@registry.tool("idp.step_up", "Okta", description="POST /api/v1/authn/factors/{id}/verify (Okta Verify push)")
def step_up(user: str) -> dict:
    return {"challenge": "chl_" + pysecrets.token_hex(4), "factor": "okta_verify_push", "user": user, "state": "WAITING"}


@registry.tool("idp.reset_mfa", "Okta", write=True, approval="step_up", description="POST /api/v1/users/{id}/lifecycle/reset_factors")
def reset_mfa(user: str, challenge: str) -> dict:
    if not challenge.startswith("chl_"):
        raise ToolError("step-up verification required")
    return {"user": user, "factors_reset": True, "enrolment_link_sent": True}


@registry.tool("idp.reset_password", "Okta", write=True, approval="step_up", description="POST /api/v1/users/{id}/lifecycle/reset_password")
def reset_password(user: str, challenge: str) -> dict:
    if not challenge.startswith("chl_"):
        raise ToolError("step-up verification required")
    return {"user": user, "reset_link_sent": True}


# ------------------------------------------------------------------------------------------- MDM (Intune / Jamf)
@registry.tool("mdm.install_app", "Intune", write=True, description="Graph: POST deviceAppManagement/mobileApps/{id}/assign")
def install_app(device: str, app: str) -> dict:
    if app not in CATALOGUE_APPS:
        raise ToolError(f"{app} is not a catalogue app; it needs a ServiceNow request with manager approval")
    return {"device": device, "app": app, "state": "install_pushed", "eta_minutes": 10}


# ------------------------------------------------------------------------------------------- Git
@registry.tool("git.read_file", "GitHub")
def read_file(repo: str, path: str) -> dict:
    p = Path(settings().data_dir) / "sample" / "repos" / repo / path
    if not p.exists():
        raise ToolError(f"{repo}/{path} not found")
    return {"repo": repo, "path": path, "content": p.read_text()}


@registry.tool("git.diff", "GitHub", description="GET /repos/{repo}/pulls/{n} (diff media type)")
def diff(repo: str, pr: int) -> dict:
    p = Path(settings().data_dir) / "sample" / "pr" / f"pr_{pr}.diff"
    if not p.exists():
        raise ToolError(f"PR {pr} not found")
    return {"repo": repo, "pr": pr, "diff": p.read_text()}


@registry.tool("git.comment_pr", "GitHub", write=True, description="POST /repos/{repo}/pulls/{n}/reviews (event=COMMENT, never APPROVE)")
def comment_pr(repo: str, pr: int, comments: list) -> dict:
    return {"repo": repo, "pr": pr, "review_id": _id("RV"), "event": "COMMENT", "comments_posted": len(comments)}


# ------------------------------------------------------------------------------------------- Observability
@registry.tool("obs.query_metrics", "Prometheus", description="GET /api/v1/query_range")
def query_metrics(service: str = "checkout-service") -> dict:
    m = json.loads(GEN("metrics", "checkout.json").read_text())
    st = sim_state().get(service, {})
    if st.get("healthy"):   # after a successful rollback the live series recovers
        m["error_rate_pct"] = {k: v + [0.6, 0.55, 0.62, 0.58, 0.6] for k, v in m["error_rate_pct"].items()}
    return m


@registry.tool("obs.query_logs", "Loki", description="GET /loki/api/v1/query_range + Drain3 templates")
def query_logs(service: str = "checkout-service", start_minute: int = 0, end_minute: int = 60) -> dict:
    from lexora.prep.logs import mine_templates

    lines = GEN("logs", "checkout.log").read_text().splitlines()
    def minute(ln: str) -> int:
        return int(ln[14:16])
    base = [ln for ln in lines if minute(ln) < 30]
    window = [ln for ln in lines if start_minute <= minute(ln) < end_minute]
    base_t = {t["template"] for t in mine_templates(base, prefer_drain3=False)}
    tpls = mine_templates(window, prefer_drain3=False)
    for t in tpls:
        t["new_since_baseline"] = t["template"] not in base_t
    return {"service": service, "lines": len(window), "templates": tpls}


@registry.tool("obs.query_traces", "Tempo")
def query_traces(service: str = "checkout-service") -> dict:
    return json.loads(GEN("traces.json").read_text())


@registry.tool("obs.slo_check", "Prometheus", description="SLO query after remediation")
def slo_check(service: str = "checkout-service") -> dict:
    st = sim_state().get(service, {})
    target = cfg()["incident"]["slo_error_rate_pct"]
    current = 0.6 if st.get("healthy") else 7.4
    return {"service": service, "error_rate_pct": current, "slo_pct": target, "recovered": current < target, "version": st.get("version")}


# ------------------------------------------------------------------------------------------- Deploys / runbooks
@registry.tool("deploy.list_changes", "Argo CD")
def list_changes(service: str | None = None) -> list:
    d = json.loads(GEN("deploys.json").read_text())
    return [x for x in d if service is None or x["service"] == service]


@registry.tool("runbook.execute", "Rundeck", write=True, approval="sre", timeout_s=30,
               description="Allow-listed verbs only; needs an SRE approval token bound to runbook + params")
def runbook_execute(runbook: str, service: str, to: str | None = None, replicas: int | None = None, dry_run: bool = False) -> dict:
    if runbook not in RUNBOOK_VERBS:
        raise ToolError(f"runbook {runbook} is not allow-listed")
    log = [f"[dry-run={dry_run}] {runbook} {service} {to or replicas or ''}".strip()]
    if dry_run:
        return {"ok": True, "log": log + ["dry-run passed: 6/6 pods would be replaced"]}
    st = sim_state()
    svc = st.setdefault(service, {})
    if runbook == "rollback-deployment":
        svc["version"], svc["healthy"] = to, True
        log += [f"argocd app rollback {service} --revision {to}", "6/6 pods healthy"]
    elif runbook == "restart-pod":
        log += [f"kubectl rollout restart deploy/{service}"]
    elif runbook == "scale-deployment":
        svc["replicas"] = replicas
        log += [f"kubectl scale deploy/{service} --replicas={replicas}"]
    _save_sim(st)
    return {"ok": True, "log": log}


# ------------------------------------------------------------------------------------------- Knowledge / CMDB / on-call / data
@registry.tool("kb.search", "ServiceNow KB / Confluence")
def kb_search(query: str, k: int = 3) -> list:
    from nexus.gateway.auth import Principal
    from nexus.rag.retriever import retrieve

    res = retrieve(query, Principal("system", "system", ["employee"]), top_n=8)
    import re as _re

    out = []
    for h in res.hits:
        if h.symbol.startswith("KB"):
            m = _re.search(r"Category: ([^·\n]+)", h.text)
            out.append({"symbol": h.symbol, "citation": h.citation, "kb": h.symbol.split("-")[0], "category": m.group(1).strip() if m else None})
    return out[:k]


@registry.tool("kb.draft_article", "ServiceNow KB", write=True, description="Draft only; publication needs owner approval")
def kb_draft(title: str, body: str) -> dict:
    return {"draft": _id("KBD"), "title": title, "state": "draft"}


@registry.tool("oncall.who", "PagerDuty")
def oncall_who(service: str) -> dict:
    return {"service": service, "primary": "sre.anita", "secondary": "sre.rohit", "escalation": "payments-sre"}


@registry.tool("cmdb.lookup", "ServiceNow CMDB")
def cmdb_lookup(ci: str) -> dict:
    data = {"checkout-service": {"owner": "Payments Platform", "tier": 1, "depends_on": ["card-vault", "payment-gateway", "orders-db"]},
            "VPN-GW-02": {"owner": "Network", "tier": 1, "depends_on": ["okta"]}}
    return data.get(ci, {"owner": "unknown", "tier": 3, "depends_on": []})


@registry.tool("db.query_readonly", "Warehouse (read-only replica)", timeout_s=10)
def db_query_readonly(sql: str, max_rows: int = 1000) -> dict:
    con = sqlite3.connect(f"file:{GEN('warehouse.db')}?mode=ro", uri=True)
    deadline = time.time() + 5.0
    con.set_progress_handler(lambda: int(time.time() > deadline), 10_000)   # non-zero aborts the query (timeout)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows)
    finally:
        con.close()
    return {"columns": cols, "rows": [list(r) for r in rows], "row_count": len(rows)}
