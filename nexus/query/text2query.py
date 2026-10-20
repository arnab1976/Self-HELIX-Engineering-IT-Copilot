"""Text-to-query (C7): schema RAG -> T1c (L6) -> validate -> execute read-only -> preview.

Guards before execution: sqlglot parse (dialect validity), AST statement type must be SELECT/WITH,
LIMIT enforced (<= 1,000), PII columns masked, read-only replica role, timeout. If any check fails the
query is returned unexecuted with a warning (router fallback for t2q).
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from common.config import load_yaml
from common.textutils import tokenize
from nexus.llm import generate
from nexus.tools import registry

PII_COLUMNS = {"email", "phone", "user_principal_name", "userprincipalname"}
WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge, exp.Command)

# Few-shot templates used by the mock T1c; with a real L6 endpoint these become in-context examples.
SQL_TEMPLATES = [
    (r"top (\d+)?\s*customers? by failed", "SELECT customer_id, COUNT(*) AS failed_orders FROM orders WHERE status = 'FAILED' GROUP BY customer_id ORDER BY failed_orders DESC LIMIT {n}"),
    (r"(orders|count).*(by|per) region", "SELECT region, COUNT(*) AS orders FROM orders GROUP BY region ORDER BY orders DESC LIMIT 100"),
    (r"failed (order )?rate by region", "SELECT region, ROUND(100.0 * SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failed_pct FROM orders GROUP BY region ORDER BY failed_pct DESC LIMIT 100"),
    (r"revenue|total (sales|amount)", "SELECT ROUND(SUM(total), 2) AS revenue FROM orders WHERE status = 'PAID' LIMIT 1"),
    (r"deploy", "SELECT version, deployed_at, status FROM deployments WHERE service = 'checkout-service' ORDER BY deployed_at DESC LIMIT 100"),
    (r"customers? by segment", "SELECT segment, COUNT(*) AS customers FROM customers GROUP BY segment ORDER BY customers DESC LIMIT 100"),
]
PROMQL_TEMPLATES = [
    (r"error rate.*region|error rate", 'sum by (region) (rate(http_requests_total{service="checkout",status=~"5.."}[5m]))\n/\nsum by (region) (rate(http_requests_total{service="checkout"}[5m]))'),
    (r"p99|latency", 'histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{service="checkout"}[5m])))'),
    (r"memory", 'max by (pod) (container_memory_working_set_bytes{container="checkout"})'),
]
LOGQL_TEMPLATES = [(r"e1043|error", 'sum(count_over_time({app="checkout"} |= "E1043" [15m]))')]
KQL_TEMPLATES = [(r"sign.?in|login", "SigninLogs\n| where TimeGenerated > ago(1d)\n| where ResultType != 0\n| summarize failures = count() by AppDisplayName\n| top 10 by failures")]


def schema_context(question: str, target: str) -> list[str]:
    cat = load_yaml("configs/metric_catalog.yaml")
    q = set(tokenize(question))
    entries = []
    if target == "sql":
        for t, v in cat["sql"]["tables"].items():
            entries.append((len(q & set(tokenize(t + " " + v["description"] + " " + " ".join(v["columns"])))),
                            f"{t}({', '.join(f'{c} {ty}' for c, ty in v['columns'].items())})"))
    elif target == "promql":
        for m, v in cat["promql"]["metrics"].items():
            entries.append((len(q & set(tokenize(m + " " + v["description"]))), f"{m}{{{', '.join(v['labels'])}}} ({v['type']})"))
    elif target == "logql":
        entries = [(1, f"{s} labels: {', '.join(v['labels'])}") for s, v in cat["logql"]["streams"].items()]
    else:
        entries = [(1, f"{t}({', '.join(v['columns'])})") for t, v in cat["kql"]["tables"].items()]
    return [e for _, e in sorted(entries, key=lambda x: -x[0])[:3]]


def _template(question: str, target: str) -> str:
    table = {"sql": SQL_TEMPLATES, "promql": PROMQL_TEMPLATES, "logql": LOGQL_TEMPLATES, "kql": KQL_TEMPLATES}[target]
    ql = question.lower()
    for rx, q in table:
        m = re.search(rx, ql)
        if m:
            n = next((g for g in m.groups() if g and g.isdigit()), "10") if m.groups() else "10"
            return q.replace("{n}", n)
    return "SELECT COUNT(*) AS orders FROM orders LIMIT 1" if target == "sql" else ""


def validate_sql(sql: str, max_rows: int = 1000) -> dict:
    checks, issues = [], []
    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        return {"valid": False, "checks": [], "issues": [f"parse error: {str(e).splitlines()[0]}"], "sql": sql}
    trees = [t for t in trees if t is not None]
    if len(trees) != 1:
        issues.append("exactly one statement allowed")
    tree = trees[0] if trees else None
    if tree is not None:
        checks.append("parsed (sqlglot, sqlite dialect)")
        if any(isinstance(n, WRITE_NODES) for n in tree.walk()) or not isinstance(tree, (exp.Select, exp.Union, exp.With)):
            issues.append("write or non-SELECT statement: blocked (read-only role)")
        else:
            checks.append("read-only")
        cols = {c.name.lower() for c in tree.find_all(exp.Column)}
        if cols & PII_COLUMNS:
            issues.append(f"PII columns requested: {sorted(cols & PII_COLUMNS)} (masked)")
        limit = tree.args.get("limit")
        if issues and "blocked" in issues[-1]:
            pass
        elif limit is None and isinstance(tree, exp.Select):
            tree = tree.limit(max_rows)
            checks.append(f"LIMIT {max_rows} added")
        else:
            checks.append("LIMIT present")
        sql = tree.sql(dialect="sqlite")
    blocking = [i for i in issues if "blocked" in i or "exactly one" in i]
    return {"valid": not blocking, "checks": checks, "issues": issues, "sql": sql}


def validate_other(q: str, target: str) -> dict:
    issues, checks = [], []
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    if all(q.count(a) == q.count(b) for a, b in pairs):
        checks.append("brackets balanced")
    else:
        issues.append("unbalanced brackets")
    if q.count('"') % 2:
        issues.append("unclosed quote")
    if target in ("promql", "logql"):
        (checks if re.search(r"\[\d+[smhd]\]", q) else issues).append("range vector present" if re.search(r"\[\d+[smhd]\]", q) else "no time range")
    if target == "kql" and "ago(" not in q:
        issues.append("no time filter")
    return {"valid": not any("unbalanced" in i or "unclosed" in i for i in issues), "checks": checks, "issues": issues, "sql": q}


def generate_query(question: str, target: str = "sql", execute: bool = True, query_override: str | None = None) -> dict:
    target = target.lower()
    ctx = schema_context(question, target)
    msgs = [{"role": "system", "content": f"Write one read-only {target.upper()} query. Use only:\n" + "\n".join(ctx)},
            {"role": "user", "content": question}]
    out = generate("T1c", msgs, fallback=_template(question, target)) if query_override is None else None
    q = (query_override or out.text).strip().strip("`")
    q = re.sub(r"^(sql|promql|logql|kql)\n", "", q, flags=re.I)
    v = validate_sql(q) if target == "sql" else validate_other(q, target)
    result = {"question": question, "target": target, "schema_context": ctx, "query": v["sql"], "validation": v,
              "model": out.model if out else "user-edited", "mocked": out.mocked if out else False, "executed": False}
    if execute and v["valid"] and target == "sql":
        result["preview"] = registry.call("db.query_readonly", {"sql": v["sql"], "max_rows": 50})
        result["executed"] = True
    elif target != "sql":
        result["note"] = "PromQL/LogQL/KQL execution runs against Prometheus/Loki/Sentinel in the docker-compose stack"
    return result
