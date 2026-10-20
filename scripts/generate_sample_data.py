"""Generate deterministic synthetic data for the MVP (spec 12.3 lists public datasets to swap in).

Outputs (data/generated/):
  tickets.csv            ~3,000 IT tickets with category, group, priority, CI, split (train/val/test)
  duplicate_pairs.csv    labelled ticket pairs for L8
  changes.csv            ~2,000 historical change requests with outcome for L9
  logs/checkout.log      one hour of checkout logs with an E1043 incident after a deploy
  metrics/checkout.json  per-minute error rate and p99 latency by region
  deploys.json           deploy and feature-flag history
  traces.json            slowest spans summary
  warehouse.db           SQLite warehouse for text-to-query (orders, customers, deployments)
  gold/retrieval.jsonl   questions with relevant files/symbols for retrieval evaluation
  gold/text2query.jsonl  questions with gold queries for execution accuracy
"""
from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "generated"
R = random.Random(2026)

CATEGORIES = {
    "Access > VPN": ("Network-L2", [
        "VPN keeps disconnecting every {n} minutes", "GlobalProtect VPN drops when working from home",
        "Cannot connect to VPN from hotel wifi", "VPN tunnel disconnects after Monday update",
        "VPN very slow and keeps reconnecting", "GlobalProtect stuck on connecting",
        "My VPN keeps disconnecting when working from home", "vpn not connecting since this morning"],
        ["I have to reconnect each time and lose my session.", "Started after the client update.",
         "Happens on home broadband, mobile hotspot works.", "Certificate error shown on connect."], "VPN-GW-0{g}"),
    "Access > Password/MFA": ("IAM-L1", [
        "Locked out of my account", "Need MFA reset after changing phone", "Okta authenticator not working",
        "Password expired cannot login", "Forgot password for SSO", "OTP not received on new phone",
        "I got a new phone and need to reset my MFA", "reset my mfa please", "my authenticator app stopped working",
        "I forgot my password and I am locked out"],
        ["Got a new phone and the authenticator no longer works.", "Account locked after too many attempts.",
         "Okta Verify push never arrives.", "Cannot log in to email or Teams."], "OKTA-PROD"),
    "Access > Request": ("IAM-Access", [
        "Need read access to {repo} repo", "Request access to shared drive {team}", "Grant permission to Jira project {team}",
        "Add me to the {team} group", "Need role change for {team} dashboard",
        "I need read access to the {repo} repo", "please give me write access to {repo}"],
        ["Joining the {team} team this week.", "Manager has approved verbally.", "Needed for the release next sprint."], "IAM-PORTAL"),
    "Software > Install": ("EUC-Software", [
        "Install {app} on my laptop", "Need {app} licence", "Request {app} installation", "{app} install failing",
        "Please install {app} on my laptop", "can you install {app} for me"],
        ["Required for the {team} project.", "The software centre shows an error.", "Please install the latest version."], "LT-{d}"),
    "Hardware > Laptop": ("EUC-Hardware", [
        "Laptop battery drains quickly", "Laptop screen flickering", "Keyboard keys not working", "Laptop very slow and overheating",
        "Charger not working"], ["Battery lasts under two hours.", "Started after a fall.", "Fan is always loud."], "LT-{d}"),
    "Email & Collaboration": ("M365-L2", [
        "Outlook calendar not syncing", "Teams meetings audio drops", "Mailbox full cannot send email", "Shared mailbox access missing",
        "Outlook keeps asking for password"], ["On phone and laptop.", "Since this morning.", "Colleagues see the same issue."], "EXO-01"),
    "Application > Incident": ("App-Support-L2", [
        "Checkout service returning 500", "Jira boards loading slowly", "Payments dashboard down", "Error 500 on payment step",
        "Confluence page not loading"], ["Affecting customers in production.", "Multiple users impacted.", "Error appears intermittently."], "{svc}"),
}
APPS = ["Docker Desktop", "IntelliJ IDEA", "Postman", "VS Code", "Zoom", "IntelliJ Ultimate", "Adobe Acrobat"]
TEAMS = ["payments", "platform", "data", "mobile", "security"]
REPOS = ["payments/checkout-service", "platform/infra-live", "docs/engineering-handbook"]
SVCS = ["CHECKOUT-PROD", "JIRA-PROD", "CONFLUENCE-PROD", "PAYDASH-PROD"]
REQUESTERS = [f"u.{n}" for n in "priya arjun kavya ravi meera john sara amit neha rahul anil divya kiran li omar".split()]


def fill(t: str) -> str:
    return t.format(n=R.choice([5, 10, 15]), repo=R.choice(REPOS), team=R.choice(TEAMS), app=R.choice(APPS),
                    g=R.choice([1, 2]), d=R.randint(1000, 9999), svc=R.choice(SVCS))


def priority_for(cat: str, text: str) -> str:
    s = text.lower()
    if "production" in s or "customers" in s or "500" in s or "down" in s:
        return "P1" if R.random() < 0.5 else "P2"
    if cat.startswith("Access > Password") or "locked" in s:
        return "P2" if R.random() < 0.3 else "P3"
    if cat.startswith("Access > Request") or cat.startswith("Software"):
        return "P4" if R.random() < 0.7 else "P3"
    return "P3"


def noisy(text: str) -> str:
    """Add realistic noise: typos, lowercase, missing words."""
    if R.random() < 0.25:
        text = text.lower()
    if R.random() < 0.15:
        words = text.split()
        if len(words) > 3:
            words.pop(R.randrange(len(words)))
        text = " ".join(words)
    if R.random() < 0.1:
        text = text.replace("VPN", "vpn ").replace("password", "pasword")
    return text


def gen_tickets(n: int = 3000) -> list[dict]:
    rows = []
    start = datetime(2026, 4, 1)
    cats = list(CATEGORIES)
    weights = [0.17, 0.22, 0.10, 0.12, 0.10, 0.14, 0.15]
    for i in range(n):
        cat = R.choices(cats, weights)[0]
        group, titles, descs, ci = CATEGORIES[cat]
        title = noisy(fill(R.choice(titles)))
        desc = noisy(fill(R.choice(descs)) + " " + fill(R.choice(descs)))
        # ~4% label noise, as in real ITSM history (reassignments)
        lab_cat = R.choice(cats) if R.random() < 0.04 else cat
        rows.append({
            "id": f"INC{40000 + i:07d}", "title": title, "description": desc,
            "requester": R.choice(REQUESTERS), "ci": fill(ci), "category": lab_cat,
            "assignment_group": CATEGORIES[lab_cat][0], "priority": priority_for(cat, title + " " + desc),
            "created_at": (start + timedelta(minutes=R.randint(0, 60 * 24 * 180))).isoformat(timespec="minutes"),
        })
    rows.sort(key=lambda r: r["created_at"])
    for j, r in enumerate(rows):  # time-based split
        r["split"] = "train" if j < 0.7 * n else "val" if j < 0.85 * n else "test"
    return rows


def gen_duplicate_pairs(tickets: list[dict], n: int = 1500) -> list[dict]:
    pairs = []
    by_ci: dict[str, list[dict]] = {}
    for t in tickets:
        by_ci.setdefault(t["ci"] + t["category"], []).append(t)
    groups = [g for g in by_ci.values() if len(g) > 3]
    for _ in range(n):
        if R.random() < 0.45:  # positives: same CI + category, close in time
            g = R.choice(groups)
            a = R.choice(g)
            near = [t for t in g if t is not a and abs((datetime.fromisoformat(t["created_at"]) - datetime.fromisoformat(a["created_at"])).total_seconds()) < 3 * 86400]
            if not near:
                continue
            pairs.append({"a": a["id"], "b": R.choice(near)["id"], "label": 1})
        elif R.random() < 0.35:  # time-close negatives: different issues raised the same day
            a = R.choice(tickets)
            near = [t for t in tickets if t["category"] != a["category"] and abs((datetime.fromisoformat(t["created_at"]) - datetime.fromisoformat(a["created_at"])).total_seconds()) < 86400]
            if not near:
                continue
            pairs.append({"a": a["id"], "b": R.choice(near)["id"], "label": 0})
        elif R.random() < 0.5:  # hard negatives: same CI but weeks apart (a different issue on the same system)
            g = R.choice(groups)
            a = R.choice(g)
            far = [t for t in g if abs((datetime.fromisoformat(t["created_at"]) - datetime.fromisoformat(a["created_at"])).total_seconds()) > 10 * 86400]
            if not far:
                continue
            pairs.append({"a": a["id"], "b": R.choice(far)["id"], "label": 0})
        else:
            a, b = R.sample(tickets, 2)
            pairs.append({"a": a["id"], "b": b["id"], "label": 0})
    return pairs


def gen_changes(n: int = 2000) -> list[dict]:
    rows = []
    start = datetime(2025, 10, 1)
    for i in range(n):
        f = {
            "services_touched": R.randint(1, 10), "lines_changed": int(10 ** R.uniform(1, 3.7)),
            "test_coverage": round(R.uniform(10, 95), 1), "author_failure_rate": round(R.uniform(0, 35), 1),
            "dependency_fanout": R.randint(0, 30), "peak_window": int(R.random() < 0.25),
            "has_rollback_plan": int(R.random() < 0.6), "change_type": R.choices(["standard", "normal", "emergency"], [0.3, 0.6, 0.1])[0],
        }
        z = (-2.9 + 0.22 * (f["services_touched"] - 2) + 0.45 * (math.log10(f["lines_changed"]) - 2) - 0.025 * (f["test_coverage"] - 60)
             + 0.05 * (f["author_failure_rate"] - 8) + 0.035 * (f["dependency_fanout"] - 6) + (0.6 if f["peak_window"] else -0.1)
             + (-0.5 if f["has_rollback_plan"] else 0.3) + {"standard": -0.4, "normal": 0.0, "emergency": 0.7}[f["change_type"]]
             + R.gauss(0, 0.3))
        f["failed"] = int(R.random() < 1 / (1 + math.exp(-z)))
        f["id"] = f"CHG{30000 + i:07d}"
        f["created_at"] = (start + timedelta(hours=i * 2.1)).isoformat(timespec="minutes")
        rows.append(f)
    return rows


def gen_logs_metrics() -> tuple[list[str], dict, list[dict], dict]:
    base = datetime(2026, 10, 7, 10, 0, 0)
    lines: list[str] = []
    metrics = {"service": "checkout-service", "start": base.isoformat(), "step_seconds": 60, "error_rate_pct": {}, "p99_latency_ms": []}
    regions = ["ap-south-1", "eu-west-1", "us-east-1"]
    for reg in regions:
        metrics["error_rate_pct"][reg] = []
    for m in range(60):
        t = base + timedelta(minutes=m)
        broken = m >= 42
        for reg in regions:
            er = (7.4 if reg == "ap-south-1" else 1.6) if broken else 0.6
            metrics["error_rate_pct"][reg].append(round(max(0.05, er + R.gauss(0, er * 0.08)), 3))
        metrics["p99_latency_ms"].append(round((540 if broken else 230) + R.gauss(0, 15), 1))
        for _ in range(40):
            sec = R.randint(0, 59)
            ts = (t + timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S")
            r = R.random()
            if broken and r < 0.45:
                lines.append(f"{ts} ERROR checkout-svc E1043 payment token decode failed user={R.randint(1000, 9999)}")
            elif r < 0.08:
                lines.append(f"{ts} WARN checkout-svc upstream gateway slow host=10.2.3.{R.randint(10, 30)} ms={R.randint(700, 990)}")
            else:
                lines.append(f"{ts} INFO checkout-svc request id={R.randint(10000, 99999)} path={R.choice(['/cart', '/pay', '/checkout'])} latency_ms={R.randint(30, 400 if broken else 90)}")
        if m in (44, 51):
            lines.append(f"{t.strftime('%Y-%m-%d %H:%M:%S')} INFO pod checkout-{R.randint(1000, 9999):x} restarted reason=OOMKilled")
    lines.sort()
    deploys = [
        {"ts": (base + timedelta(minutes=20)).isoformat(), "kind": "feature_flag", "service": "checkout-service", "detail": "new_cart_ui flipped on"},
        {"ts": (base + timedelta(minutes=40)).isoformat(), "kind": "deploy", "service": "checkout-service", "detail": "v3.18.0", "previous": "v3.17.2", "by": "argo-cd"},
        {"ts": (base - timedelta(days=2)).isoformat(), "kind": "deploy", "service": "card-vault", "detail": "v5.2.1", "previous": "v5.2.0", "by": "argo-cd"},
    ]
    traces = {"service": "checkout-service", "window": "10:30-11:00", "slowest_spans": [
        {"span": "checkout.charge", "p99_ms_before": 240, "p99_ms_after": 550, "error_rate_after": 0.07},
        {"span": "vault.decode_token", "p99_ms_before": 12, "p99_ms_after": 15, "error_rate_after": 0.46},
        {"span": "gateway.charge", "p99_ms_before": 180, "p99_ms_after": 270, "error_rate_after": 0.01}]}
    return lines, metrics, deploys, traces


def gen_warehouse(path: Path) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript("""
      CREATE TABLE customers (id TEXT PRIMARY KEY, segment TEXT, country TEXT);
      CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id TEXT, status TEXT, total REAL, region TEXT, created_at TEXT);
      CREATE TABLE deployments (id INTEGER PRIMARY KEY, service TEXT, version TEXT, deployed_at TEXT, status TEXT);
    """)
    custs = [(f"C-{10000 + i}", R.choice(["retail", "smb", "enterprise"]), R.choice(["IN", "GB", "US", "DE"])) for i in range(400)]
    con.executemany("INSERT INTO customers VALUES (?,?,?)", custs)
    start = datetime(2026, 9, 1)
    orders = []
    for i in range(12000):
        c = R.choice(custs)[0]
        st = R.choices(["PAID", "FAILED", "REFUNDED"], [0.9, 0.07, 0.03])[0]
        orders.append((i + 1, c, st, round(R.uniform(5, 900), 2), R.choice(["ap-south-1", "eu-west-1", "us-east-1"]),
                       (start + timedelta(minutes=R.randint(0, 60 * 24 * 40))).isoformat(timespec="minutes")))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    deps = [(1, "checkout-service", "v3.17.2", "2026-10-01T09:10", "succeeded"), (2, "card-vault", "v5.2.1", "2026-10-05T11:00", "succeeded"),
            (3, "checkout-service", "v3.18.0", "2026-10-07T10:40", "rolled_back")]
    con.executemany("INSERT INTO deployments VALUES (?,?,?,?,?)", deps)
    con.commit()
    con.close()


RETRIEVAL_GOLD = [
    {"q": "where is card validation done?", "relevant": ["CardValidator.validate_card", "CardValidator.luhn_ok"]},
    {"q": "who calls validate_card?", "relevant": ["CheckoutService.charge"]},
    {"q": "how is the cart total computed?", "relevant": ["Cart.total"]},
    {"q": "what does error E1043 mean?", "relevant": ["runbook-payment-errors.md#E1043 payment token decode failed"]},
    {"q": "how do I validate a shipping address postcode?", "relevant": ["validate_address"]},
    {"q": "what HTTP status does checkout return for validation errors?", "relevant": ["post_checkout", "ADR-012-card-tokenisation.md#Decision"]},
    {"q": "how do I reset MFA after changing my phone?", "relevant": ["KB0010021-mfa-reset.md#KB0010021: Reset MFA after changing your phone"]},
    {"q": "vpn keeps disconnecting fix", "relevant": ["KB0012034-vpn-disconnects.md#Resolution"]},
    {"q": "where are luhn checksums verified", "relevant": ["CardValidator.luhn_ok"]},
    {"q": "rules for outbound http calls timeout", "relevant": ["engineering-handbook.md#HTTP calls"]},
]

T2Q_GOLD = [
    {"q": "top 5 customers by failed orders", "target": "sql",
     "gold": "SELECT customer_id, COUNT(*) AS failed_orders FROM orders WHERE status = 'FAILED' GROUP BY customer_id ORDER BY failed_orders DESC LIMIT 5"},
    {"q": "number of orders by region", "target": "sql",
     "gold": "SELECT region, COUNT(*) AS orders FROM orders GROUP BY region ORDER BY orders DESC LIMIT 100"},
    {"q": "total revenue from paid orders", "target": "sql", "gold": "SELECT ROUND(SUM(total), 2) AS revenue FROM orders WHERE status = 'PAID' LIMIT 1"},
    {"q": "failed order rate by region", "target": "sql",
     "gold": "SELECT region, ROUND(100.0 * SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failed_pct FROM orders GROUP BY region ORDER BY failed_pct DESC LIMIT 100"},
    {"q": "list deployments of checkout-service", "target": "sql",
     "gold": "SELECT version, deployed_at, status FROM deployments WHERE service = 'checkout-service' ORDER BY deployed_at DESC LIMIT 100"},
    {"q": "customers by segment", "target": "sql", "gold": "SELECT segment, COUNT(*) AS customers FROM customers GROUP BY segment ORDER BY customers DESC LIMIT 100"},
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tickets = gen_tickets()
    write_csv(OUT / "tickets.csv", tickets)
    write_csv(OUT / "duplicate_pairs.csv", gen_duplicate_pairs(tickets))
    write_csv(OUT / "changes.csv", gen_changes())
    lines, metrics, deploys, traces = gen_logs_metrics()
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / "checkout.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "metrics").mkdir(exist_ok=True)
    (OUT / "metrics" / "checkout.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (OUT / "deploys.json").write_text(json.dumps(deploys, indent=1), encoding="utf-8")
    (OUT / "traces.json").write_text(json.dumps(traces, indent=1), encoding="utf-8")
    gen_warehouse(OUT / "warehouse.db")
    (OUT / "gold").mkdir(exist_ok=True)
    (OUT / "gold" / "retrieval.jsonl").write_text("\n".join(json.dumps(x) for x in RETRIEVAL_GOLD) + "\n", encoding="utf-8")
    (OUT / "gold" / "text2query.jsonl").write_text("\n".join(json.dumps(x) for x in T2Q_GOLD) + "\n", encoding="utf-8")
    print(f"generated: {len(tickets)} tickets, {len(lines)} log lines, warehouse + gold sets -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
