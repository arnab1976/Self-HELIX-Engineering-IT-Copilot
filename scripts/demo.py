"""Scripted 15-minute demo (spec 12.4, weeks 11-12) against the in-process API. Run after `make bootstrap`.

    python scripts/demo.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.INFO)

from fastapi.testclient import TestClient  # noqa: E402

from nexus.api.main import app  # noqa: E402

c = TestClient(app)
_tok: dict[str, dict] = {}


def h(user: str) -> dict:
    if user not in _tok:
        _tok[user] = {"Authorization": "Bearer " + c.post("/v1/dev/token", json={"user": user}).json()["access_token"]}
    return _tok[user]


def title(t: str) -> None:
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def main() -> None:
    title("C5  Codebase Q&A with citations (engineer, internal repo -> T1b)")
    r = c.post("/v1/chat", json={"messages": [{"content": "where is card validation done and who calls it?"}],
                                 "repo": "payments/checkout-service"}, headers=h("u.arjun")).json()
    print(r["answer"])
    print("route:", r["route"]["tier"], "| ACL-filtered chunks:", r["retrieval"]["acl_filtered"], "| hallucinated:", r["hallucinated"])

    title("Gateway  secret in prompt -> blocked before any model call")
    r = c.post("/v1/chat", json={"messages": [{"content": 'checkout fails, config: AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"'}]}, headers=h("u.arjun"))
    print(r.status_code, json.dumps(r.json()["detail"]["guard"]["secrets"]))

    title("Policy gate  engineer asks about restricted payments/card-vault")
    r = c.post("/v1/chat", json={"messages": [{"content": "explain tokenise_card"}], "repo": "payments/card-vault"}, headers=h("u.arjun"))
    print(r.status_code, r.json()["detail"])

    title("C4  IDE completion (T1a, FIM)")
    r = c.post("/v1/complete", json={"repo": "payments/checkout-service", "path": "src/checkout/validators.py",
                                     "prefix": "def validate_card(card: Card) -> None:\n    if not card.number:\n        ", "suffix": "\n    return None\n"},
               headers=h("u.arjun")).json()
    print(repr(r["completion"]), f"{r['latency_ms']} ms", "context:", r["context_symbols"])

    title("C1  IT self-service: MFA reset with step-up verification (Priya)")
    r = c.post("/v1/chat", json={"messages": [{"content": "I got a new phone and need to reset my MFA"}]}, headers=h("u.priya")).json()
    print("bot:", r["answer"], "| intent:", r["intent"], r["confidence"])
    r = c.post("/v1/helpdesk/step-up", json={"thread_id": r["thread_id"], "approved": True}, headers=h("u.priya")).json()
    print("bot:", r["answer"])

    title("C2  VPN issue -> L8 links it to the open outage instead of a new ticket")
    r = c.post("/v1/chat", json={"messages": [{"content": "My VPN keeps disconnecting every few minutes when working from home"}]},
               headers=h("u.priya")).json()
    print("bot:", r["answer"])

    title("C10 Access request -> manager approval (Venkat) -> grant with bound token")
    r = c.post("/v1/chat", json={"messages": [{"content": "I need read access to the payments/checkout-service repo"}]}, headers=h("u.priya")).json()
    print("bot:", r["answer"])
    a = c.post(f"/v1/actions/{r['interrupt']['action_id']}/approve", json={"decision": "approve"}, headers=h("venkat.k")).json()
    print("bot:", a["state"]["reply"])

    title("C6  PR review #4182 (validators confirm, bot only COMMENTs)")
    r = c.post("/v1/reviews", json={"repo": "payments/checkout-service", "pr": 4182}, headers=h("u.arjun")).json()
    for cm in r["comments"]:
        print(f"  [{cm['severity']:6}] {cm['path']}:{cm['line']}  {cm['message']}")

    title("C7  Text-to-query (sqlglot-validated, read-only execution)")
    r = c.post("/v1/query/generate", json={"question": "failed order rate by region"}, headers=h("u.arjun")).json()
    print(r["query"])
    print(r["preview"]["columns"], r["preview"]["rows"])

    title("C8  Incident copilot: alert -> brief -> SRE approves rollback -> verify -> postmortem")
    c.post("/v1/admin/reset-demo", json={}, headers=h("sre.anita"))
    r = c.post("/v1/incidents/INC-2291/brief", json={}, headers=h("sre.anita")).json()
    print(r["brief"]["summary"])
    a = c.post(f"/v1/actions/{r['interrupt']['action_id']}/approve", json={"decision": "approve"}, headers=h("sre.anita")).json()
    print("verify:", a["state"]["verify"])
    print(a["state"]["postmortem"].split("## Timeline")[0])

    title("C9  Change risk (L9 + TreeSHAP)")
    r = c.post("/v1/changes/risk", json={"services_touched": 6, "lines_changed": 2400, "test_coverage": 35, "peak_window": 1,
                                         "has_rollback_plan": 0}, headers=h("u.arjun")).json()
    print(f"{r['probability']:.1%} {r['band']} -> {r['route']}\n{r['brief']}")

    title("Platform dashboard")
    print(json.dumps(c.get("/v1/dashboard", headers=h("u.arjun")).json(), indent=1))


if __name__ == "__main__":
    main()
