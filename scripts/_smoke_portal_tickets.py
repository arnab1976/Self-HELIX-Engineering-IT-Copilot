"""Smoke: portal tickets merge into Collect tickets simulation."""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def main() -> None:
    body = {
        "mode": "simulation",
        "sources": ["tickets"],
        "sample_text": "",
        "filename": "incident.txt",
        "dedupe_threshold": 0.82,
        "portal_tickets": [
            {
                "id": "INC-P-SMOKE-0001",
                "title": "Portal chatbot MFA reset test",
                "description": "Created from Agent 01 self-service portal smoke test.",
                "category": "Access > Password/MFA",
                "assignment_group": "IAM-L1",
                "priority": "P2",
                "requester": "demo.user@helix.local",
                "ci": "OKTA-PROD",
                "split": "train",
                "created_at": "2026-10-21T18:00:00Z",
            }
        ],
    }
    req = urllib.request.Request(
        BASE + "/v1/lexora/collect/run",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.load(resp)
    print(
        "version",
        result.get("dataset_version"),
        "portal_in",
        result.get("portal_tickets_in"),
        "docs_in",
        result.get("documents_in"),
    )
    text = (result.get("handoff") or {}).get("text") or ""
    assert result.get("portal_tickets_in") == 1
    assert "INC-P-SMOKE-0001" in text
    assert "employee_self_service_portal" in text
    print("OK")


if __name__ == "__main__":
    main()
