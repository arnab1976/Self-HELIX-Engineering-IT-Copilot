"""End-to-end through the HTTP API: the five agents and the copilots."""


def test_learning_portal_routes_and_application_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "HELIX Engineering" in response.text
    assert "17-step" in response.text
    assert client.get("/learning").status_code == 200
    portal = client.get("/portal/")
    assert portal.status_code == 200
    assert "Learning portal" in portal.text


def test_auth_required(client):
    assert client.post("/v1/chat", json={"messages": [{"content": "hi"}]}).status_code == 401


def test_code_chat_with_citations_and_cache(client, auth):
    body = {"messages": [{"content": "where is card validation done and who calls it?"}], "repo": "payments/checkout-service"}
    r = client.post("/v1/chat", json=body, headers=auth("u.arjun")).json()
    assert r["mode"] == "code" and r["route"]["tier"] == "T1b"
    assert any("validators.py" in c for c in r["citations"]) and not r["hallucinated"]
    assert client.post("/v1/chat", json=body, headers=auth("u.arjun")).json()["cache"]["hit"]


def test_secret_blocked_and_restricted_repo_denied(client, auth):
    r = client.post("/v1/chat", json={"messages": [{"content": 'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE" why?'}]}, headers=auth("u.arjun"))
    assert r.status_code == 400 and r.json()["detail"]["blocked"]
    r = client.post("/v1/chat", json={"messages": [{"content": "explain tokenise_card"}], "repo": "payments/card-vault"}, headers=auth("u.arjun"))
    assert r.status_code == 403


def test_helpdesk_mfa_with_step_up(client, auth):
    r = client.post("/v1/chat", json={"messages": [{"content": "I got a new phone and need to reset my MFA"}]}, headers=auth("u.priya")).json()
    assert r["mode"] == "it" and r["interrupt"]["type"] == "step_up"
    r2 = client.post("/v1/helpdesk/step-up", json={"thread_id": r["thread_id"], "approved": True}, headers=auth("u.priya")).json()
    assert "MFA factors were reset" in r2["answer"]


def test_access_request_needs_the_right_manager(client, auth):
    r = client.post("/v1/chat", json={"messages": [{"content": "I need read access to the payments/checkout-service repo"}]},
                    headers=auth("u.priya")).json()
    aid = r["interrupt"]["action_id"]
    assert client.post(f"/v1/actions/{aid}/approve", json={"decision": "approve"}, headers=auth("sre.anita")).status_code == 403
    ok = client.post(f"/v1/actions/{aid}/approve", json={"decision": "approve"}, headers=auth("venkat.k")).json()
    assert "You now have read access" in ok["state"]["reply"]


def test_vpn_ticket_is_linked_as_duplicate(client, auth):
    r = client.post("/v1/tickets/triage", json={"title": "VPN disconnects after Monday update",
                                                "description": "GlobalProtect drops every few minutes when working from home",
                                                "ci": "VPN-GW-02"}, headers=auth("u.priya")).json()
    assert r["decision"].startswith("duplicate_of:") and r["result"]["action"] == "linked_to_existing"


def test_completion(client, auth):
    r = client.post("/v1/complete", json={"repo": "payments/checkout-service", "path": "src/checkout/validators.py",
                                          "prefix": "def validate_card(card):\n    if not card.number:\n        ", "suffix": "\n"},
                    headers=auth("u.arjun")).json()
    assert r["completion"] == 'raise ValidationError("card number is required")' and r["post_filters"]["passed"]


def test_pr_review_finds_sql_injection_and_secret(client, auth):
    r = client.post("/v1/reviews", json={"repo": "payments/checkout-service", "pr": 4182}, headers=auth("u.arjun")).json()
    high = [c for c in r["comments"] if c["severity"] == "high"]
    assert any("SQL" in c["message"] for c in high) and any("Stripe" in c["message"] for c in high)
    assert r["posted"]["event"] == "COMMENT"


def test_text_to_query_executes_read_only(client, auth):
    r = client.post("/v1/query/generate", json={"question": "top 5 customers by failed orders"}, headers=auth("u.arjun")).json()
    assert r["executed"] and len(r["preview"]["rows"]) == 5
    r = client.post("/v1/query/generate", json={"question": "x", "query": "DELETE FROM orders"}, headers=auth("u.arjun")).json()
    assert not r["executed"] and not r["validation"]["valid"]


def test_incident_end_to_end_with_sre_approval(client, auth):
    client.post("/v1/admin/reset-demo", json={}, headers=auth("sre.anita"))
    r = client.post("/v1/incidents/INC-9001/brief", json={}, headers=auth("sre.anita")).json()
    assert r["status"] == "waiting_for_human"
    assert r["brief"]["hypotheses"][0]["cause"].startswith("Deploy checkout-service v3.18.0")
    aid = r["interrupt"]["action_id"]
    assert client.post(f"/v1/actions/{aid}/approve", json={}, headers=auth("u.arjun")).status_code == 403
    done = client.post(f"/v1/actions/{aid}/approve", json={"decision": "approve"}, headers=auth("sre.anita")).json()
    assert done["workflow_status"] == "completed" and done["state"]["verify"]["recovered"]
    assert "Postmortem draft" in done["state"]["postmortem"]


def test_change_risk(client, auth):
    r = client.post("/v1/changes/risk", json={"services_touched": 8, "lines_changed": 3000, "test_coverage": 20, "peak_window": 1,
                                              "has_rollback_plan": 0}, headers=auth("u.arjun")).json()
    assert r["band"] == "high" and r["route"] == "full CAB review" and r["factors"]


def test_feedback_dashboard_models(client, auth):
    assert client.post("/v1/feedback", json={"trace_id": "tr_1", "signal": "accept", "target": "review_comment"},
                       headers=auth("u.arjun")).json()["stored"]
    d = client.get("/v1/dashboard", headers=auth("u.arjun")).json()
    assert d["reviews"]["accepted"] >= 1
    assert len(client.get("/v1/models", headers=auth("u.arjun")).json()) >= 8
    assert client.get("/healthz").json()["status"] == "ok"
