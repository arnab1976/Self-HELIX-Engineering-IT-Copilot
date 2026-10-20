# API (spec 11)

All endpoints require `Authorization: Bearer <token>` and return `x-trace-id`. Dev tokens: `POST /v1/dev/token {"user": "u.arjun"}`.
Interactive docs at `/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat` | POST (JSON or SSE with `stream: true`) | Code, knowledge and IT chat. `mode`: auto, code, knowledge, it |
| `/v1/helpdesk/step-up` | POST | User approves or denies the Okta Verify push for a helpdesk thread |
| `/v1/complete` | POST | IDE inline completion |
| `/v1/tickets/triage` | POST | Classify, dedupe and route a ticket |
| `/v1/tickets` | GET | Open tickets |
| `/v1/reviews` | POST | Review a PR (`pr` number or raw `diff`) |
| `/v1/query/generate` | POST | Text-to-query; pass `query` to validate/run an edited query |
| `/v1/incidents/{id}/brief` | POST | Run the incident workflow up to the human gate |
| `/v1/incidents` | GET | Incident list with briefs |
| `/v1/actions` | GET | Pending human approvals |
| `/v1/actions/{id}/approve` | POST | `decision`: approve, modify (with `params`), reject. Mints the bound approval token and resumes the workflow |
| `/v1/changes/risk` | POST | Change-risk score, TreeSHAP factors, CAB brief |
| `/v1/feedback` | POST | accept / dismiss / thumbs / edit on an answer, completion or review comment |
| `/v1/router/explain` | POST | Router candidates, scores and policy reasons |
| `/v1/guard/scan` | POST | Run the gateway guardrails on text |
| `/v1/models` | GET | Model registry (cards) |
| `/v1/tools` | GET | Tool catalogue |
| `/v1/traces/{trace_id}` | GET | Spans recorded for a trace |
| `/v1/dashboard` | GET | KPIs: deflection, acceptance, safety, incidents |
| `/healthz`, `/metrics` | GET | Health, Prometheus |

## Examples

```bash
TOKEN=$(curl -s -XPOST localhost:8000/v1/dev/token -H 'content-type: application/json' -d '{"user":"u.arjun"}' | jq -r .access_token)

curl -s localhost:8000/v1/complete -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{
  "repo": "payments/checkout-service", "path": "src/checkout/validators.py", "language": "python",
  "prefix": "def validate_card(card: Card) -> None:\n    if not card.number:\n        ", "suffix": "\n    return None\n"}'
# {"completion": "raise ValidationError(\"card number is required\")", "model": "...", "latency_ms": ..., "trace_id": "tr_..."}

curl -s localhost:8000/v1/tickets/triage -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"title": "VPN disconnects after Monday update", "description": "GlobalProtect drops every few minutes", "ci": "VPN-GW-02"}'
```
