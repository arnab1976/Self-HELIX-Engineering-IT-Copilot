# Architecture

## Request path (NEXUS)

```
IDE / Teams-Slack / ITSM webhook / PagerDuty
        │
        ▼
Gateway ── auth (OIDC; dev JWT) ── rate limit (token bucket) ── DLP: secrets (block/redact), PII (redact), injection score
        │                                                         nexus/gateway, nexus/guardrails
        ▼
Mode / intent ── L1 triage classifier for IT intents; heuristics for code vs knowledge        nexus/api/routes.py
        │
        ▼
Policy gate (OPA) ── role → readable repo classes; restricted repos never reach external tiers; write tools need tokens
        │                                                         configs/policy/helix.rego, nexus/router/policy.py
        ▼
Router ── argmax Q − λc·cost − λl·latency, s.t. Q ≥ Q_min, context ≤ window; completion fixed to T1a
        │                                                         nexus/router/router.py, configs/helix.yaml
        ├── RAG: scope (ACL) → BM25 + dense (L2) + code graph → RRF → rerank (L3) → cite → hallucinated-API check
        │        nexus/rag/*
        ├── Workflows (LangGraph): helpdesk, triage, pr_review, incident, change_risk        nexus/workflows/*
        │        └── Tools (MCP): timeouts, retries, circuit breakers, idempotency, approval tokens   nexus/tools/*
        └── Model tiers: T1a FIM · T1b code (L5) · T1c (L6/L7) · T2 general · T3 frontier      nexus/llm/providers.py
        │
        ▼
Validators (Semgrep-style, secrets, licence) → response with trace_id → feedback → LEXORA label queue
```

## LEXORA loop

```
ingest (git, ITSM, Confluence, logs)          lexora/ingest
  → secrets redact (drop if unresolved)       lexora/prep/secrets.py
  → PII anonymise (hashed surrogates)         lexora/prep/pii.py
  → licence detect (copyleft excluded from FIM only)   lexora/prep/licence.py
  → MinHash LSH dedupe                        lexora/prep/dedupe.py
  → structure-aware chunking + Drain templates        lexora/prep/chunking.py, logs.py
  → FIM samples (PSM/SPM, rate 0.5)           lexora/prep/fim.py
  → dataset version + scan report             var/datasets/<version>/
  → train L1..L9                              lexora/train/*
  → evaluate + release gates                  lexora/eval/*
  → model card (+ rollback target)            lexora/registry
```

## Where each spec section lives

| Spec | Code |
|---|---|
| 4.1 Adaptation decision | `configs/train/recipes.yaml`, `lexora/train/__init__.py` |
| 4.2 Nine artefacts | `lexora/train/l1_triage.py` … `l9_change_risk.py` |
| 4.3 Data pipeline | `lexora/prep/pipeline.py` |
| 4.4 Training recipes | `lexora/train/qlora.py`, `configs/train/*` |
| 4.5 Evaluation & gates | `lexora/eval/metrics.py`, `gates.py`, `configs/helix.yaml: release_gates` |
| 5 Capacity | `configs/helix.yaml: tiers`, `infra/terraform/main.tf` |
| 6.1 Router policy | `nexus/router/*`, `configs/policy/helix.rego` |
| 6.3 Retrieval | `nexus/rag/retriever.py` |
| 6.4 Completion | `nexus/completion/service.py`, `clients/vscode-extension` |
| 7.1 Incident workflow | `nexus/workflows/incident.py` |
| 7.2 Self-service | `nexus/workflows/helpdesk.py` |
| 7.3 PR review | `nexus/workflows/pr_review.py` |
| 7.4 Tool catalogue | `nexus/tools/catalog.py`, `mcp_server.py` |
| 8 Security, caching | `nexus/guardrails/*`, `nexus/rag/cache.py` |
| 9 Observability | `nexus/observability/metrics.py`, `common/tracing.py`, `/v1/dashboard` |
| 11 API | `nexus/api/routes.py`, `docs/API.md` |
| 12 Build guide | this repository, `docs/BUILD_GUIDE.md` |

## Key design decisions

- **Model cards are the contract.** The router reads task tags, gate result and permitted data classes from the
  registry; LEXORA can ship a new version without NEXUS code changes, and every card names its rollback target.
- **Repository classification is a hard policy input**, evaluated before router scoring, so a restricted repo
  cannot fall back to an external model even under load.
- **Approval tokens are HMACs over (action id, tool, exact params, approver, expiry).** A write tool rejects any
  call whose parameters differ from what the approver saw; dry runs need their own token.
- **Retrieved content is untrusted data.** It is wrapped in `<untrusted>` blocks, injection-scored, and tool
  allow-lists per workflow step limit what a successful injection could do.
- **Mock provider = evidence-based fallback.** Every service computes a grounded fallback answer from retrieved
  evidence; a real model improves the wording, and if it fails the system degrades to the evidence.
