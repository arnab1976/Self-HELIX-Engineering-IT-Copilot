# HELIX: Enterprise Engineering & IT Copilot

Reference implementation of **UC-3** from the *LEXORA & NEXUS Solution Architecture and Build Specification v1.0*
(Feuji AI Practice, October 2026): one copilot for 100,000 employees, covering IT self-service, ticket triage,
IDE completion, codebase Q&A, PR review, text-to-query, incident response and change risk.

- **LEXORA** (offline) cleans engineering data, trains the nine model artefacts (L1–L9), evaluates them against
  release gates and registers model cards.
- **NEXUS** (online) is the governed gateway: identity, DLP and secret scanning, policy gate, router, hybrid RAG,
  MCP tools with approval tokens, and five LangGraph agents.

Everything runs on a laptop with **no GPU and no external services**: synthetic data, CPU baselines for L1/L8/L9,
an in-process index, and a deterministic mock provider for any model tier without an endpoint. Point a tier at
Ollama or vLLM and the same code path uses the real model.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
make install            # pip install -r requirements.txt
make bootstrap          # synthetic data -> LEXORA pipeline -> train L1/L8/L9 -> index -> evaluate (~10 s)
make run                # NEXUS API on http://localhost:8000
```

Then open:
- **http://localhost:8000/**: standalone-style interactive learning portal; choose **Start journey** to begin the 17-step architecture labs
- **http://localhost:8000/portal**: API-backed application with one tab per capability and a persona switcher
- **http://localhost:8000/learning**: direct link to the learning portal
- **http://localhost:8000/docs**: OpenAPI (dev: `POST /v1/dev/token {"user": "u.arjun"}` then *Authorize*)

### Agent 1: Collect & sanitise engineering data

Open **Step 1 · Collect & sanitise engineering data** in the learning portal. The agent starts by stating its
purpose, objective, and outputs, then describes its source-connector, privacy/secret-steward, licence/quality,
and dataset-registrar roles.

- **Interactive lab (simulation only)** provides a separate editable example for code repositories, ITSM
  tickets, knowledge/runbooks, application logs, and metrics/traces. Choose one category and run the LEXORA
  preparation pipeline to see its collection, secret scan, PII redaction, licence, dedupe, and registration
  stages. No live source is contacted from this lab.
- **Sources & data collected (live only)** lists the real source connectors. Select a source card to reveal
  its API endpoint and server-side connection status, select configured sources, then run the live pipeline.
  Configure credentials and source identifiers in the API server environment; never enter tokens or passwords
  in the browser. GitHub additionally needs an `owner/repository`. Metrics/traces currently have no live
  connector.
- The API-backed pipeline scans and redacts secrets, masks emails, phone numbers and IP addresses, records
  licence metadata, excludes copyleft code from FIM samples, deduplicates code, chunks content, and registers
  a dataset. Review `manifest.json` and `scan_report.json` under
  `var/datasets/collection-agent/<dataset-version>/` before training or retrieval.

Connector environment variables:

| Source | Variables |
|---|---|
| GitHub | `HELIX_GITHUB_TOKEN` |
| ServiceNow | `HELIX_SERVICENOW_INSTANCE`, `HELIX_SERVICENOW_USER`, `HELIX_SERVICENOW_PASSWORD` |
| Confluence | `HELIX_CONFLUENCE_SITE`, `HELIX_CONFLUENCE_EMAIL`, `HELIX_CONFLUENCE_API_TOKEN`, `HELIX_CONFLUENCE_SPACE_ID` |
| Loki | `HELIX_LOKI_URL`, `HELIX_LOKI_QUERY` |

The collection endpoints are development-only. Use least-privilege source credentials and keep secrets in the
deployment's secret manager or server environment.

### Standalone interactive learning portal

Open `HELIX_Engineering_IT_Copilot.html` directly in a modern browser to use the learning portal without starting
the API server. The page is self-contained, works offline, and uses simulated data. It includes the 17-step
LEXORA/NEXUS journey, interactive browser-only labs, calculations, interview drills, and a searchable technology
catalogue. The API-backed application remains available at `/portal/` when running the server.

```bash
make demo               # scripted end-to-end walkthrough of every capability (no server needed)
make test               # 37 tests: prep, metrics, guardrails, router, RAG, tools, all five agents via HTTP
```

Without `make`: `pip install -r requirements.txt && python scripts/generate_sample_data.py && python scripts/bootstrap.py && uvicorn nexus.api.main:app --reload`.

## Dev personas

| User | Roles | Use it to |
|---|---|---|
| `u.priya` | employee | IT helpdesk: MFA reset, VPN, software, access requests |
| `u.arjun` | engineer | Code chat, completion, PR review, text-to-query |
| `sre.anita` | SRE | Fire incidents and approve runbooks |
| `u.kavya` | payments engineer | Read the **restricted** `payments/card-vault` repo (self-hosted tiers only) |
| `agent.sam` | service desk | Ticket triage |
| `venkat.k` | manager (Priya's) | Approve access requests |

## Repository map (follows spec 12.1)

```
helix-engit-copilot/
├── common/                 config, logging, tracing, SQLite store, embeddings, text utils
├── lexora/
│   ├── ingest/             GitHub, ServiceNow, Confluence, Loki connectors + local sample loader
│   ├── prep/               secrets, PII, licence, MinHash LSH dedupe, tree-sitter/ast chunking, Drain, FIM, pipeline
│   ├── train/              L1 triage, L2 embedder, L3 reranker, L4 FIM LoRA, L5 QLoRA SFT+DPO, L6, L7, L8, L9, calibration
│   ├── eval/               metrics (pass@k, Recall/MRR/nDCG, ECE, AUC), gates, retrieval + text2query + pass@k harnesses
│   └── registry/           model cards (the LEXORA -> NEXUS contract), optional MLflow logging
├── nexus/
│   ├── gateway/            OIDC/dev JWT auth, RBAC principal, token-bucket rate limits
│   ├── guardrails/         DLP + secret scan, prompt-injection scoring, Semgrep-style validators
│   ├── router/             OPA policy gate (+ Python mirror), tier scoring and fallbacks
│   ├── llm/                OpenAI-compatible providers per tier (vLLM/Ollama/OpenAI) + mock provider
│   ├── rag/                BM25, vector store (in-memory/Qdrant), code graph, RRF, reranker, semantic cache, grounded answers
│   ├── completion/         FIM context assembly, post-filters
│   ├── query/              text-to-query: schema RAG, sqlglot validation, read-only execution
│   ├── ml/                 runtime for L1, L8, L9, anomaly scoring
│   ├── tools/              MCP tool catalogue: itsm, idp, mdm, git, obs, deploy, runbook, kb, cmdb, db (+ MCP server)
│   ├── workflows/          LangGraph agents: helpdesk, triage, pr_review, incident, change_risk
│   ├── observability/      Prometheus metrics
│   └── api/                FastAPI /v1 endpoints (spec 11)
├── clients/
│   ├── web-portal/         static portal served at /portal
│   ├── vscode-extension/   TypeScript inline-completion provider (debounce + cancel) and Ask command
│   └── chat-bot/           Slack Bolt bot with step-up and approval buttons
├── configs/                helix.yaml (tiers, router, gates, roles, repos), OPA policy, training recipes, Drain3, metric catalogue
├── data/sample/            sample repos (incl. a deliberately dirty file), docs, KB, postmortems, PR diff
├── scripts/                generate_sample_data.py, bootstrap.py, demo.py
├── infra/                  Dockerfile, docker-compose (OPA, Ollama, Qdrant, OpenSearch, Neo4j, Postgres, Redis, Prometheus, Grafana, Loki), Helm, Terraform
├── docs/                   ARCHITECTURE.md, BUILD_GUIDE.md, API.md, adr/
└── tests/
```

## The five agents

| Agent | Trigger | Graph (LangGraph) | Human gate |
|---|---|---|---|
| Helpdesk (C1, C10) | chat message | classify → step-up / install / request access / escalate → close | user approves Okta push; manager approves access |
| Ticket triage (C2) | new ticket | classify (L1) → duplicates (L8) → KB → decide → act | none (auto-route only at p ≥ 0.85) |
| PR review (C6, C11) | PR webhook | fetch → context → review (L5, JSON schema) → validate → post COMMENT | humans merge; bot never approves |
| Incident copilot (C8) | PagerDuty alert | metrics → logs/Drain → traces → changes → correlate → similar → brief → gate → execute → verify → postmortem | SRE approves the runbook (token bound to params) |
| Change risk (C9) | change request | enrich → L9 score + TreeSHAP → CAB brief → route | CAB |

## Going from MVP to production

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md). In short: set the `HELIX_T*_URL` endpoints (vLLM pools), train
L2–L7 on a GPU with `requirements-train.txt`, set `HELIX_OIDC_ISSUER`, start OPA, swap the mock tools in
`nexus/tools/catalog.py` for the real ServiceNow/Okta/Intune/Rundeck APIs, and move SQLite to PostgreSQL.

All data in this repository is synthetic. The AWS key in `data/sample` is AWS's public documentation example.
