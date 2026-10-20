# Build guide: from laptop MVP to production

Each step swaps one MVP component for its production equivalent **by configuration or by replacing one
function body**. Interfaces stay the same, so the tests keep passing.

## 1. Real models for the tiers (Ollama, then vLLM)

```bash
ollama pull qwen2.5-coder:1.5b-base       # T1a FIM
ollama pull qwen2.5-coder:7b-instruct     # T1b code chat / review
ollama pull llama3.1:8b                   # T1c / T2
export HELIX_T1A_URL=http://localhost:11434/v1 HELIX_T1A_MODEL=qwen2.5-coder:1.5b-base
export HELIX_T1B_URL=http://localhost:11434/v1 HELIX_T1B_MODEL=qwen2.5-coder:7b-instruct
export HELIX_T2_URL=http://localhost:11434/v1  HELIX_T2_MODEL=llama3.1:8b
make run
```

Production uses vLLM per tier (`vllm serve <model> --enable-lora --lora-modules review=<adapter>` for
multi-LoRA, `--enable-prefix-caching`, speculative decoding for T1a). `nexus/llm/providers.py` talks to any
OpenAI-compatible endpoint; structured output for PR review uses `response_format` (xgrammar in vLLM).
FIM tokens per model family live in `lexora/prep/fim.py`.

## 2. Train the GPU artefacts

```bash
pip install -r requirements-train.txt            # CUDA machine, >= 24 GB for QLoRA on 7-8B
python -m lexora.train.l1_triage --backend transformers
python -m lexora.train.l2_embedder --base BAAI/bge-m3      # then HELIX_EMBEDDER=st:var/artifacts/l2_embedder/final
python -m lexora.train.l3_reranker                         # then HELIX_RERANKER=ce:var/artifacts/l3_reranker
python -m lexora.train.l4_fim                              # or Axolotl: configs/train/l4_fim_axolotl.yaml
python -m lexora.train.l5_review --stage sft
python -m lexora.train.l5_review --stage dpo --adapter var/artifacts/l5_review/sft/adapter   # after collecting feedback
python -m lexora.train.l6_text2query
python -m lexora.train.l7_incident
```

Every script supports `--dry-run` to build and inspect its dataset first. Replace the synthetic sources with
real ones via `lexora/ingest/*` (GitHub PRs and review threads, ServiceNow incidents, Confluence spaces, Loki).

## 3. Identity and policy

- Set `HELIX_OIDC_ISSUER` and `HELIX_OIDC_AUDIENCE`; `nexus/gateway/auth.py` validates RS256 tokens via JWKS and
  reads roles from the `roles` or `groups` claim. Set `HELIX_ENV=prod` to disable `/v1/dev/token`.
- Start OPA (`docker compose --profile ... up opa`) and set `HELIX_OPA_URL`. Policy: `configs/policy/helix.rego`.
  In production make the gate fail closed (see the comment in `nexus/router/policy.py`).

## 4. Real enterprise tools

Replace bodies in `nexus/tools/catalog.py`, keep names, signatures and `ToolSpec` flags:

| Tool | Production API |
|---|---|
| `itsm.*` | ServiceNow Table API / Jira Service Management REST |
| `idp.step_up`, `idp.reset_mfa` | Okta Factors / Lifecycle API, Entra ID Graph |
| `mdm.install_app` | Intune Graph `mobileApps/{id}/assign`, Jamf Pro API |
| `git.*` | GitHub REST (`lexora/ingest/git_connector.py` has a client) |
| `obs.*` | Prometheus `query_range`, Loki `query_range` (+ Drain3), Tempo search |
| `deploy.list_changes` | Argo CD applications history |
| `runbook.execute` | Rundeck job run / Ansible AWX job template (allow-listed only) |
| `db.query_readonly` | read-replica DSN with a read-only role |

Expose them to other agents with `python -m nexus.tools.mcp_server`.

## 5. Stores

| MVP | Production |
|---|---|
| SQLite `common/store.py` | PostgreSQL (same tables) |
| LangGraph `SqliteSaver` | `langgraph-checkpoint-postgres` `PostgresSaver` |
| In-memory vectors | `QdrantVectorStore` in `nexus/rag/vectorstore.py` (`HELIX_QDRANT_URL`) |
| In-process BM25 | OpenSearch index with the same fields |
| `CodeGraph` | Neo4j with SCIP-derived symbol edges |
| In-memory semantic cache | Redis vector search |

## 6. Observability

`/metrics` exposes Prometheus counters for the five planes. Add OpenTelemetry tracing (FastAPI instrumentation)
and Langfuse for LLM generations (`HELIX_LANGFUSE_*`). The portal's Platform tab shows `/v1/dashboard`.

## 7. Release process

1. LEXORA trains a new version and writes a model card with its scorecard.
2. `lexora/eval/gates.py` checks `configs/helix.yaml: release_gates`; only passing cards are promoted.
3. NEXUS reads the production card; the previous version is recorded as `rollback_target`.
4. Shadow → canary → full rollout per tier; feedback flows back via `/v1/feedback`.
