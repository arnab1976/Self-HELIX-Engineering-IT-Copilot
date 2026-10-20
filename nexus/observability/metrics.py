"""Prometheus metrics for the five observability planes (spec 9): completion, chat & RAG, workflows, safety, cost."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter("helix_requests_total", "API requests", ["endpoint", "status"])
LATENCY = Histogram("helix_request_latency_seconds", "API latency", ["endpoint"],
                    buckets=(0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10))
COMPLETIONS = Counter("helix_completions_total", "Inline completions", ["outcome"])       # shown | empty | filtered
FEEDBACK = Counter("helix_feedback_total", "Feedback signals", ["target", "signal"])
DEFLECTION = Counter("helix_helpdesk_resolutions_total", "Helpdesk outcomes", ["resolution"])
GUARDRAIL = Counter("helix_guardrail_events_total", "Guardrail events", ["kind"])        # secret_blocked | injection | acl_denied
ROUTED = Counter("helix_routed_total", "Router decisions", ["route", "tier"])
TOKENS_COST = Counter("helix_estimated_cost_usd_total", "Estimated model cost", ["tier"])
