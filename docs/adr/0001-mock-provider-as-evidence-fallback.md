# ADR-0001: Mock provider as an evidence-based fallback

Status: Accepted

## Context
The MVP must run every flow on a laptop with no GPU, and production must degrade safely when a model tier is down.

## Decision
Each service computes a deterministic, grounded answer from its evidence (retrieved chunks, validator findings,
correlated signals) and passes it to `generate(tier, messages, fallback=...)`. With no endpoint configured, or on
an HTTP error, the provider returns the fallback. With an endpoint, the model writes the answer from the same evidence.

## Consequences
- Tests are deterministic and need no network.
- It is always visible which part of an answer came from a model (`mocked: true` in responses).
- Fallbacks are deliberately plain; quality improves once real tiers are configured.
