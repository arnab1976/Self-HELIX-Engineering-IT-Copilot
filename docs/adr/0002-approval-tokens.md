# ADR-0002: Approval tokens bound to exact parameters

Status: Accepted

## Context
Agents propose production actions (rollbacks, access grants). An approver must see exactly what will run, and an
agent (or an injected instruction) must not be able to change the action after approval.

## Decision
`token = exp~approver~action_id~HMAC_SHA256(secret, canonical{action_id, tool, params, approver, exp})`.
Write tools that need approval verify the token against the parameters of the actual call. Dry runs require their
own token. Tokens expire after `approvals.token_ttl_minutes` (default 10). Idempotency keys prevent double execution.

## Consequences
- Modifying a parameter (e.g. rollback target) requires a new approval; the API's `modify` decision mints it.
- Every execution is auditable: approver, token, parameters and result are stored in `actions` and `audit`.
