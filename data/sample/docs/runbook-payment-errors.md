# Runbook: payment errors

Service: checkout-service · Owner: Payments SRE · On-call: payments-primary

## E1043 payment token decode failed
Meaning: the checkout service could not decode the token returned by the card vault.

Checks:
1. Compare the card-vault tokeniser version with the token format expected by checkout (`tok_v2_`).
2. Check whether errors started after a checkout or vault deployment (Argo CD history).
3. Query `sum(count_over_time({app="checkout"} |= "E1043" [5m]))` in Loki.

Remediation:
- If errors began after a checkout deployment, run runbook `rollback-deployment` for checkout-service to the previous version.
- If the vault was upgraded, pin checkout to the vault compatibility flag `token_format=v1` and page Payments Platform.

## E2001 gateway timeout
Meaning: the upstream payment gateway did not respond within 5 seconds.
Remediation: check gateway status page; enable the retry-with-backoff feature flag `gateway_retry`.
