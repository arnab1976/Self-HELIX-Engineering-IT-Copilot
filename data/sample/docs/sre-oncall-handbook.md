# SRE on-call handbook

Owner: sre-oncall@example.test  
Services: checkout-api, card-vault, auth-gateway

## Severity ladder
- Sev-1: customer checkout blocked across regions
- Sev-2: elevated error rate or latency SLO burn
- Sev-3: single-AZ degradation with workaround

## First 15 minutes
1. Confirm alert in Prometheus / PagerDuty.
2. Check last deploy in the correlation window.
3. Capture trace ids for failing spans.
4. Page payments on-call if card-vault token decode errors appear.

## Runbook links
- Payment timeout runbook
- ADR-012 card tokenisation
- Loki query: `{service="checkout"} |= "E1043"`

## Contacts
Primary: sre-oncall@example.test  
Backup: payments-platform@example.test
