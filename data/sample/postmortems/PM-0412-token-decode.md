# PM-0412: Checkout E1043 after vault upgrade

Severity: SEV-2 · Duration: 34 minutes · Service: checkout-service

## Summary
Card vault 5.0 switched tokens to format v2. Checkout could not decode the new tokens and returned E1043; checkout error rate reached 6.1%.

## Resolution
Rolled back the vault to 4.9 and released checkout 3.12 with v2 decoding.

## Lessons
Add a contract test between checkout and card-vault for the token format. Alert on new E1043 log templates.
