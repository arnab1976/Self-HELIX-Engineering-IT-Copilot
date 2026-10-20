# PM-0301: Checkout pods OOMKilled after cache change

Severity: SEV-3 · Duration: 21 minutes · Service: checkout-service

## Summary
A change to the in-process cart cache removed the size limit; pods were OOMKilled under load.

## Resolution
Rolled back checkout 2.40 to 2.39 and restored the cache limit.
