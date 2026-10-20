# PM-0388: Payment gateway timeouts during sale

Severity: SEV-3 · Duration: 52 minutes · Service: checkout-service

## Summary
Upstream payment gateway latency rose to 6 seconds during a sale; checkout requests timed out with E2001.

## Resolution
Enabled the `gateway_retry` feature flag and scaled checkout pods from 6 to 12.

## Lessons
Set explicit HTTP timeouts; add gateway p99 latency to the checkout dashboard.
