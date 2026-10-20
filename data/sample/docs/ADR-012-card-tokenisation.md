# ADR-012: Card tokenisation

Status: Accepted · Owner: Payments Platform · Repo: payments/checkout-service

## Context
Card numbers (PANs) must never be stored by the checkout service. PCI scope must stay limited to the card vault.

## Decision
Cards are validated in the checkout service before tokenisation: presence, Luhn checksum and expiry.
Validation is implemented in `CardValidator.validate_card` and failures return HTTP 422.
After validation the checkout service calls the card vault's `tokenise_card`, which returns an opaque token.
Tokens use format v2 (`tok_v2_` prefix) from vault release 5.0 onwards.

## Consequences
- The checkout service and the card vault must agree on the token format. A contract test is required.
- Any change to token decoding in checkout must be released together with the vault.
