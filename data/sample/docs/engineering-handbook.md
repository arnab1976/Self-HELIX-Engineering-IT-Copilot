# Engineering handbook

## Code review
Every pull request needs one approving review from a code owner. The HELIX review bot comments but never approves or merges.
Security findings marked high must be resolved before merge. Use parameterised queries; never build SQL with f-strings.

## Logging
Use the structured logger (`logger.info(msg, extra={...})`). Never log card numbers, tokens or full request bodies.
Do not use `print` in service code.

## HTTP calls
Every outbound HTTP call must set a timeout (default 5 seconds) and keep TLS verification on.

## Secrets
Secrets live in HashiCorp Vault. Never commit keys. Load them at runtime with `vault.read(path)`.

## Deployments
Production deploys go through Argo CD. Friday-evening deploys need change-advisory approval.
