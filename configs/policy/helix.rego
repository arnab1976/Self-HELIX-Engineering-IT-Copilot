# OPA policy gate in front of the NEXUS router (spec 6.1, 8.1).
# Load with:  opa run --server --addr :8181 configs/policy/helix.rego configs/policy/data.json
# Query:      POST /v1/data/helix/gateway/decision  {"input": {...}}
# nexus/router/policy.py mirrors this logic in Python when HELIX_OPA_URL is not set.
package helix.gateway

import rego.v1

default allow := false

# ---- inputs ---------------------------------------------------------------
# input.roles            ["engineer", ...]
# input.repo_class       "public" | "internal" | "confidential" | "restricted" | null
# input.target_tier      "T1a" | "T1b" | "T1c" | "T2" | "T3"
# input.tool             optional tool name, e.g. "runbook.execute"
# input.has_approval     bool
# input.injection_score  0..1
# input.secret_findings  int (high-confidence findings left after redaction)

readable_classes contains c if {
	some r in input.roles
	some c in data.roles[r].repo_classes
}

can_read if not input.repo_class
can_read if input.repo_class in readable_classes

external_tier if data.tiers[input.target_tier].external == true

tier_ok if not external_tier
tier_ok if {
	external_tier
	not input.repo_class
}
tier_ok if {
	external_tier
	input.repo_class in {"public", "internal"}
}

write_tools := {"runbook.execute", "idp.reset_mfa", "idp.reset_password", "itsm.request_access", "git.open_pr"}

tool_ok if not input.tool
tool_ok if {
	input.tool
	not input.tool in write_tools
}
tool_ok if {
	input.tool in write_tools
	input.has_approval == true
}

injection_ok if input.injection_score < 0.80

secrets_ok if input.secret_findings == 0

allow if {
	can_read
	tier_ok
	tool_ok
	injection_ok
	secrets_ok
}

reasons contains "role cannot read this repository class" if not can_read
reasons contains "external tier not permitted for this repository class" if not tier_ok
reasons contains "write tool requires a bound approval token" if not tool_ok
reasons contains "prompt-injection score above threshold" if not injection_ok
reasons contains "high-confidence secret present" if not secrets_ok

decision := {"allow": allow, "reasons": reasons}
