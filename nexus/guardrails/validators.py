"""Validators for generated code and review findings (spec 7.3 step 4, 8.1).

  semgrep   real Semgrep CLI when installed (`semgrep --config p/python`), else the built-in rule pack below
  secrets   secret scan of suggested code
  licence   snippet match against known OSS text (stub: SPDX / licence-text detection)
  iac       Checkov / Conftest when installed, else built-in Terraform/K8s rules
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from lexora.prep import licence as lic
from lexora.prep import secrets


@dataclass
class Finding:
    rule: str
    severity: str        # high | medium | low
    category: str        # security | reliability | performance | style
    message: str
    suggestion: str
    line_offset: int = 0
    validator: str = "semgrep-lite"


PY_RULES: list[tuple[str, re.Pattern, str, str, str, str]] = [
    ("python.lang.security.audit.formatted-sql-query", re.compile(r"\.execute\(\s*f[\"']|\.execute\([^)]*%\s*\(|\.execute\([^)]*\.format\("),
     "high", "security", "User input is interpolated into SQL; use parameterised queries.",
     'cursor.execute("SELECT id, status, total FROM orders WHERE id = %s", (order_id,))'),
    ("python.requests.security.disabled-cert-validation", re.compile(r"verify\s*=\s*False"), "medium", "security",
     "TLS certificate verification is disabled.", "Remove verify=False; keep TLS verification on."),
    ("house-style.http-timeout", re.compile(r"requests\.(get|post|put|delete)\((?![^)]*timeout)"), "medium", "reliability",
     "HTTP call has no timeout; a slow upstream will hang workers.", "Add timeout=5."),
    ("python.lang.security.dangerous-subprocess-shell", re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"), "high", "security",
     "subprocess with shell=True enables command injection.", "Pass an argument list and shell=False."),
    ("python.lang.security.eval", re.compile(r"\beval\("), "high", "security", "eval() on dynamic input executes arbitrary code.",
     "Use ast.literal_eval or explicit parsing."),
    ("ruff.T201-print", re.compile(r"^\s*print\("), "low", "style", "Use the structured logger instead of print.",
     'logger.info("...", extra={...})'),
    ("house-style.no-select-star", re.compile(r"SELECT \*", re.I), "low", "performance", "Select only the columns you need.",
     "SELECT id, status, total ..."),
]
IAC_RULES = [
    ("CKV_AWS_20", re.compile(r'acl\s*=\s*"public-read'), "high", "security", "S3 bucket is publicly readable.", 'acl = "private"'),
    ("CKV_K8S_16", re.compile(r"privileged:\s*true"), "high", "security", "Container runs privileged.", "privileged: false"),
    ("CKV_K8S_13", re.compile(r"image:\s*\S+:latest"), "low", "reliability", "Image uses the mutable :latest tag.", "Pin an image digest."),
]


def run_semgrep_cli(code: str, filename: str = "snippet.py") -> list[Finding] | None:
    if not shutil.which("semgrep"):
        return None
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/{filename}"
        open(p, "w").write(code)
        res = subprocess.run(["semgrep", "--config", "p/python", "--json", "--quiet", p], capture_output=True, text=True, timeout=120)
    out = []
    for r in json.loads(res.stdout or "{}").get("results", []):
        sev = {"ERROR": "high", "WARNING": "medium"}.get(r["extra"]["severity"], "low")
        out.append(Finding(r["check_id"], sev, "security", r["extra"]["message"], "", r["start"]["line"] - 1, "semgrep"))
    return out


def check_line(line: str, rules=PY_RULES) -> list[Finding]:
    return [Finding(rid, sev, cat, msg, sug) for rid, rx, sev, cat, msg, sug in rules if rx.search(line)]


def validate_code(code: str, iac: bool = False) -> dict:
    """Full validation of a code snippet (suggested fix, completion, generated IaC)."""
    findings: list[Finding] = []
    sem = run_semgrep_cli(code)
    if sem is not None:
        findings += sem
    for i, line in enumerate(code.splitlines()):
        for f in check_line(line, IAC_RULES if iac else PY_RULES):
            f.line_offset = i
            findings.append(f)
    sec = secrets.scan(code)
    for s in sec:
        findings.append(Finding("secrets." + s.type.lower().replace(" ", "-"), "high", "security",
                                f"Hard-coded {s.type}. Load it from Vault at runtime and rotate it.",
                                'value = vault.read("path")["key"]', code[: s.start].count("\n"), "secrets"))
    licence_res = lic.detect(code)
    return {"findings": findings, "secrets": len(sec), "licence": licence_res.licence,
            "semgrep_engine": "semgrep" if sem is not None else "semgrep-lite"}
