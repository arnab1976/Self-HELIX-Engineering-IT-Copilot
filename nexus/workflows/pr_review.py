"""PR review agent (C6, C11; spec 7.3).

webhook -> fetch diff -> context (callers via code graph + style guide) -> review (T1b / L5, JSON schema)
        -> validate (Semgrep, linter, secret scan, licence) -> post as COMMENT (never approve/merge)
Accepted / dismissed comments arrive later through /v1/feedback and become DPO pairs for L5.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from nexus.gateway.auth import Principal
from nexus.guardrails.validators import check_line, validate_code
from nexus.llm import generate
from nexus.rag.retriever import retrieve
from nexus.tools import registry
from nexus.workflows.runtime import checkpointer

REVIEW_SCHEMA = {"type": "json_schema", "json_schema": {"name": "review", "schema": {
    "type": "object", "required": ["comments"], "properties": {"comments": {"type": "array", "items": {
        "type": "object", "required": ["path", "line", "severity", "category", "message"],
        "properties": {"path": {"type": "string"}, "line": {"type": "integer"},
                       "severity": {"enum": ["high", "medium", "low"]},
                       "category": {"enum": ["security", "reliability", "performance", "style"]},
                       "message": {"type": "string"}, "suggestion": {"type": "string"}}}}}}}}


def parse_diff(diff: str) -> list[dict]:
    """Added lines with their new-file path and line number."""
    out, path, ln = [], None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)", line)
        if m:
            ln = int(m.group(1))
            continue
        if line.startswith("diff --git"):
            path = line.split(" b/")[-1]
            continue
        if line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append({"path": path, "line": ln, "text": line[1:]})
            ln += 1
        elif not line.startswith("-"):
            ln += 1
    return out


def rule_review(diff: str) -> list[dict]:
    """Seed reviewer: validator rules applied to added lines. Used as the mock L5 output and as an SFT teacher."""
    comments = []
    for a in parse_diff(diff):
        for f in check_line(a["text"]):
            comments.append({"path": a["path"], "line": a["line"], "severity": f.severity, "category": f.category,
                             "message": f.message, "suggestion": f.suggestion, "rule": f.rule})
        sec = validate_code(a["text"])
        for f in sec["findings"]:
            if f.validator == "secrets":
                comments.append({"path": a["path"], "line": a["line"], "severity": "high", "category": "security",
                                 "message": f.message, "suggestion": f.suggestion, "rule": f.rule})
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(comments, key=lambda c: (order[c["severity"]], c["line"]))


class PRState(TypedDict, total=False):
    repo: str
    pr: int
    diff: str
    context: list
    comments: list
    validated: list
    posted: dict
    model: str


def fetch(s: PRState) -> PRState:
    if s.get("diff"):
        return {}
    return {"diff": registry.call("git.diff", {"repo": s["repo"], "pr": s["pr"]})["diff"]}


def context(s: PRState) -> PRState:
    added = parse_diff(s["diff"])
    names = [x for t in re.findall(r"def (\w+)|self\.(\w+)", s["diff"]) for x in t if x]
    q = " ".join(dict.fromkeys(names)) or "checkout service"
    res = retrieve(f"{q} callers style guide", Principal("review-bot", "HELIX review bot", ["engineer"]), repo=s.get("repo"), top_n=5)
    return {"context": [h.citation for h in res.hits] + [f"{len(added)} added lines"]}


def review(s: PRState) -> PRState:
    seed = rule_review(s["diff"])
    msgs = [{"role": "system", "content": "You are the HELIX code reviewer. Return JSON matching the schema. Comment only on added lines."},
            {"role": "user", "content": f"Context: {s.get('context')}\n\nDiff:\n{s['diff']}"}]
    out = generate("T1b", msgs, fallback=json.dumps({"comments": seed}), response_format=REVIEW_SCHEMA)
    try:
        comments = json.loads(out.text)["comments"]
    except (json.JSONDecodeError, KeyError, TypeError):
        comments = seed                       # schema violation -> fall back to validator-backed comments
    added = {(a["path"], a["line"]) for a in parse_diff(s["diff"])}
    comments = [c for c in comments if (c.get("path"), c.get("line")) in added]   # drop comments on lines outside the diff
    return {"comments": comments, "model": "lexora/code-review@1.0.4" if not out.mocked else out.model}


def validate(s: PRState) -> PRState:
    added = {(a["path"], a["line"]): a["text"] for a in parse_diff(s["diff"])}
    out = []
    for c in s["comments"]:
        line = added.get((c["path"], c["line"]), "")
        v = validate_code(line)
        confirmed = any(f.severity == c["severity"] or f.validator == "secrets" for f in v["findings"])
        c = {**c, "validators": {"semgrep": "confirmed" if confirmed and c["category"] == "security" else ("lint" if confirmed else "unconfirmed"),
                                 "secrets": "found" if v["secrets"] else "clean", "engine": v["semgrep_engine"]}}
        if c["severity"] == "high" and c["category"] == "security" and not confirmed:
            c["severity"] = "medium"            # unconfirmed security claims are downgraded, never posted as high
        if c.get("suggestion"):
            sv = validate_code(c["suggestion"])
            c["suggestion_validated"] = not any(f.severity == "high" for f in sv["findings"])
        out.append(c)
    return {"validated": out}


def post(s: PRState) -> PRState:
    res = registry.call("git.comment_pr", {"repo": s.get("repo", "unknown"), "pr": s.get("pr", 0), "comments": s["validated"]}, actor="review-bot")
    return {"posted": res}


@lru_cache
def graph():
    g = StateGraph(PRState)
    for n, f in [("fetch", fetch), ("context", context), ("review", review), ("validate", validate), ("post", post)]:
        g.add_node(n, f)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "context")
    g.add_edge("context", "review")
    g.add_edge("review", "validate")
    g.add_edge("validate", "post")
    g.add_edge("post", END)
    return g.compile(checkpointer=checkpointer())
