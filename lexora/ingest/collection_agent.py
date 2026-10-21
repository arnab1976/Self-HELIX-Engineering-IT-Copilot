"""LEXORA collection agent: fetch engineering sources, sanitise them, and register a dataset."""
from __future__ import annotations

import csv
import logging
import re
import time
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from common.config import settings
from lexora.ingest.base import Document
from lexora.ingest.confluence_connector import ConfluenceConnector
from lexora.ingest.git_connector import GitHubConnector
from lexora.ingest.itsm_connector import ServiceNowConnector
from lexora.ingest.local import load_markdown, load_repos
from lexora.ingest.logs_connector import loki_query_range, read_log_file
from lexora.prep import pipeline

log = logging.getLogger("lexora.collection_agent")

SOURCE_DETAILS = {
    "code": {
        "label": "Git repositories",
        "types": "Source code, configuration, infrastructure-as-code, Markdown and pull-request diffs/reviews",
        "simulation": "data/sample/repos",
        "connector": "GitHub REST API",
        "environment": ["HELIX_GITHUB_TOKEN"],
    },
    "tickets": {
        "label": "IT service management",
        "types": "Incident title and description, category, priority, assignment group, configuration item and timestamps",
        "simulation": "data/generated/tickets.csv",
        "connector": "ServiceNow Table API",
        "environment": ["HELIX_SERVICENOW_INSTANCE", "HELIX_SERVICENOW_USER", "HELIX_SERVICENOW_PASSWORD"],
    },
    "knowledge": {
        "label": "Knowledge and runbooks",
        "types": "Knowledge-base and Confluence pages, engineering documentation, runbooks and postmortems",
        "simulation": "data/sample/docs, kb and postmortems",
        "connector": "Confluence Cloud REST API",
        "environment": ["HELIX_CONFLUENCE_SITE", "HELIX_CONFLUENCE_EMAIL", "HELIX_CONFLUENCE_API_TOKEN",
                        "HELIX_CONFLUENCE_SPACE_ID"],
    },
    "logs": {
        "label": "Application logs",
        "types": "Timestamped log lines, severity, service, event codes and operational context",
        "simulation": "data/generated/logs/checkout.log",
        "connector": "Grafana Loki query_range API",
        "environment": ["HELIX_LOKI_URL", "HELIX_LOKI_QUERY"],
    },
    "observability": {
        "label": "Metrics and traces",
        "types": "Service-level indicators, latency/error time series and distributed trace spans",
        "simulation": "data/generated/metrics and traces.json",
        "connector": None,
        "environment": [],
    },
}

AGENT_ROLES = [
    {"name": "Source connector", "description": "Fetch only the selected repository, ticket, knowledge, or log data using server-managed, least-privilege access."},
    {"name": "Privacy & secret steward", "description": "Detect and redact credentials and personal data before any dataset is registered."},
    {"name": "Licence & quality reviewer", "description": "Record licence metadata, exclude copyleft code from FIM training, and remove near-duplicate code."},
    {"name": "Dataset registrar", "description": "Chunk safe content, write a versioned manifest and scan report, and publish the new dataset version."},
]


def _setting_value(name: str) -> str:
    return getattr(settings(), name.removeprefix("HELIX_").lower())


def _normalize_servicenow_instance(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = value.split("/")[0]
    if value.lower().endswith(".service-now.com"):
        value = value[: -len(".service-now.com")]
    return value.strip()


def _normalize_confluence_site(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = value.split("/")[0]
    if value.lower().endswith(".atlassian.net"):
        value = value[: -len(".atlassian.net")]
    return value.strip()


def _credential_value(credentials: dict | None, key: str, env_name: str = "") -> str:
    credentials = credentials or {}
    raw = str(credentials.get(key) or "").strip() or _setting_value(env_name or f"HELIX_{key.upper()}")
    if key == "servicenow_instance":
        return _normalize_servicenow_instance(raw)
    if key == "confluence_site":
        return _normalize_confluence_site(raw)
    return raw


def _configured_sources() -> dict[str, bool]:
    return {
        "code": bool(settings().github_token),
        "tickets": all(_setting_value(name) for name in SOURCE_DETAILS["tickets"]["environment"]),
        "knowledge": all(_setting_value(name) for name in SOURCE_DETAILS["knowledge"]["environment"]),
        "logs": all(_setting_value(name) for name in SOURCE_DETAILS["logs"]["environment"]),
        "observability": False,
    }


def _safe_base_url(value: str, fallback: str) -> str:
    if not value:
        return fallback
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return fallback
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
    except ValueError:
        return fallback
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def _connection_url(source: str) -> str | None:
    config = settings()
    if source == "code":
        return "https://api.github.com/repos/{owner}/{repository}/git/trees/{ref}"
    if source == "tickets":
        instance = _safe_base_url(config.servicenow_instance, "https://<your-instance>.service-now.com")
        return f"{instance}/api/now/table/incident"
    if source == "knowledge":
        site = _safe_base_url(config.confluence_site, "https://<your-site>.atlassian.net")
        space_id = config.confluence_space_id or "{spaceId}"
        return f"{site}/wiki/api/v2/spaces/{space_id}/pages"
    if source == "logs":
        endpoint = _safe_base_url(config.loki_url, "https://<your-loki-host>")
        return f"{endpoint}/loki/api/v1/query_range"
    return None


def source_catalog() -> dict:
    configured = _configured_sources()
    sources = []
    for key, details in SOURCE_DETAILS.items():
        sources.append({
            "id": key,
            **details,
            "configured": configured[key],
            "mode": "simulation_only" if details["connector"] is None else "simulation_and_connected",
            "api_url": _connection_url(key),
        })
    return {
        "roles": AGENT_ROLES,
        "sources": sources,
        "credentials_location": (
            "Enter connector credentials in the source-card dialog (dev learning portal) "
            "or set the listed HELIX_* variables on the API server."
        ),
    }


def _simulation_documents(sources: list[str]) -> list[Document]:
    """Load a rich local corpus per category so Step 02 can emit ≥5 chunks for Adapt."""
    data_root = Path(settings().data_dir)
    docs: list[Document] = []
    if "code" in sources:
        docs.extend(load_repos()[:120])
    if "knowledge" in sources:
        docs.extend(load_markdown("docs", "confluence", "docs/engineering-handbook"))
        docs.extend(load_markdown("kb", "kb", "docs/engineering-handbook"))
        docs.extend(load_markdown("postmortems", "postmortem", "docs/engineering-handbook"))
    if "tickets" in sources:
        ticket_path = data_root / "generated" / "tickets.csv"
        with ticket_path.open(newline="", encoding="utf-8") as handle:
            for row in islice(csv.DictReader(handle), 240):
                text = "\n".join(f"{key}: {value}" for key, value in row.items() if value)
                docs.append(Document("itsm", f"{row.get('id', 'ticket')}.json", text, "ticket", "servicenow/incidents",
                                     {"category": row.get("category"), "priority": row.get("priority"),
                                      "assignment_group": row.get("assignment_group"), "title": row.get("title"),
                                      "description": row.get("description")}))
    if "logs" in sources:
        lines = read_log_file()[:2500]
        if lines:
            # Split into several log shards so the handoff carries multiple records
            shard = 200
            for i in range(0, len(lines), shard):
                part = lines[i : i + shard]
                docs.append(Document(
                    "logs", f"checkout-part-{i // shard + 1:02d}.log", "\n".join(part),
                    "log", "loki/checkout", {"shard": i // shard + 1},
                ))
    if "observability" in sources:
        metrics = data_root / "generated" / "metrics" / "checkout.json"
        traces = data_root / "generated" / "traces.json"
        deploys = data_root / "generated" / "deploys.json"
        for source_path, source_name, kind in (
            (metrics, "metrics", "metric"),
            (traces, "traces", "trace"),
            (deploys, "deploys", "deploy"),
        ):
            if source_path.exists():
                docs.append(Document("observability", source_path.name, source_path.read_text(encoding="utf-8"),
                                     "doc", f"observability/{source_name}", {"record_type": kind}))
        # Synthetic SLO / region windows so observability always yields ≥5 Step 02 chunks
        for hour in range(8, 20):
            docs.append(Document(
                "observability", f"slo-window-{hour:02d}.json",
                (
                    f'{{"service":"checkout-api","hour":{hour},"latency_p95_ms":{180 + hour * 12},'
                    f'"error_rate":{round(0.01 + hour * 0.002, 4)},"availability":{round(99.9 - hour * 0.01, 3)},'
                    f'"trace_sample":"tr-{hour:02d}91","region":"ap-south-1",'
                    f'"notes":"SLO burn window {hour}:00 — correlate with deploy and card-vault decode errors."}}'
                ),
                "doc", "observability/slo", {"record_type": "slo", "hour": hour},
            ))
        for region in ("ap-south-1", "eu-west-1", "us-east-1"):
            docs.append(Document(
                "observability", f"region-health-{region}.json",
                (
                    f'{{"service":"checkout-api","region":"{region}","healthy_hosts":12,'
                    f'"unhealthy_hosts":1,"p99_ms":410,"error_budget_remaining_pct":64.2}}'
                ),
                "doc", "observability/region", {"record_type": "region", "region": region},
            ))
    return docs


def _build_handoff_text(sanitized_docs: list[Document], *, max_docs: int = 48, max_chars: int = 120_000) -> tuple[str, int]:
    """Concatenate many sanitised documents so Step 02 receives a multi-record corpus, not one preview."""
    parts: list[str] = []
    used = 0
    count = 0
    for doc in sanitized_docs:
        body = (doc.text or "").strip()
        if not body:
            continue
        block = f"===== {doc.path} =====\n{body}\n"
        if parts and used + len(block) > max_chars:
            break
        if not parts and len(block) > max_chars:
            block = block[:max_chars]
        parts.append(block)
        used += len(block)
        count += 1
        if count >= max_docs:
            break
    return "\n".join(parts), count


def _docs_from_portal_tickets(portal_tickets: list[dict] | None) -> list[Document]:
    """Convert Agent 01 embedded-portal tickets into labelled ITSM documents for Collect."""
    docs: list[Document] = []
    for i, raw in enumerate(portal_tickets or []):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        description = str(raw.get("description") or "").strip()
        category = str(raw.get("category") or "").strip()
        assignment_group = str(raw.get("assignment_group") or "").strip()
        priority = str(raw.get("priority") or "P3").strip() or "P3"
        if not (title and category and assignment_group):
            continue
        ticket_id = str(raw.get("id") or f"PORTAL{i+1:04d}").strip() or f"PORTAL{i+1:04d}"
        requester = str(raw.get("requester") or "portal.user").strip() or "portal.user"
        ci = str(raw.get("ci") or "PORTAL-APP").strip() or "PORTAL-APP"
        split = str(raw.get("split") or "train").strip() or "train"
        created_at = str(raw.get("created_at") or "").strip()
        text = "\n".join(
            [
                f"id: {ticket_id}",
                f"title: {title}",
                f"description: {description}",
                f"requester: {requester}",
                f"ci: {ci}",
                f"category: {category}",
                f"assignment_group: {assignment_group}",
                f"priority: {priority}",
                f"created_at: {created_at}",
                f"split: {split}",
                "source_channel: employee_self_service_portal",
            ]
        )
        docs.append(
            Document(
                "itsm",
                f"{ticket_id}.json",
                text,
                "ticket",
                "portal/self-service",
                {
                    "category": category,
                    "priority": priority,
                    "assignment_group": assignment_group,
                    "title": title,
                    "description": description,
                    "portal": True,
                },
            )
        )
    return docs


def _require_source_credentials(source: str, credentials: dict | None) -> None:
    credentials = credentials or {}
    required = {
        "code": [("github_token", "HELIX_GITHUB_TOKEN")],
        "tickets": [
            ("servicenow_instance", "HELIX_SERVICENOW_INSTANCE"),
            ("servicenow_user", "HELIX_SERVICENOW_USER"),
            ("servicenow_password", "HELIX_SERVICENOW_PASSWORD"),
        ],
        "knowledge": [
            ("confluence_site", "HELIX_CONFLUENCE_SITE"),
            ("confluence_email", "HELIX_CONFLUENCE_EMAIL"),
            ("confluence_api_token", "HELIX_CONFLUENCE_API_TOKEN"),
            ("confluence_space_id", "HELIX_CONFLUENCE_SPACE_ID"),
        ],
        "logs": [
            ("loki_url", "HELIX_LOKI_URL"),
            ("loki_query", "HELIX_LOKI_QUERY"),
        ],
    }.get(source, [])
    missing = [env for key, env in required if not _credential_value(credentials, key, env)]
    if missing:
        raise ValueError(
            f"{SOURCE_DETAILS[source]['label']} connector is not configured. "
            f"Enter credentials in the source dialog or set: {', '.join(missing)}"
        )


def _connected_documents(sources: list[str], github_repo: str, credentials: dict | None = None) -> list[Document]:
    docs: list[Document] = []
    credentials = credentials or {}
    for source in sources:
        if source == "observability":
            raise ValueError("Metrics and traces are available in simulation mode only; no live connector is configured.")
        _require_source_credentials(source, credentials)
        try:
            if source == "code":
                repo = (github_repo or credentials.get("github_repo") or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
                    raise ValueError("Enter a GitHub repository as owner/repository.")
                ref = (credentials.get("github_ref") or "HEAD").strip() or "HEAD"
                connector = GitHubConnector(token=_credential_value(credentials, "github_token", "HELIX_GITHUB_TOKEN"))
                try:
                    docs.extend(connector.files(repo, ref=ref, limit=300))
                    for pr in connector.pull_requests(repo, limit=10):
                        comments = "\n".join(f"{item['path']}: {item['body']}" for item in pr["review_comments"])
                        text = "\n".join(part for part in (
                            f"# {pr['title']}", pr["body"], pr["diff"], f"Review comments:\n{comments}" if comments else ""
                        ) if part)
                        if text:
                            docs.append(Document("git", f"pull-requests/PR-{pr['number']}.md", text, "doc", repo,
                                                 {"pull_request": pr["number"], "merged": pr["merged"]}))
                finally:
                    connector.http.close()
            elif source == "tickets":
                connector = ServiceNowConnector(
                    _credential_value(credentials, "servicenow_instance", "HELIX_SERVICENOW_INSTANCE"),
                    _credential_value(credentials, "servicenow_user", "HELIX_SERVICENOW_USER"),
                    _credential_value(credentials, "servicenow_password", "HELIX_SERVICENOW_PASSWORD"),
                )
                try:
                    incidents = connector.incidents(query="active=true", limit=300, page=100)
                finally:
                    connector.http.close()
                docs.extend(
                    Document("itsm", f"{row.get('id', 'incident')}.json",
                             "\n".join(f"{key}: {value}" for key, value in row.items() if value),
                             "ticket", "servicenow/incidents", {"priority": row.get("priority")})
                    for row in incidents
                )
            elif source == "knowledge":
                space_id = _credential_value(credentials, "confluence_space_id", "HELIX_CONFLUENCE_SPACE_ID")
                connector = ConfluenceConnector(
                    _credential_value(credentials, "confluence_site", "HELIX_CONFLUENCE_SITE"),
                    _credential_value(credentials, "confluence_email", "HELIX_CONFLUENCE_EMAIL"),
                    _credential_value(credentials, "confluence_api_token", "HELIX_CONFLUENCE_API_TOKEN"),
                )
                try:
                    pages = connector.space_pages(space_id, limit=100)
                finally:
                    connector.http.close()
                for doc in pages:
                    doc.repo = f"confluence/{space_id}"
                docs.extend(pages)
            elif source == "logs":
                end_ns = time.time_ns()
                start_ns = end_ns - 24 * 60 * 60 * 1_000_000_000
                loki_url = _credential_value(credentials, "loki_url", "HELIX_LOKI_URL").rstrip("/")
                loki_query = _credential_value(credentials, "loki_query", "HELIX_LOKI_QUERY")
                lines = loki_query_range(
                    loki_url,
                    loki_query,
                    start_ns,
                    end_ns,
                    limit=1000,
                )
                if lines:
                    docs.append(Document("logs", "loki-query.log", "\n".join(lines), "log", "loki/connected"))
        except httpx.HTTPError as exc:
            log.exception("Collection source request failed: %s", source)
            raise RuntimeError(
                f"{SOURCE_DETAILS[source]['label']} fetch failed ({type(exc).__name__}). "
                "Check the credentials, endpoint, and source access."
            ) from exc
    return docs


def collect_and_register(
    *,
    mode: str,
    sources: list[str],
    sample_text: str,
    filename: str,
    github_repo: str,
    dedupe_threshold: float,
    credentials: dict | None = None,
    portal_tickets: list[dict] | None = None,
    portal_only: bool = False,
    portal_track: str = "",
) -> dict:
    valid_sources = set(SOURCE_DETAILS)
    unknown = sorted(set(sources) - valid_sources)
    if unknown:
        raise ValueError(f"Unknown data source(s): {', '.join(unknown)}.")
    if not sources:
        raise ValueError("Select at least one data source.")

    portal_docs = _docs_from_portal_tickets(portal_tickets)
    track = (portal_track or "").strip().lower()
    if track not in {"", "fine-tune", "retrieve"}:
        raise ValueError("portal_track must be fine-tune, retrieve, or empty.")

    if portal_only:
        # Manual Employee Self-Service tickets only — do not pull simulated CSV / live ITSM.
        if not portal_docs:
            raise ValueError(
                "No manual portal tickets to collect. Create at least one ticket in the "
                "Employee Self-Service portal, then continue."
            )
        docs = list(portal_docs)
        sources = ["tickets"]
        sample_text = ""
    else:
        if mode == "simulation":
            docs = _simulation_documents(sources)
        elif mode == "connected":
            docs = _connected_documents(sources, github_repo, credentials)
        else:
            raise ValueError("Mode must be simulation or connected.")
        if mode == "connected" and not docs:
            raise ValueError("The selected connected sources returned no records. Check the repository, query window, and source permissions.")
        # Employee self-service portal tickets — merged ahead of simulated/connected ITSM rows
        if portal_docs:
            docs = portal_docs + docs

    if sample_text.strip():
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", filename):
            raise ValueError("Use a simple file name containing letters, numbers, dots, underscores, or hyphens.")
        kind = "doc" if filename.lower().endswith((".md", ".txt")) else "code"
        docs.insert(0, Document("user_sample", filename, sample_text, kind, "user/submitted-sample"))
    if not docs:
        raise ValueError("The selected sources returned no records. Check the data files, source permissions, or query window.")
    if len(docs) > 500:
        docs = docs[:500]

    sanitized_docs = []
    for doc in docs:
        clean, _ = pipeline.sanitise(doc)
        if clean:
            sanitized_docs.append(clean)
    if not sanitized_docs:
        raise ValueError("No records passed sanitisation. Review the source data and secret findings; no dataset was registered.")
    # Prefer source-category docs over the optional form sample for the multi-doc handoff corpus.
    preferred = [d for d in sanitized_docs if d.source != "user_sample"] or sanitized_docs
    preview_doc = preferred[0]
    primary = "tickets" if portal_only else sources[0]
    # Tickets: carry the full labelled ITSM corpus so Agent 05 L1 trains on the same
    # Agent 01 handoff (not a later silent tickets.csv reload). Other sources stay compact.
    handoff_max_docs = max(len(preferred), 240 if primary == "tickets" else 48)
    if portal_only:
        handoff_max_docs = max(len(preferred), len(portal_docs))
    handoff_text, handoff_docs = _build_handoff_text(preferred, max_docs=handoff_max_docs)
    # Keep a short UI preview; full multi-doc corpus goes in handoff.text for Step 02.
    safe_preview = handoff_text[:12000]

    output_root = Path(settings().state_dir) / "datasets" / "collection-agent"
    manifest = pipeline.run(docs=docs, out_root=output_root, dedupe_threshold=dedupe_threshold)
    source_meta = SOURCE_DETAILS.get(primary, {})
    path_by_source = {
        "code": "corpus/code-bundle.txt",
        "tickets": "corpus/tickets-bundle.txt",
        "knowledge": "corpus/knowledge-bundle.txt",
        "logs": "corpus/logs-bundle.txt",
        "observability": "corpus/observability-bundle.txt",
    }
    kind_by_source = {
        "code": "code",
        "tickets": "ticket",
        "knowledge": "doc",
        "logs": "log",
        "observability": "doc",
    }
    return {
        "agent": "LEXORA Collect & Sanitise",
        "mode": mode,
        "sources_selected": sources,
        "documents_in": manifest["documents_in"],
        "documents_out": manifest["documents_out"],
        "chunks": manifest["chunks"],
        "fim_samples": manifest["fim_samples"],
        "secrets_redacted": manifest["secrets_redacted"],
        "pii_masked": manifest["pii_masked"],
        "near_duplicates_dropped": manifest["near_duplicates_dropped"],
        "copyleft_excluded_from_fim": manifest["copyleft_excluded_from_fim"],
        "dataset_version": manifest["version"],
        "source_mix": manifest["source_mix"],
        "seconds": manifest["seconds"],
        "portal_tickets_in": len(portal_docs),
        "portal_only": bool(portal_only),
        "portal_track": track or "",
        "preview": safe_preview,
        "primary_source": primary,
        "filename": path_by_source.get(primary, preview_doc.path if preview_doc else "handoff.txt"),
        "handoff": {
            "primary_source": primary,
            "source_label": (
                "Manual portal tickets"
                if portal_only
                else source_meta.get("label", primary)
            ),
            "path": path_by_source.get(primary, preview_doc.path if preview_doc else "handoff.txt"),
            "kind": kind_by_source.get(primary, preview_doc.kind if preview_doc else "doc"),
            "text": handoff_text,
            "dataset_version": manifest["version"],
            "mode": "portal_manual" if portal_only else mode,
            "documents_out": manifest["documents_out"],
            "handoff_documents": handoff_docs,
            "handoff_chars": len(handoff_text),
            "portal_only": bool(portal_only),
            "portal_track": track or "",
            "portal_tickets_in": len(portal_docs),
        },
        "next": (
            (
                f"Manual portal corpus ready ({handoff_docs} ticket(s)). "
                f"Continue to Agent 02 · Chunk, then Adapt track '{track or 'retrieve'}'."
            )
            if portal_only
            else (
                f"Review the scan report. Step 02 receives a multi-document handoff "
                f"({handoff_docs} records, {len(handoff_text)} chars) — enough material for ≥5 chunks."
            )
        ),
    }
