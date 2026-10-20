"""ITSM connector: ServiceNow Table API (production) or the generated CSV (MVP).

ServiceNow: GET /api/now/table/incident?sysparm_query=sys_updated_on>javascript:gs.daysAgoStart(1)
Fields mapped to the L1 training schema: title, description, category, assignment_group, priority, ci.
"""
from __future__ import annotations

import csv
from pathlib import Path

import httpx

from common.config import settings

FIELD_MAP = {"short_description": "title", "description": "description", "category": "category",
             "assignment_group": "assignment_group", "priority": "priority", "cmdb_ci": "ci",
             "caller_id": "requester", "sys_created_on": "created_at", "number": "id"}


def load_tickets_csv(path: Path | None = None) -> list[dict]:
    path = path or Path(settings().data_dir) / "generated" / "tickets.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ServiceNowConnector:
    def __init__(self, instance: str, user: str, password: str, timeout: float = 30.0):
        self.http = httpx.Client(base_url=f"https://{instance}.service-now.com", auth=(user, password), timeout=timeout,
                                 headers={"Accept": "application/json"})

    def incidents(self, query: str = "active=false", limit: int = 10_000, page: int = 1000) -> list[dict]:
        rows, offset = [], 0
        while offset < limit:
            r = self.http.get("/api/now/table/incident", params={
                "sysparm_query": query, "sysparm_limit": page, "sysparm_offset": offset,
                "sysparm_display_value": "true", "sysparm_fields": ",".join(FIELD_MAP)})
            r.raise_for_status()
            batch = r.json()["result"]
            if not batch:
                break
            rows += [{FIELD_MAP[k]: (v.get("display_value") if isinstance(v, dict) else v) for k, v in b.items()} for b in batch]
            offset += page
        return rows
