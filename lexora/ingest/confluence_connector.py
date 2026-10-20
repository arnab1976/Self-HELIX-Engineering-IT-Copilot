"""Confluence Cloud connector (REST v2). Pages are converted to markdown-ish text for heading-aware chunking."""
from __future__ import annotations

import re

import httpx

from lexora.ingest.base import Document


class ConfluenceConnector:
    def __init__(self, site: str, email: str, api_token: str, timeout: float = 30.0):
        self.http = httpx.Client(base_url=f"https://{site}.atlassian.net/wiki/api/v2", auth=(email, api_token), timeout=timeout)

    def space_pages(self, space_id: str, limit: int = 500) -> list[Document]:
        docs, cursor = [], None
        while len(docs) < limit:
            params = {"space-id": space_id, "body-format": "storage", "limit": 50}
            if cursor:
                params["cursor"] = cursor
            r = self.http.get("/pages", params=params)
            r.raise_for_status()
            data = r.json()
            for p in data["results"]:
                docs.append(Document("confluence", p["title"], html_to_text(p["body"]["storage"]["value"]), "doc",
                                     meta={"page_id": p["id"], "version": p["version"]["number"]}))
            nxt = data.get("_links", {}).get("next")
            if not nxt:
                break
            cursor = re.search(r"cursor=([^&]+)", nxt).group(1)
        return docs


def html_to_text(html: str) -> str:
    html = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n" + "#" * int(m.group(1)) + " " + m.group(2) + "\n", html, flags=re.S)
    html = re.sub(r"<li[^>]*>", "\n- ", html)
    html = re.sub(r"</p>|<br\s*/?>", "\n", html)
    return re.sub(r"<[^>]+>", "", html).strip()
