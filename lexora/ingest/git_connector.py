"""GitHub connector (REST v3). Pulls files, PRs and review threads for LEXORA training pairs.

Usage:
    HELIX_GITHUB_TOKEN=... python -m lexora.ingest.git_connector your-org/your-repo --out var/raw/git
Production adds GitLab/Bitbucket equivalents and incremental sync via webhooks.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import httpx

from common.config import settings
from lexora.ingest.base import Document

API = "https://api.github.com"


class GitHubConnector:
    def __init__(self, token: str | None = None, timeout: float = 30.0):
        token = token or settings().github_token
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=API, headers=headers, timeout=timeout)

    def _paged(self, url: str, params: dict | None = None):
        params = {**(params or {}), "per_page": 100}
        while url:
            r = self.http.get(url, params=params)
            r.raise_for_status()
            yield from r.json()
            url = r.links.get("next", {}).get("url")
            params = None

    def files(self, repo: str, ref: str = "HEAD",
              exts: tuple[str, ...] = (".py", ".ts", ".go", ".java", ".md", ".proto", ".tf", ".yaml", ".yml", ".sql"),
              limit: int = 500) -> list[Document]:
        tree = self.http.get(f"/repos/{repo}/git/trees/{ref}", params={"recursive": 1}).raise_for_status().json()
        docs = []
        for node in tree.get("tree", []):
            if node["type"] == "blob" and node["path"].endswith(exts) and node.get("size", 0) < 200_000:
                blob = self.http.get(f"/repos/{repo}/git/blobs/{node['sha']}").raise_for_status().json()
                text = base64.b64decode(blob["content"]).decode("utf-8", "ignore")
                kind = "doc" if node["path"].endswith(".md") else "code"
                docs.append(Document("git", node["path"], text, kind, repo, {"sha": node["sha"]}))
                if len(docs) >= limit:
                    break
        return docs

    def pull_requests(self, repo: str, state: str = "closed", limit: int = 200) -> list[dict]:
        """PR description <-> diff pairs (L2 training) and review threads (L5 SFT/DPO)."""
        out = []
        for pr in self._paged(f"/repos/{repo}/pulls", {"state": state}):
            if len(out) >= limit:
                break
            response = self.http.get(f"/repos/{repo}/pulls/{pr['number']}", headers={"Accept": "application/vnd.github.diff"})
            response.raise_for_status()
            diff = response.text
            comments = list(self._paged(f"/repos/{repo}/pulls/{pr['number']}/comments"))
            out.append({"number": pr["number"], "title": pr["title"], "body": pr.get("body") or "", "diff": diff,
                        "merged": bool(pr.get("merged_at")),
                        "review_comments": [{"path": c["path"], "line": c.get("line"), "body": c["body"]} for c in comments]})
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", default="var/raw/git")
    a = ap.parse_args()
    gh = GitHubConnector()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = gh.files(a.repo)
    (out / f"{a.repo.replace('/', '__')}_files.jsonl").write_text("\n".join(json.dumps(d.__dict__) for d in docs))
    prs = gh.pull_requests(a.repo)
    (out / f"{a.repo.replace('/', '__')}_prs.jsonl").write_text("\n".join(json.dumps(p) for p in prs))
    print(f"{len(docs)} files, {len(prs)} PRs -> {out}")


if __name__ == "__main__":
    main()
