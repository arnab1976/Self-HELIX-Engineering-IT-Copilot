"""Load the sample corpus from data/sample (repos, docs, KB, postmortems). Same output as the real connectors."""
from __future__ import annotations

from pathlib import Path

from common.config import settings
from lexora.ingest.base import Document

CODE_EXT = {".py", ".ts", ".js", ".go", ".java", ".proto", ".tf", ".yaml", ".yml", ".sql"}


def load_repos(root: Path | None = None) -> list[Document]:
    """data/sample/repos/<org>/<repo>/**  ->  Document(repo="<org>/<repo>")."""
    root = root or Path(settings().data_dir) / "sample" / "repos"
    docs: list[Document] = []
    for org in sorted(p for p in root.iterdir() if p.is_dir()):
        for repo in sorted(p for p in org.iterdir() if p.is_dir()):
            name = f"{org.name}/{repo.name}"
            for f in sorted(repo.rglob("*")):
                if f.is_file() and f.suffix in CODE_EXT:
                    docs.append(Document("git", str(f.relative_to(repo)), f.read_text(encoding="utf-8"), "code", name,
                                         {"language": f.suffix.lstrip(".")}))
    return docs


def load_markdown(folder: str, source: str, repo: str) -> list[Document]:
    root = Path(settings().data_dir) / "sample" / folder
    return [Document(source, f.name, f.read_text(encoding="utf-8"), "doc", repo) for f in sorted(root.glob("*.md"))]


def load_all() -> list[Document]:
    return (load_repos()
            + load_markdown("docs", "confluence", "docs/engineering-handbook")
            + load_markdown("kb", "kb", "docs/engineering-handbook")
            + load_markdown("postmortems", "postmortem", "docs/engineering-handbook"))
