from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """One raw unit of source data: a file, a KB article, a ticket, a log file."""

    source: str                     # git | confluence | kb | itsm | logs | postmortem
    path: str                       # repo-relative path or document title
    text: str
    kind: str = "code"              # code | doc | ticket | log
    repo: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8", "ignore")).hexdigest()[:7]

    @property
    def id(self) -> str:
        return f"{self.repo or self.source}:{self.path}"
