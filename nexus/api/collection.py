"""Dev-only API for the LEXORA engineering-data collection agent."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings
from lexora.ingest.collection_agent import collect_and_register, source_catalog

collection_api = APIRouter()
log = logging.getLogger("nexus.api.collection")


class SourceCredentials(BaseModel):
    """Optional per-request connector credentials for the learning-portal dialogs (dev only)."""

    github_token: str = Field(default="", max_length=500)
    github_repo: str = Field(default="", max_length=200)
    github_ref: str = Field(default="HEAD", max_length=200)
    servicenow_instance: str = Field(default="", max_length=200)
    servicenow_user: str = Field(default="", max_length=200)
    servicenow_password: str = Field(default="", max_length=500)
    confluence_site: str = Field(default="", max_length=200)
    confluence_email: str = Field(default="", max_length=320)
    confluence_api_token: str = Field(default="", max_length=500)
    confluence_space_id: str = Field(default="", max_length=200)
    loki_url: str = Field(default="", max_length=500)
    loki_query: str = Field(default="", max_length=2_000)


class CollectionRequest(BaseModel):
    mode: Literal["simulation", "connected"] = "simulation"
    sources: list[Literal["code", "tickets", "knowledge", "logs", "observability"]] = Field(
        default=["code", "tickets", "knowledge", "logs", "observability"], min_length=1, max_length=5
    )
    sample_text: str = Field(default="", max_length=30_000)
    filename: str = Field(default="snippet.py", max_length=100)
    github_repo: str = Field(default="", max_length=200)
    credentials: SourceCredentials | None = None
    dedupe_threshold: float = Field(default=0.82, ge=0.5, le=0.99)


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


@collection_api.get("/lexora/collect/sources", tags=["LEXORA collection"])
def collection_sources():
    _require_dev()
    return source_catalog()


@collection_api.post("/lexora/collect/run", tags=["LEXORA collection"])
def run_collection(request: CollectionRequest):
    _require_dev()
    try:
        payload = request.model_dump()
        credentials = payload.pop("credentials", None) or {}
        if not payload.get("github_repo") and credentials.get("github_repo"):
            payload["github_repo"] = credentials["github_repo"]
        return collect_and_register(**payload, credentials=credentials)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        log.exception("Collection pipeline source data is unavailable")
        raise HTTPException(status_code=503, detail="Selected simulation data is unavailable. Generate the sample data and retry.") from exc
