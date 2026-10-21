"""Dev API: save / list Employee Self-Service portal tickets in PostgreSQL."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import settings
from nexus.api.collection import PortalTicket
from nexus.api import portal_db

portal_tickets_api = APIRouter()
log = logging.getLogger("nexus.api.portal_tickets")


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


class PortalSaveRequest(BaseModel):
    tickets: list[PortalTicket] = Field(default_factory=list, max_length=200)


@portal_tickets_api.get("/lexora/portal/tickets/health", tags=["LEXORA portal"])
def portal_db_health():
    _require_dev()
    try:
        return {"status": "ok", **portal_db.ping()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                f"PostgreSQL unavailable ({type(exc).__name__}: {exc}). "
                "Start Docker Desktop, then: docker compose -f infra/docker-compose.yml --profile stores up -d postgres"
            ),
        ) from exc


@portal_tickets_api.post("/lexora/portal/tickets/save", tags=["LEXORA portal"])
def save_portal_tickets(request: PortalSaveRequest):
    _require_dev()
    if not request.tickets:
        raise HTTPException(400, "No tickets to save. Submit or select at least one portal ticket.")
    try:
        rows = [t.model_dump() for t in request.tickets]
        result = portal_db.upsert_tickets(rows)
        return {
            "status": "saved",
            "saved": result["saved"],
            "ids": result["ids"],
            "store": "postgresql",
            "note": f"Saved {result['saved']} ticket(s) to PostgreSQL portal_tickets table.",
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("portal ticket save failed")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not save to PostgreSQL ({type(exc).__name__}: {exc}). "
                "Ensure Docker Desktop is running and postgres is up on :5432."
            ),
        ) from exc


@portal_tickets_api.get("/lexora/portal/tickets", tags=["LEXORA portal"])
def list_portal_tickets(limit: int = 100):
    _require_dev()
    try:
        rows = portal_db.list_tickets(limit=limit)
        return {"count": len(rows), "tickets": rows, "store": "postgresql"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL unavailable ({type(exc).__name__}: {exc}).",
        ) from exc
