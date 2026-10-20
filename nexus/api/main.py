"""NEXUS API application.

    uvicorn nexus.api.main:app --reload --port 8000
    open http://localhost:8000/           (welcome page)   http://localhost:8000/portal (application)
    http://localhost:8000/docs            (OpenAPI)
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from common.config import ROOT, settings
from common.logging import get_logger, log_event, trace_id_var
from common.tracing import new_trace_id
from nexus.api.routes import api
from nexus.observability import metrics as M

log = get_logger("nexus.api")

app = FastAPI(title="HELIX NEXUS API", version="0.1.0",
              description="Enterprise Engineering & IT Copilot - UC-3 on LEXORA & NEXUS. Dev: POST /v1/dev/token then Authorize.")
app.add_middleware(CORSMiddleware, allow_origins=["*"] if settings().is_dev else [], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def trace_and_metrics(request: Request, call_next):
    tid = request.headers.get("x-trace-id") or new_trace_id()
    trace_id_var.set(tid)
    t0 = time.perf_counter()
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    try:
        resp = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        log.exception("unhandled error")
        resp = JSONResponse({"detail": f"internal error: {exc!r}", "trace_id": tid}, status_code=500)
    endpoint = request.scope.get("route").path if request.scope.get("route") else endpoint
    dt = time.perf_counter() - t0
    M.REQUESTS.labels(endpoint, str(resp.status_code)).inc()
    M.LATENCY.labels(endpoint).observe(dt)
    resp.headers["x-trace-id"] = tid
    if request.url.path.startswith("/v1"):
        log_event(log, "request", path=request.url.path, status=resp.status_code, ms=round(dt * 1000, 1))
    return resp


app.include_router(api)


learning_portal = Path(ROOT) / "HELIX_Engineering_IT_Copilot.html"


@app.get("/healthz", tags=["platform"])
def healthz():
    from nexus.rag.index import index_path

    return {"status": "ok", "env": settings().env, "index": index_path().exists()}


@app.get("/metrics", tags=["platform"])
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(learning_portal)


@app.get("/learning", include_in_schema=False)
def learning():
    return FileResponse(learning_portal)


portal_dir = Path(ROOT) / "clients" / "web-portal"
if portal_dir.exists():
    app.mount("/portal", StaticFiles(directory=portal_dir, html=True), name="portal")
