"""Request/response models for the /v1 API (spec 11)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    surface: Literal["ide", "chat", "portal"] = "portal"
    mode: Literal["auto", "code", "knowledge", "it"] = "auto"
    repo: str | None = Field(None, description="repository scope, e.g. payments/checkout-service")
    stream: bool = False
    redact_secrets: bool = False
    complexity: float = 0.3
    thread_id: str | None = None


class CompleteRequest(BaseModel):
    repo: str | None = None
    path: str = "untitled.py"
    language: str = "python"
    prefix: str
    suffix: str = ""
    max_tokens: int = 64


class TriageRequest(BaseModel):
    title: str
    description: str = ""
    requester: str | None = None
    ci: str = ""
    create: bool = True


class ReviewRequest(BaseModel):
    repo: str = "payments/checkout-service"
    pr: int | None = 4182
    diff: str | None = None


class QueryRequest(BaseModel):
    question: str
    target: Literal["sql", "promql", "logql", "kql"] = "sql"
    execute: bool = True
    query: str | None = Field(None, description="optional edited query to validate/execute instead of generating")


class IncidentRequest(BaseModel):
    alert: dict = Field(default_factory=lambda: {"name": "CheckoutErrorRateHigh", "service": "checkout-service", "severity": "SEV-2"})


class ApproveRequest(BaseModel):
    decision: Literal["approve", "reject", "modify"] = "approve"
    params: dict | None = Field(None, description="modified parameters (decision=modify)")
    comment: str = ""


class StepUpRequest(BaseModel):
    thread_id: str
    approved: bool = True


class ChangeRequest(BaseModel):
    services_touched: int = 3
    lines_changed: int = 250
    test_coverage: float = 55
    author_failure_rate: float = 10
    dependency_fanout: int = 9
    peak_window: int = 1
    has_rollback_plan: int = 0
    change_type: Literal["standard", "normal", "emergency"] = "normal"


class FeedbackRequest(BaseModel):
    trace_id: str
    signal: Literal["accept", "dismiss", "thumbs_up", "thumbs_down", "edit"]
    target: str = "answer"
    payload: dict = Field(default_factory=dict)


class RouteRequest(BaseModel):
    route: str = "code_q"
    repo: str | None = None
    complexity: float = 0.3
    context_tokens: int = 4000
