"""Pydantic request / response models for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Request body for POST /chat and POST /chat/stream.

    ``message`` is the current field name. ``question`` is still accepted so
    older callers keep working; the validator folds it into ``message`` so
    every handler only ever reads one attribute.
    """

    message: str | None = Field(default=None, min_length=1, max_length=4096)
    question: str | None = Field(default=None, min_length=1, max_length=4096)
    user_id: str = Field(default="default_user")
    thread_id: str | None = None

    @model_validator(mode="after")
    def _fold_legacy_question(self) -> "ChatRequest":
        if self.message is None:
            self.message = self.question

        if self.message is None:
            raise ValueError("message is required")

        return self


class AgentRunRequest(BaseModel):
    """Request body for POST /agent/run."""

    question: str = Field(..., min_length=1, max_length=4096)
    user_id: str = Field(default="default_user")
    session_id: str | None = None
    tools: list[str] = Field(default_factory=list, max_length=10)
    require_approval: bool = False


class AgentStreamRequest(BaseModel):
    """Request body for POST /agent/stream."""

    question: str = Field(..., min_length=1, max_length=4096)
    user_id: str = Field(default="default_user")
    session_id: str | None = None
    tools: list[str] = Field(default_factory=list, max_length=10)
    require_approval: bool = False


class ApprovalRequest(BaseModel):
    """Request body for POST /agent/{session_id}/approve."""

    approved: bool = True
    feedback: str | None = None


class MemoryEntryRequest(BaseModel):
    """Request body for POST /memory — store a fact for a user."""

    user_id: str = Field(default="default_user")
    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., min_length=1, max_length=4096)
    category: str | None = "preference"


class MemoryEntry(BaseModel):
    """A single long-term memory fact."""

    key: str
    value: str
    category: str
    created_at: str


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = "ok"
    ollama: str = "unknown"
    chroma: str = "unknown"
    redis: str = "unknown"
    version: str = "2.1.0"


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    answer: str
    sources: list[str] | None = None


class AgentRunResponse(BaseModel):
    """Response for POST /agent/run."""

    answer: str
    session_id: str
    approval_required: bool = False
    tool_results: list[dict[str, Any]] | None = None
    sources: list[str] | None = None


class ErrorResponse(BaseModel):
    """Standardised error response."""

    detail: str
    error_code: str = "error"


class IncidentInvestigationRequest(BaseModel):
    """Request model for POST /incidents/investigate."""

    service_name: str | None = Field(default=None, description="Service or component name")
    error_message: str | None = Field(default=None, description="Observed error message or exception")
    time_range: str | None = Field(default=None, description="Time window for incident (e.g. '15m', '1h', '24h')")
    severity: str | None = Field(default=None, description="Severity level ('critical', 'high', 'medium', 'low')")
    request_id: str | None = Field(default=None, description="Specific request or trace ID")
    endpoint: str | None = Field(default=None, description="API endpoint path if known")
    details: str | None = Field(default=None, description="Additional context or user observations")
    user_id: str = Field(default="default_user")
    session_id: str | None = None


class IncidentInvestigationResponse(BaseModel):
    """Structured response model for incident investigation."""

    incident_summary: str
    root_cause: str
    confidence: float
    observed_facts: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any] | str] = Field(default_factory=list)
    timeline: list[dict[str, Any] | str] = Field(default_factory=list)
    root_cause_candidates: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
