"""Production Incident Investigator tool — MCP tool for LangGraph agent.

Collects and correlates real telemetry from application logs and metrics based on
dynamic runtime input provided by the user.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field

from backend.agent.mcp.engine import IncidentInvestigator
from backend.agent.mcp.sources import IncidentQueryParams


class InvestigateIncidentInput(BaseModel):
    """Input schema for the investigate_incident tool."""

    service_name: Optional[str] = Field(
        default=None,
        description="Name of the service or component experiencing issues (e.g. 'backend', 'auth', 'llm', 'rag').",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message, exception name, or symptom observed by the user.",
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time window for incident analysis, e.g. '15m', '1h', '24h', or ISO timestamp.",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity level, e.g. 'critical', 'high', 'medium', 'low'.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Specific request ID or trace ID associated with the error if known.",
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="API endpoint path where failures or latency spikes were observed.",
    )
    details: Optional[str] = Field(
        default=None,
        description="Additional context, symptoms, or user observations about the incident.",
    )


def _run_investigation(
    service_name: Optional[str] = None,
    error_message: Optional[str] = None,
    time_range: Optional[str] = None,
    severity: Optional[str] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    details: Optional[str] = None,
) -> str:
    """Execute the incident investigation engine and serialize result to JSON."""
    params = IncidentQueryParams(
        service_name=service_name,
        error_message=error_message,
        time_range=time_range,
        severity=severity,
        request_id=request_id,
        endpoint=endpoint,
        details=details,
    )
    investigator = IncidentInvestigator()
    result = investigator.investigate(params)
    return json.dumps(result, indent=2)


@lc_tool("investigate_incident", args_schema=InvestigateIncidentInput)
def investigate_incident(
    service_name: Optional[str] = None,
    error_message: Optional[str] = None,
    time_range: Optional[str] = None,
    severity: Optional[str] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    details: Optional[str] = None,
) -> str:
    """Investigate a production incident dynamically using real logs and metrics.

    Correlates error logs, request latencies, LLM/RAG latencies, tool failures, and Redis status.
    Returns structured JSON with root-cause candidates, confidence score, observed facts, timeline,
    and recommended remediation actions.
    """
    return _run_investigation(
        service_name=service_name,
        error_message=error_message,
        time_range=time_range,
        severity=severity,
        request_id=request_id,
        endpoint=endpoint,
        details=details,
    )
