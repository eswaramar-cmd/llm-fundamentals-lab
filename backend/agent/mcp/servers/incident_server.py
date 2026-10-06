"""Incident MCP Server — exposes the investigate_incident tool for production diagnostics."""

from __future__ import annotations

import logging
from typing import Optional
from fastmcp import FastMCP

from backend.agent.tools.investigate_incident import _run_investigation

logger = logging.getLogger(__name__)

# Initialize FastMCP Incident Server
incident_mcp = FastMCP("IncidentMCPServer")


@incident_mcp.tool(
    name="investigate_incident",
    description=(
        "Investigate a production incident dynamically using real logs and metrics. "
        "Correlates error logs, request latencies, LLM/RAG latencies, tool failures, and Redis status. "
        "Returns structured JSON with root-cause candidates, confidence score, observed facts, timeline, "
        "and recommended remediation actions."
    ),
)
def investigate_incident(
    service_name: Optional[str] = None,
    error_message: Optional[str] = None,
    time_range: Optional[str] = None,
    severity: Optional[str] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    details: Optional[str] = None,
) -> str:
    """Run production incident diagnostic.

    Args:
        service_name: Name of the service/component experiencing issues (e.g. 'backend', 'auth', 'llm').
        error_message: Error message or symptom observed.
        time_range: Time window (e.g. '15m', '1h', '24h').
        severity: Severity level (e.g. 'critical', 'high', 'medium', 'low').
        request_id: Specific request or trace ID.
        endpoint: API endpoint path where failures or latency spikes were observed.
        details: Additional context or observations.
    """
    try:
        return _run_investigation(
            service_name=service_name,
            error_message=error_message,
            time_range=time_range,
            severity=severity,
            request_id=request_id,
            endpoint=endpoint,
            details=details,
        )
    except Exception as exc:
        logger.error("Incident investigation MCP tool error: %s", exc)
        return f'{{"error": "Investigation failed: {exc}"}}'


if __name__ == "__main__":
    incident_mcp.run()
