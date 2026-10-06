"""Model Context Protocol (MCP) server interface and tool definitions.

Standardizes investigate_incident as an MCP tool conforming to the MCP schema specification.
"""

from __future__ import annotations

import json
from typing import Any

from backend.agent.mcp.engine import IncidentInvestigator
from backend.agent.mcp.sources import IncidentQueryParams

INVESTIGATE_INCIDENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "investigate_incident",
    "description": (
        "Investigate a production incident dynamically. Analyzes existing application logs "
        "and metrics to correlate errors, latencies, tool failures, and Redis status. "
        "Returns root-cause ranking, confidence, observed facts, timeline, and remediation recommendations."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "service_name": {
                "type": "string",
                "description": "Name of the service or component experiencing an incident.",
            },
            "error_message": {
                "type": "string",
                "description": "Error message, symptom, or exception string observed.",
            },
            "time_range": {
                "type": "string",
                "description": "Time range or window for the incident (e.g. '15m', '1h', '24h', or ISO timestamp).",
            },
            "severity": {
                "type": "string",
                "description": "Severity level (e.g. 'critical', 'high', 'medium', 'low').",
            },
            "request_id": {
                "type": "string",
                "description": "Optional request ID or trace ID associated with the failure.",
            },
            "endpoint": {
                "type": "string",
                "description": "Optional API endpoint path where errors occurred.",
            },
            "details": {
                "type": "string",
                "description": "Additional context or user explanation of the issue.",
            },
        },
        "required": [],
    },
}


class MCPServer:
    """Lightweight Model Context Protocol handler for incident investigation."""

    def __init__(self, investigator: IncidentInvestigator | None = None) -> None:
        self.investigator = investigator or IncidentInvestigator()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tools catalog."""
        return [INVESTIGATE_INCIDENT_TOOL_SCHEMA]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool by name."""
        if name != "investigate_incident":
            raise ValueError(f"Unknown MCP tool: {name}")

        params = IncidentQueryParams(
            service_name=arguments.get("service_name"),
            error_message=arguments.get("error_message"),
            time_range=arguments.get("time_range"),
            severity=arguments.get("severity"),
            request_id=arguments.get("request_id"),
            endpoint=arguments.get("endpoint"),
            details=arguments.get("details"),
            user_id=arguments.get("user_id", "default_user"),
        )
        return self.investigator.investigate(params)
