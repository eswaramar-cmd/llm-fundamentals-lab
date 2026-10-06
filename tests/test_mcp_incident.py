"""Test FastMCP Incident Diagnostics tool."""

import json
import pytest
from backend.agent.mcp.servers.incident_server import incident_mcp, investigate_incident


def test_investigate_incident_direct():
    """Test investigate_incident directly."""
    result = investigate_incident(service_name="backend", endpoint="/api/chat", time_range="1h")
    assert result is not None
    data = json.loads(result)
    assert "findings" in data or "root_causes" in data or "observed_facts" in data or "telemetry" in data or "status" in data or "incident_id" in data


@pytest.mark.asyncio
async def test_investigate_incident_mcp_server():
    """Test investigate_incident through FastMCP server interface."""
    tools = await incident_mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "investigate_incident" in tool_names

    result = await incident_mcp.call_tool(
        "investigate_incident",
        {"service_name": "backend", "endpoint": "/api/chat", "time_range": "1h"}
    )

    if hasattr(result, "content") and result.content:
        text_content = result.content[0].text
    elif isinstance(result, list) and result:
        text_content = result[0].text if hasattr(result[0], "text") else str(result[0])
    else:
        text_content = str(result)

    assert len(text_content) > 10
    parsed = json.loads(text_content)
    assert isinstance(parsed, dict)
    assert "incident_summary" in parsed or "findings" in parsed or "root_causes" in parsed or "status" in parsed

