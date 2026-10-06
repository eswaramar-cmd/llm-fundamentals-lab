"""Test FastMCP Calculator tool."""

import pytest
import asyncio
from backend.agent.mcp.servers.research_server import research_mcp, calculator


def test_calculator_direct():
    """Test calculator tool function directly."""
    result = calculator(expression="25 * 16")
    assert result == "400"

    result = calculator(expression="100 / 4")
    assert result in ("25", "25.0")

    result = calculator(expression="2 ** 8")
    assert result == "256"


@pytest.mark.asyncio
async def test_calculator_mcp_server():
    """Test running calculator through FastMCP server interface."""
    tools = await research_mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "calculator" in tool_names

    # Call tool via FastMCP
    result = await research_mcp.call_tool("calculator", {"expression": "25 * 16"})
    # Result from fastmcp call_tool contains content
    assert result is not None
    text_content = result[0].text if isinstance(result, list) else str(result)
    assert "400" in text_content
