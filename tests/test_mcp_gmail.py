"""Test FastMCP Gmail tool (safe preview mode)."""

import pytest
from backend.agent.mcp.servers.gmail_server import gmail_mcp, send_email


def test_gmail_preview_direct():
    """Test send_email preview mode directly."""
    result = send_email(
        to_email="test@example.com",
        subject="Test MCP Email",
        body="Hello from MCP test",
        confirmed=False,
    )
    assert "PREVIEW ONLY" in result
    assert "To: test@example.com" in result
    assert "Subject: Test MCP Email" in result
    assert "Body: Hello from MCP test" in result


@pytest.mark.asyncio
async def test_gmail_mcp_server():
    """Test send_email preview mode through FastMCP server interface."""
    tools = await gmail_mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "send_email" in tool_names

    result = await gmail_mcp.call_tool(
        "send_email",
        {
            "to_email": "alice@example.com",
            "subject": "System Status Update",
            "body": "All systems operational.",
            "confirmed": False,
        }
    )
    assert result is not None
    if hasattr(result, "content") and result.content:
        text_content = result.content[0].text
    elif isinstance(result, list) and result:
        text_content = result[0].text if hasattr(result[0], "text") else str(result[0])
    else:
        text_content = str(result)

    assert "PREVIEW ONLY" in text_content
    assert "alice@example.com" in text_content
    assert "System Status Update" in text_content
