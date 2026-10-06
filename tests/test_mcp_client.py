"""Test LangGraph MCP Client integration and tool discovery."""

import pytest
from backend.agent.mcp.client import get_mcp_client, LangGraphMCPClient


def test_mcp_client_discovery():
    """Verify that MCP client discovers all tools across all MCP servers."""
    client = get_mcp_client()
    tools = client.discover_tools()

    expected_tools = [
        "calculator",
        "knowledge_base_search",
        "web_search",
        "investigate_incident",
        "send_email",
        "save_memory",
    ]

    for tool_name in expected_tools:
        assert tool_name in tools, f"Expected tool '{tool_name}' to be discovered by MCP Client"
        tool_obj = tools[tool_name]
        assert hasattr(tool_obj, "name")
        assert hasattr(tool_obj, "description")
        assert hasattr(tool_obj, "invoke")


def test_mcp_client_intent_filtering():
    """Verify that intent filtering returns the correct tool subsets."""
    client = get_mcp_client()

    calc_tools = client.get_tools_for_intent("calculate")
    assert len(calc_tools) == 1
    assert calc_tools[0].name == "calculator"

    rag_tools = client.get_tools_for_intent("rag")
    assert len(rag_tools) == 1
    assert rag_tools[0].name == "knowledge_base_search"

    incident_tools = client.get_tools_for_intent("incident")
    assert len(incident_tools) == 1
    assert incident_tools[0].name == "investigate_incident"

    email_tools = client.get_tools_for_intent("email")
    assert len(email_tools) == 1
    assert email_tools[0].name == "send_email"

    direct_tools = client.get_tools_for_intent("direct")
    assert len(direct_tools) == 0


def test_mcp_client_execution_calculator():
    """Test executing a tool call through MCP Client."""
    client = get_mcp_client()
    res = client.call_tool_sync("calculator", {"expression": "12 * 12"})
    assert res == "144"


def test_mcp_client_execution_gmail_preview():
    """Test executing gmail send preview through MCP Client."""
    client = get_mcp_client()
    res = client.call_tool_sync(
        "send_email",
        {
            "to_email": "mcp-test@example.com",
            "subject": "MCP Test",
            "body": "Hello from MCP Client test",
            "confirmed": False,
        }
    )
    assert "PREVIEW ONLY" in res
    assert "mcp-test@example.com" in res
