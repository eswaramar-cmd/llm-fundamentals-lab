"""End-to-End Test Suite for MCP Tools and LangGraph Integration.

Tests all 5 required tools through the MCP architecture:
1. Calculator
2. Knowledge Base Search
3. Web Search
4. Incident Diagnostics
5. Gmail Automation (Safe Preview)
6. FastAPI MCP endpoint
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.agent.mcp.client import get_mcp_client
from backend.agent.tools.web_search import SearchResponse, SearchResult


client = TestClient(app)


def test_e2e_fastapi_mcp_tools_endpoint():
    """Verify GET /api/mcp/tools endpoint returns all discovered tools."""
    response = client.get("/api/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert "mcp_servers" in data
    assert "tools" in data
    assert set(data["mcp_servers"]) >= {"research", "incident", "gmail"}

    tool_names = [t["name"] for t in data["tools"]]
    assert "calculator" in tool_names
    assert "knowledge_base_search" in tool_names
    assert "web_search" in tool_names
    assert "investigate_incident" in tool_names
    assert "send_email" in tool_names


def test_e2e_calculator_tool():
    """E2E Test 1: Calculator — 'What is 25 * 16?'."""
    mcp_client = get_mcp_client()
    result = mcp_client.call_tool_sync("calculator", {"expression": "25 * 16"})
    assert result == "400"


def test_e2e_knowledge_base_search_tool():
    """E2E Test 2: Knowledge Base — 'What does my document say about artificial intelligence?'."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "ai_research.pdf", "chunk": 1}
    mock_doc.page_content = "Artificial intelligence enables machines to learn from experience."

    with patch("backend.agent.tools.knowledge_base._get_vectorstore") as mock_get_vs:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        mock_get_vs.return_value = mock_vs

        mcp_client = get_mcp_client()
        result = mcp_client.call_tool_sync(
            "knowledge_base_search",
            {"question": "What does my document say about artificial intelligence?", "k": 1}
        )
        assert "ai_research.pdf" in result
        assert "Artificial intelligence" in result


def test_e2e_web_search_tool():
    """E2E Test 3: Web search — 'What is the latest AI news?'."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = SearchResponse(
        query="latest AI news",
        results=[
            SearchResult(
                title="Latest Advancements in AI",
                url="https://news.example.com/ai",
                snippet="Breakthroughs in reasoning and efficiency announced today."
            )
        ]
    )

    with patch("backend.agent.tools.web_search.get_search_provider", return_value=mock_provider):
        mcp_client = get_mcp_client()
        result = mcp_client.call_tool_sync(
            "web_search",
            {"query": "What is the latest AI news?", "num_results": 1}
        )
        assert "Latest Advancements in AI" in result
        assert "https://news.example.com/ai" in result


def test_e2e_incident_diagnostics_tool():
    """E2E Test 4: Incident diagnostics — Test valid diagnostic request."""
    mcp_client = get_mcp_client()
    result = mcp_client.call_tool_sync(
        "investigate_incident",
        {"service_name": "backend", "endpoint": "/api/chat", "time_range": "1h"}
    )
    assert result is not None
    data = json.loads(result)
    assert isinstance(data, dict)
    assert "incident_summary" in data or "findings" in data or "root_causes" in data or "status" in data


def test_e2e_gmail_safe_preview():
    """E2E Test 5: Gmail — Safe non-destructive preview operation."""
    mcp_client = get_mcp_client()
    result = mcp_client.call_tool_sync(
        "send_email",
        {
            "to_email": "user@example.com",
            "subject": "Investigation Summary",
            "body": "Investigation completed successfully.",
            "confirmed": False,
        }
    )
    assert "PREVIEW ONLY" in result
    assert "user@example.com" in result
    assert "Investigation Summary" in result
