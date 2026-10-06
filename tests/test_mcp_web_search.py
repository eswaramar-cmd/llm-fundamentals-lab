"""Test FastMCP Web Search tool."""

import pytest
from unittest.mock import MagicMock
from backend.agent.mcp.servers.research_server import research_mcp, web_search
from backend.agent.tools.web_search import SearchResponse, SearchResult


def test_web_search_direct():
    """Test web_search tool directly with mock provider."""
    mock_provider = MagicMock()
    mock_provider.search.return_value = SearchResponse(
        query="latest AI news",
        results=[
            SearchResult(
                title="New LLM Released",
                url="https://example.com/news",
                snippet="A new state of the art language model has been released."
            )
        ]
    )

    from backend.agent.tools.web_search import perform_web_search
    result = perform_web_search("latest AI news", num_results=1, provider=mock_provider)
    assert "New LLM Released" in result
    assert "https://example.com/news" in result


@pytest.mark.asyncio
async def test_web_search_mcp_server():
    """Test web_search through FastMCP server interface."""
    tools = await research_mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "web_search" in tool_names

    mock_provider = MagicMock()
    mock_provider.search.return_value = SearchResponse(
        query="fastmcp python",
        results=[
            SearchResult(
                title="FastMCP Documentation",
                url="https://github.com/jlowin/fastmcp",
                snippet="FastMCP is a fast Python framework for Model Context Protocol."
            )
        ]
    )

    from unittest.mock import patch
    with patch("backend.agent.tools.web_search.get_search_provider", return_value=mock_provider):
        result = await research_mcp.call_tool(
            "web_search",
            {"query": "fastmcp python", "num_results": 1}
        )
        assert result is not None
        text_content = result[0].text if isinstance(result, list) else str(result)
        assert "FastMCP Documentation" in text_content
        assert "https://github.com/jlowin/fastmcp" in text_content
