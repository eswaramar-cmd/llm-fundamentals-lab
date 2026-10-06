"""Test FastMCP Knowledge Base Search tool."""

import pytest
from unittest.mock import patch, MagicMock
from backend.agent.mcp.servers.research_server import research_mcp, knowledge_base_search


def test_knowledge_base_search_direct():
    """Test knowledge_base_search tool directly with mock vector store."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test_ai.pdf", "chunk": 1}
    mock_doc.page_content = "Artificial intelligence is the simulation of human intelligence."

    with patch("backend.agent.tools.knowledge_base._get_vectorstore") as mock_get_vs:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        mock_get_vs.return_value = mock_vs

        result = knowledge_base_search(question="artificial intelligence", k=1)
        assert "test_ai.pdf" in result
        assert "Artificial intelligence" in result


@pytest.mark.asyncio
async def test_knowledge_base_search_mcp_server():
    """Test knowledge_base_search through FastMCP server interface."""
    tools = await research_mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "knowledge_base_search" in tool_names

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "ai_overview.txt", "chunk": 0}
    mock_doc.page_content = "AI refers to machines that can learn and solve problems."

    with patch("backend.agent.tools.knowledge_base._get_vectorstore") as mock_get_vs:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        mock_get_vs.return_value = mock_vs

        result = await research_mcp.call_tool(
            "knowledge_base_search",
            {"question": "What does my document say about artificial intelligence?", "k": 1}
        )
        assert result is not None
        text_content = result[0].text if isinstance(result, list) else str(result)
        assert "ai_overview.txt" in text_content
        assert "AI refers to machines" in text_content
