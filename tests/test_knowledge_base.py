"""Tests for the knowledge_base_search tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.tools.knowledge_base import knowledge_base_search


class TestKnowledgeBaseSearch:
    """Test the knowledge base search tool."""

    @pytest.fixture
    def mock_docs(self):
        """Mock LangChain Document objects."""
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Apex Care is open from 9 AM to 6 PM.",
                metadata={"source": "test.pdf", "chunk": 0},
            ),
            Document(
                page_content="Apex Care is located in Hyderabad.",
                metadata={"source": "test.pdf", "chunk": 1},
            ),
        ]

    def test_tool_name(self):
        assert knowledge_base_search.name == "knowledge_base_search"

    def test_tool_args_schema(self):
        args = knowledge_base_search.args
        assert "question" in args
        assert "k" in args

    @patch("backend.agent.tools.knowledge_base._get_vectorstore")
    def test_successful_search(self, mock_vs, mock_docs):
        mock_vs.return_value.similarity_search.return_value = mock_docs
        result = knowledge_base_search.invoke({"question": "Apex Care location", "k": 2})
        assert "Hyderabad" in result
        assert "9 AM to 6 PM" in result

    @patch("backend.agent.tools.knowledge_base._get_vectorstore")
    def test_no_results(self, mock_vs):
        mock_vs.return_value.similarity_search.return_value = []
        result = knowledge_base_search.invoke({"question": "nothing here"})
        assert "No relevant documents" in result

    @patch("backend.agent.tools.knowledge_base._get_vectorstore")
    def test_chroma_error(self, mock_vs):
        mock_vs.side_effect = ConnectionError("ChromaDB unreachable")
        result = knowledge_base_search.invoke({"question": "test"})
        assert "error" in result.lower()
